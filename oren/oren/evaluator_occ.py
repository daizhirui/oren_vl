from typing import Callable, Dict, List, Optional

from tqdm import tqdm

from oren import MarchingCubes, np, o3d, torch
from oren.evaluator_base import EvaluatorBase
from oren.occ_network import OccNetwork, OccNetworkConfig


class OccEvaluator(EvaluatorBase):
    """Evaluator for OccNetwork.

    Sign convention: occupancy logits where logit > 0 = occupied, logit < 0 = free, logit = 0 = decision boundary.
    Marching cubes runs on negated logits so the SDF-like sign convention (positive = outside) is recovered. The
    iso_value argument stays in the original occupancy sign and extract_mesh negates both the field and the iso so a
    config of `mesh_iso_value: +8.0` traces the surface at occupancy logit=8 (deep in the occupied region),
    `mesh_iso_value: 0.0` at the decision boundary, and `mesh_iso_value: -1.0` at slightly-free.
    """

    def __init__(
        self,
        batch_size: int,
        clean_mesh: bool = True,
        model_cfg: OccNetworkConfig | None = None,
        model: torch.nn.Module | None = None,
        model_path: str | None = None,
        device: str = "cuda",
        interactive: bool = False,
    ):
        """Build an OccNetwork evaluator.

        Args:
            batch_size: per-batch point count for model inference; non-positive disables batching.
            clean_mesh: if True, filter mesh-extraction grid vertices through the octree's voxel-size mask.
            model_cfg: network configuration; required when `model` is not provided so the model can be re-created.
            model: optional already-instantiated OccNetwork; if provided, `model_path` is ignored.
            model_path: optional checkpoint path; loaded via `create_model` when `model` is None.
            device: torch device for inference.
            interactive: if True, enable interactive matplotlib visualizations during evaluation.
        """
        self.batch_size = batch_size
        self.clean_mesh = clean_mesh
        self.model_cfg = model_cfg

        super().__init__(
            self.forward_model,
            model,
            model_path,
            self.create_model,
            device=device,
            absolute_sdf=False,  # unused for occupancy metrics
            interactive=interactive,
        )

        self.model: OccNetwork

    def create_model(self, model_path: str) -> torch.nn.Module:
        """Construct an OccNetwork from `self.model_cfg` and load its weights.

        Args:
            model_path: path to a saved state_dict.

        Returns:
            OccNetwork moved to `self.device` and put in eval mode.
        """
        assert self.model_cfg is not None
        tqdm.write("Creating OccNetwork...")
        model = OccNetwork(self.model_cfg)
        model.to(self.device)
        tqdm.write(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def forward_model(
        self,
        model,
        points: torch.Tensor,
        get_grad: bool,
        auto_grad: bool = True,
        finite_diff_eps: float = 0.01,
        prior_only: bool = False,
        device: str = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward the model with the given points.

        Returns dict with keys:
            occ_prior:     (...,) prior occupancy logits
            occ_residual:  (...,) residual occupancy logits (None entries when prior_only)
            occ:           (...,) final occupancy logits (prior + residual)
            grad:          dict with occ_prior and occ gradients (only if get_grad)
        """
        if points.shape[0] == 0:
            return dict()

        if self.batch_size <= 0:
            bs = points.shape[0]
        else:
            bs = int(self.batch_size * 3 * points.shape[0] / points.numel()) + 1

        occ_prior = []
        occ_residual = []
        occ_pred = []
        occ_prior_grad = []
        occ_grad = []

        for i in tqdm(range(0, points.shape[0], bs), desc="Batches", ncols=120, position=1, leave=False):
            j = min(i + bs, points.shape[0])
            points_batch = points[i:j].to(self.device)
            points_batch.requires_grad_(auto_grad)
            out_batch = model(points_batch, prior_only=prior_only)
            occ_pred_batch = out_batch.pred.squeeze(-1)
            occ_prior_batch = (
                out_batch.prior.squeeze(-1) if out_batch.prior is not None else torch.zeros_like(occ_pred_batch)
            )
            occ_residual_batch = (
                out_batch.implicit.squeeze(-1) if out_batch.implicit is not None else torch.zeros_like(occ_pred_batch)
            )

            if get_grad:
                if auto_grad:
                    occ_prior_grad_batch = torch.autograd.grad(
                        outputs=[occ_prior_batch],
                        inputs=[points_batch],
                        grad_outputs=[torch.ones_like(occ_prior_batch)],
                        create_graph=True,
                        allow_unused=True,
                    )[0]
                    occ_prior_grad.append(occ_prior_grad_batch.detach().cpu())
                    occ_grad_batch = torch.autograd.grad(
                        outputs=[occ_prior_batch + occ_residual_batch],
                        inputs=[points_batch],
                        grad_outputs=[torch.ones_like(occ_pred_batch)],
                        create_graph=True,
                        allow_unused=True,
                    )[0]
                    occ_grad.append(occ_grad_batch.detach().cpu())
                else:
                    occ_grad_batch = torch.empty_like(points_batch)
                    occ_prior_grad_batch = torch.empty_like(points_batch)
                    for k in range(3):
                        offset = torch.zeros((3,), device=points_batch.device)
                        offset[k] = finite_diff_eps
                        offset = offset.view(*[1] * (points_batch.ndim - 1), 3)
                        out_plus = model(points_batch + offset, prior_only=prior_only)
                        out_minus = model(points_batch - offset, prior_only=prior_only)
                        occ_plus = out_plus.pred.squeeze(-1)
                        occ_minus = out_minus.pred.squeeze(-1)
                        occ_prior_plus = (
                            out_plus.prior.squeeze(-1) if out_plus.prior is not None else torch.zeros_like(occ_plus)
                        )
                        occ_prior_minus = (
                            out_minus.prior.squeeze(-1) if out_minus.prior is not None else torch.zeros_like(occ_minus)
                        )
                        occ_prior_grad_batch[..., k] = (occ_prior_plus - occ_prior_minus) / (2 * finite_diff_eps)
                        occ_grad_batch[..., k] = (occ_plus - occ_minus) / (2 * finite_diff_eps)
                    occ_prior_grad.append(occ_prior_grad_batch.detach().cpu())
                    occ_grad.append(occ_grad_batch.detach().cpu())

            occ_prior.append(occ_prior_batch.detach().cpu())
            occ_residual.append(occ_residual_batch.detach().cpu())
            occ_pred.append(occ_pred_batch.detach().cpu())

        if device is None:
            device = self.device
        if len(occ_prior) == 1:
            occ_prior = occ_prior[0].to(device)
            occ_residual = occ_residual[0].to(device)
            occ_pred = occ_pred[0].to(device)
            if get_grad:
                occ_prior_grad = occ_prior_grad[0].to(device)
                occ_grad = occ_grad[0].to(device)
        else:
            occ_prior = torch.cat(occ_prior, dim=0).to(device)
            occ_residual = torch.cat(occ_residual, dim=0).to(device)
            occ_pred = torch.cat(occ_pred, dim=0).to(device)
            if get_grad:
                occ_prior_grad = torch.cat(occ_prior_grad, dim=0).to(device)
                occ_grad = torch.cat(occ_grad, dim=0).to(device)

        result = dict(occ_prior=occ_prior, occ_residual=occ_residual, occ=occ_pred)
        if get_grad:
            result["grad"] = dict(occ_prior=occ_prior_grad, occ=occ_grad)
        return result

    @torch.no_grad()
    def extract_mesh(
        self,
        bound_min: List[float],
        bound_max: List[float],
        grid_resolution: float,
        fields: Optional[List[str]] = None,
        iso_value: float = 0.0,
        grid_vertex_filter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """Run marching cubes on negated occupancy logits to recover the iso-surface.

        See class docstring for the sign-convention rationale. Mirrors EvaluatorBase.extract_mesh but applies negation
        to the field values before marching cubes is invoked.

        Args:
            bound_min: (3,) lower corner of the extraction grid in world coordinates.
            bound_max: (3,) upper corner of the extraction grid in world coordinates.
            grid_resolution: spacing of the extraction grid.
            fields: list of occupancy fields to extract; defaults to `["occ_prior", "occ"]`.
            iso_value: iso-value in occupancy-logit sign; internally negated to match the flipped field.
            grid_vertex_filter: optional callable mapping (N, 3) points to a boolean keep-mask; defaults to the
                octree's `grid_vertex_filter` when `clean_mesh` is True.

        Returns:
            List of `o3d.geometry.TriangleMesh`, one per entry in `fields`; empty meshes when the grid filter rejects
            every vertex.
        """
        if self.clean_mesh and grid_vertex_filter is None:
            grid_vertex_filter = self.model.octree.grid_vertex_filter

        if fields is None:
            fields = ["occ_prior", "occ"]

        self.model.eval()

        occ_grid = self.extract_field_grid(
            bound_min=bound_min,
            bound_max=bound_max,
            grid_resolution=grid_resolution,
            grid_vertex_filter=grid_vertex_filter,
        )

        if not occ_grid:
            return [o3d.geometry.TriangleMesh() for _ in fields]

        mask = None
        if grid_vertex_filter is not None:
            mask = occ_grid["mask"].cpu().numpy().astype(np.bool_)

        meshes: List[o3d.geometry.TriangleMesh] = []
        for field in fields:
            assert field in occ_grid, f"Field {field} not found in model output"
            values = -occ_grid[field].cpu().numpy().astype(np.float64)  # negate: positive logit -> negative SDF-like
            mc = MarchingCubes()

            if mask is None:
                grid_values = values
            else:
                grid_shape = occ_grid["grid_shape"].cpu().numpy().astype(np.int32)
                grid_values = np.ones(grid_shape, dtype=np.float64)  # unmasked cells: "outside" in SDF-like sign
                grid_values[mask] = values

            vertices, triangles, triangle_normals = mc.run(
                coords_min=bound_min,
                grid_res=[grid_resolution, grid_resolution, grid_resolution],
                grid_shape=grid_values.shape,
                grid_values=grid_values.flatten(),
                mask=mask.flatten() if mask is not None else None,
                # iso_value is given in occupancy sign: flip it to match the negated field.
                iso_value=-iso_value,
                row_major=True,
                parallel=True,
            )

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices.T)
            mesh.triangles = o3d.utility.Vector3iVector(triangles.T)
            mesh.triangle_normals = o3d.utility.Vector3dVector(triangle_normals.T)

            meshes.append(mesh)

        return meshes
