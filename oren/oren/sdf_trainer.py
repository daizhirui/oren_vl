import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oren import torch
from oren.evaluator_oren import OrenEvaluator
from oren.frame import Frame
from oren.sdf_criterion import SdfCriterion
from oren.sdf_network import SdfNetwork
from oren.trainer_base import TrainerBase
from oren.trainer_config import TrainerConfig
from oren.utils.registry import get_identifier, register_trainer
from oren.utils.sampling import generate_sdf_samples


@register_trainer
class SdfTrainer(TrainerBase):
    """Trainer for SdfNetwork. Computes SDF gradients (autodiff or finite difference) for the eikonal-aware criterion,
    and extracts SDF-named meshes/slices for evaluation."""

    model: SdfNetwork
    criterion: SdfCriterion
    evaluator: OrenEvaluator

    def __init__(self, cfg: TrainerConfig, data_stream=None):
        assert cfg.model_identifier == get_identifier(
            SdfNetwork
        ), f"SdfTrainer requires SdfNetwork, got {cfg.model_identifier}"
        assert cfg.criterion_identifier == get_identifier(
            SdfCriterion
        ), f"SdfTrainer requires SdfCriterion, got {cfg.criterion_identifier}"

        super().__init__(cfg, data_stream=data_stream)

    def _create_evaluator(self):
        return OrenEvaluator(
            batch_size=self.cfg.batch_size,
            clean_mesh=self.cfg.clean_mesh,
            model_cfg=self.cfg.model,
            model=self.model,
            device=self.cfg.device,
        )

    def train_with_frame(self, frame: Frame | None):
        """Run `num_iterations` SDF training iterations against the selected key frames and `frame`.

        Args:
            frame: current Frame to also sample from; if None, only key-frame samples are used.

        Returns:
            True if training should continue, False if interrupted by `training_frame_start_callback`.
        """
        self.num_iterations = self.cfg.num_iterations_per_frame
        if self.epoch < self.cfg.num_init_frames:
            self.num_iterations = self.cfg.init_frame_iterations

        if self.training_frame_start_callback is not None:
            if not self.training_frame_start_callback(self, frame):
                self.logger.info("Training interrupted by callback, exiting.")
                return False  # exit training

        with self.timer_select_key_frames:
            self.selected_key_frame_indices = self.key_frame_set.select_key_frames()
        self._sample_and_train(current_frame=frame, key_frame_indices=self.selected_key_frame_indices)
        return True

    def train_with_frames(self, frames: list[Frame]) -> bool:
        """Offline-epoch step. Mirrors :meth:`VlTrainer.train_with_frames`: splice the batch in as a temporary
        key-frame set so every batch frame contributes samples, run the inner sample-and-train unit once, then
        restore. Scatter mode short-circuits per-frame via :meth:`train_with_frame`.

        Args:
            frames: a batch of frames produced by :class:`TrainerBase._train_bounded_offline`.

        Returns:
            True to keep training; False if a callback short-circuits via :meth:`train_with_frame` (scatter mode).
        """
        if not frames:
            return True

        if self.cfg.mode == "scatter":
            for f in frames:
                if not self.train_with_frame(f):
                    return False
            return True

        self.num_iterations = self.cfg.num_iterations_per_frame
        if self.epoch < self.cfg.num_init_frames:
            self.num_iterations = self.cfg.init_frame_iterations

        saved_frames = self.key_frame_set.frames
        self.key_frame_set.frames = list(frames)
        batch_indices = list(range(len(frames)))
        try:
            self._sample_and_train(current_frame=None, key_frame_indices=batch_indices)
        finally:
            self.key_frame_set.frames = saved_frames
        return True

    def _sample_and_train(self, current_frame: Frame | None, key_frame_indices: list[int]) -> None:
        """Sample rays once from `key_frame_indices` plus `current_frame`, generate SDF samples, then run
        `self.num_iterations` optimizer steps over those samples.
        """
        with self.timer_sample_rays:
            rays_o_all, rays_d_all, depth_samples_all = self.key_frame_set.sample_by_num(
                total_num_samples=self.cfg.num_rays_total,
                sample_frame_fn=Frame.sample_rays,
                key_frame_indices=key_frame_indices,
                current_frame=current_frame,
                device=self.cfg.device,
            )

            if self.cfg.extra_surface_sample:
                self.extra_surface_pcd = self.key_frame_set.sample_by_ratio(
                    ratio=1.0 / self.cfg.frame_downsample,
                    sample_frame_fn=Frame.sample_points,
                    key_frame_indices=key_frame_indices,
                    current_frame=current_frame,
                )

        with self.timer_generate_sdf_samples:
            self.samples = generate_sdf_samples(
                rays_d_all=rays_d_all,
                rays_o_all=rays_o_all,
                depth_samples_all=depth_samples_all,
                cfg=self.cfg.sample_rays,
                extra_surface_pcd=self.extra_surface_pcd,
                device=self.cfg.device,
            )

            mask = torch.ones_like(self.samples.sampled_xyz[..., 0], dtype=torch.bool)
            if not self.cfg.data.apply_bound:
                mask = mask & (self.samples.sampled_xyz >= self.bound_min).all(dim=-1)
                mask = mask & (self.samples.sampled_xyz <= self.bound_max).all(dim=-1)

        # ---- scatter mode ----
        if self.cfg.mode == "scatter":
            self._scatter_samples(mask)
            return

        # ---- optimization mode ----
        # Query the model on the sampled points in batches, compute losses and gradients, and step the optimizer.

        # self.samples.sampled_xyz: (n, m, 3)
        num_rays = self.samples.sampled_xyz.shape[0]
        if self.cfg.grad_method == "autodiff":
            self.samples.sampled_xyz.requires_grad_(True)
        else:
            with self.timer_compute_offset_points:
                (
                    offset_points_plus,
                    offset_points_minus,
                ) = self.compute_offset_points_for_finite_diff(self.samples.sampled_xyz)
            with self.timer_find_voxel_indices_offset_points:
                voxel_indices_plus = self.find_voxel_indices(offset_points_plus)  # (n, m, 3)
                voxel_indices_minus = self.find_voxel_indices(offset_points_minus)  # (n, m, 3)
        with self.timer_find_voxel_indices_sampled_xyz:
            voxel_indices = self.find_voxel_indices(self.samples.sampled_xyz)  # (n, m)
            mask = mask & (voxel_indices != -1)

        bs = int(self.cfg.batch_size / self.samples.sampled_xyz.shape[1])

        for _ in range(self.num_iterations):
            self.model.train()
            with self.timer_training_iteration, torch.enable_grad():
                self.optimizer.zero_grad()
                sdf_pred_all = []
                sdf_prior_all = []
                sdf_grad_all = []
                sdf_prior_grad_all = []
                for i in range(0, num_rays, bs):
                    j = min(i + bs, num_rays)
                    points = self.samples.sampled_xyz[i:j]  # (b, m, 3)
                    voxel_indices_batch = voxel_indices[i:j]
                    out = self.model(points, voxel_indices_batch)
                    sdf_pred = out.pred.squeeze(-1)
                    # Substitute zeros for absent branches so the criterion (which expects tensors) keeps working
                    # in implicit-only / explicit-only modes; for hybrid (the default) both branches are present.
                    sdf_prior = out.prior.squeeze(-1) if out.prior is not None else torch.zeros_like(sdf_pred)
                    sdf_res = out.implicit.squeeze(-1) if out.implicit is not None else torch.zeros_like(sdf_pred)
                    if self.cfg.grad_method == "autodiff":
                        sdf_grad = self.compute_sdf_grad_autodiff(points, sdf_prior + sdf_res)
                        sdf_prior_grad = self.compute_sdf_grad_autodiff(points, sdf_prior)
                    else:
                        sdf_grad, sdf_prior_grad = self.compute_sdf_grad_finite_difference(
                            points=points,
                            offset_points_plus=offset_points_plus[i:j],
                            offset_points_minus=offset_points_minus[i:j],
                            voxel_indices_plus=voxel_indices_plus[i:j],
                            voxel_indices_minus=voxel_indices_minus[i:j],
                        )[:2]

                    sdf_pred_all.append(sdf_pred)
                    sdf_prior_all.append(sdf_prior)  # (b, m)
                    sdf_grad_all.append(sdf_grad)  # (b, m, 3)
                    sdf_prior_grad_all.append(sdf_prior_grad)  # (b, m, 3)

                if len(sdf_pred_all) == 1:
                    sdf_pred_all = sdf_pred_all[0]
                    sdf_prior_all = sdf_prior_all[0]
                    sdf_grad_all = sdf_grad_all[0]
                    sdf_prior_grad_all = sdf_prior_grad_all[0]
                else:
                    sdf_pred_all = torch.cat(sdf_pred_all, dim=0)
                    sdf_prior_all = torch.cat(sdf_prior_all, dim=0)
                    sdf_grad_all = torch.cat(sdf_grad_all, dim=0)
                    sdf_prior_grad_all = torch.cat(sdf_prior_grad_all, dim=0)

                loss, self.loss_dict = self.criterion(
                    pred_sdf=sdf_pred_all,
                    pred_prior=sdf_prior_all,
                    pred_grad=sdf_grad_all,
                    pred_prior_grad=sdf_prior_grad_all,
                    gt_sdf_perturb=self.samples.perturbation_sdf,
                    gt_sdf_stratified=self.samples.stratified_sdf,
                    positive_perturbation_mask=self.samples.positive_perturbation_mask,
                    perturb_sigma_pos=self.cfg.sample_rays.sigma_s_pos,
                    perturb_sigma_neg=self.cfg.sample_rays.sigma_s_neg,
                    n_stratified=self.cfg.sample_rays.n_stratified,
                    n_perturbed=self.cfg.sample_rays.n_perturbed,
                    valid_mask=mask,
                )
                loss.backward()
                self.optimizer.step()
            self.global_step += 1

            self.logger.info(f"step {self.global_step} loss_dict: {self.loss_dict}")
            for k, v in self.loss_dict.items():
                self.logger.tb.add_scalar(f"loss/{k}", v, self.global_step)

            if self.training_iteration_end_callback is not None:
                self.training_iteration_end_callback(self)

    @torch.no_grad()
    def _scatter_samples(self, mask: torch.Tensor) -> None:
        """Non-learning fusion: scatter the GT SDF labels at every sampled point into the SDF field's `values` via
        its `prior_fuser`. No optimizer, no criterion, no gradients.

        Sample layout in `self.samples`:

            [:n_stratified]                              stratified free-space samples, GT in `stratified_sdf`
            [n_stratified : n_stratified+n_perturbed]    perturbed samples, GT in `perturbation_sdf`
            [-1]                                         on-surface sample, GT = 0

        Args:
            mask: `(n_rays, n_strat + n_pert + 1)` bool tensor produced by :meth:`train_with_frame`'s pre-loop bound
                filter -- False entries are dropped before the scatter. Out-of-octree samples are dropped implicitly
                by the fuser (invalid `vertex_indices` are masked inside `Fuser.scatter`).
        """
        with self.timer_training_iteration:
            pts_flat = self.samples.sampled_xyz[mask]  # (M, 3)
            if pts_flat.numel() == 0:
                return

            n_strat = self.cfg.sample_rays.n_stratified
            n_pert = self.cfg.sample_rays.n_perturbed

            # Build the per-sample GT SDF tensor with zeros at the surface slot. Allocate from `sampled_xyz` so dtype
            # / device match without an extra cast.
            gt_sdf = self.samples.sampled_xyz.new_zeros(self.samples.sampled_xyz.shape[:-1])
            if self.samples.stratified_sdf is not None and n_strat > 0:
                gt_sdf[:, :n_strat] = self.samples.stratified_sdf
            if self.samples.perturbation_sdf is not None and n_pert > 0:
                gt_sdf[:, n_strat : n_strat + n_pert] = self.samples.perturbation_sdf
            # Surface slot at [..., -1] stays at 0 from `new_zeros`.

            sdf_flat = gt_sdf[mask]  # (M,)

            self.model.scatter_update(pts_flat, sdf_flat, level=1)

        self.global_step += 1
        if self.training_iteration_end_callback is not None:
            self.training_iteration_end_callback(self)

    @staticmethod
    def compute_sdf_grad_autodiff(points: torch.Tensor, pred_sdf: torch.Tensor):
        """Compute d(pred_sdf)/d(points) via autograd, retaining the graph for backprop.

        Args:
            points: (..., 3) input points with `requires_grad=True`.
            pred_sdf: (...,) predicted SDF values produced from `points`.

        Returns:
            (..., 3) gradient of the SDF with respect to the input points.
        """
        sdf_grad = torch.autograd.grad(
            outputs=pred_sdf,
            inputs=[points],
            grad_outputs=torch.ones_like(pred_sdf),
            create_graph=True,
            # retain_graph=True,
        )[0]
        return sdf_grad

    @torch.no_grad()
    def compute_offset_points_for_finite_diff(self, points: torch.Tensor):
        """
        Compute the offset points for finite difference gradient estimation.
        Args:
            points: (..., 3) points to compute the offset points for

        Returns:
            (..., 3, 3) tensor of points + offset
            (..., 3, 3) tensor of points - offset
        """
        eps = self.cfg.finite_difference_eps
        offset_points_plus = []
        offset_points_minus = []
        for i in range(3):
            points_plus = points.clone()
            points_plus[..., i] += eps  # (..., 3)
            offset_points_plus.append(points_plus)

            points_minus = points.clone()
            points_minus[..., i] -= eps  # (..., 3)
            offset_points_minus.append(points_minus)

        offset_points_plus = torch.stack(offset_points_plus, dim=-2)  # (..., 3, 3)
        offset_points_minus = torch.stack(offset_points_minus, dim=-2)  # (..., 3, 3)

        return offset_points_plus, offset_points_minus

    def compute_sdf_grad_finite_difference(
        self,
        points: torch.Tensor,
        offset_points_plus: torch.Tensor | None = None,
        offset_points_minus: torch.Tensor | None = None,
        voxel_indices_plus: torch.Tensor | None = None,
        voxel_indices_minus: torch.Tensor | None = None,
    ):
        """
        Compute the gradient of the SDF at the given points using finite difference.
        Args:
            points: (..., 3) points to compute the gradient for
            offset_points_plus: (..., 3, 3) tensor of points + offset, if None, will be computed
            offset_points_minus: (..., 3, 3) tensor of points - offset, if None, will be computed
            voxel_indices_plus: (..., 3) voxel indices for offset_points_plus, if None, will be computed
            voxel_indices_minus: (..., 3) voxel indices for offset_points_minus, if None, will be computed

        Returns:
            (..., 3) gradient of the SDF at the given points
            (..., 3) gradient of the SDF prior at the given points
            (..., 3, 3) offset_points_plus
            (..., 3, 3) offset_points_minus
            (..., 3) voxel_indices_plus
            (..., 3) voxel_indices_minus
        """
        eps = self.cfg.finite_difference_eps
        if offset_points_plus is None or offset_points_minus is None:
            offset_points_plus, offset_points_minus = self.compute_offset_points_for_finite_diff(points)
        out_plus = self.model(offset_points_plus, voxel_indices_plus)
        out_minus = self.model(offset_points_minus, voxel_indices_minus)

        if out_plus.prior is not None and out_minus.prior is not None:
            prior_grad = (out_plus.prior.squeeze(-1) - out_minus.prior.squeeze(-1)) / (2 * eps)
        else:
            prior_grad = torch.zeros_like(out_plus.pred.squeeze(-1))

        if out_plus.implicit is not None and out_minus.implicit is not None:
            res_grad = (out_plus.implicit.squeeze(-1) - out_minus.implicit.squeeze(-1)) / (2 * eps)
            grad = prior_grad + res_grad
        else:
            grad = prior_grad

        return (
            grad,
            prior_grad,
            offset_points_plus,
            offset_points_minus,
            voxel_indices_plus,
            voxel_indices_minus,
        )

    def save_mesh(
        self,
        path: str,
        prior: bool = False,
        bound_min: list | None = None,
        bound_max: list | None = None,
        grid_resolution: float | None = None,
        iso_value: float | None = None,
    ) -> None:
        """Extract and write a mesh of the SDF (or SDF prior) field via marching cubes.

        Args:
            path: target file name; treated as absolute if it starts with `/`, otherwise placed under the logger's
                mesh directory.
            prior: if True, extract the `sdf_prior` field instead of the full `sdf`.
            bound_min: optional (3,) lower bound of the extraction grid; defaults to `cfg.bound_min`.
            bound_max: optional (3,) upper bound of the extraction grid; defaults to `cfg.bound_max`.
            grid_resolution: optional grid spacing; defaults to `cfg.mesh_resolution`.
            iso_value: optional iso-value for marching cubes; defaults to `cfg.mesh_iso_value`.
        """

        field = "sdf_prior" if prior else "sdf"
        bound_min = bound_min if bound_min is not None else self.cfg.bound_min
        bound_max = bound_max if bound_max is not None else self.cfg.bound_max
        grid_resolution = grid_resolution if grid_resolution is not None else self.cfg.mesh_resolution
        iso_value = iso_value if iso_value is not None else self.cfg.mesh_iso_value
        self.logger.info(
            f"Extracting mesh ({field}) with bound_min={bound_min}, bound_max={bound_max}, "
            f"grid_resolution={grid_resolution}, iso_value={iso_value}..."
        )

        [mesh] = self.evaluator.extract_mesh(
            bound_min=bound_min,
            bound_max=bound_max,
            grid_resolution=grid_resolution,
            fields=[field],
            iso_value=iso_value,
        )
        self.logger.log_mesh(mesh, path)
        self.logger.info(f"Mesh ({field}) saved to {path}.")

    def query_sdf(self, points: torch.Tensor, return_grad: bool = True, prior_only: bool = False) -> dict:
        """Forward the model on the given points. Returns dict with sdf and (optional) sdf_grad.

        Args:
            points: (..., 3) world-coordinate query points.
            return_grad: if True, compute gradients via autograd.
            prior_only: if True, only evaluate the SDF prior from the octree (skip the implicit head).

        Returns:
            dict with `voxel_indices`, `sdf_prior`, `sdf_residual`, `sdf`, and (when `return_grad`) a nested `grad`
            dict containing `sdf_prior` and `sdf` gradients.
        """
        return self.evaluator.forward_model(
            self.model,
            points.to(self.cfg.device),
            get_grad=return_grad,
            auto_grad=True,
            prior_only=prior_only,
            device=self.cfg.device,
        )

    def evaluate(self, epoch_dir: str | None = None):
        """Extract and persist evaluation artifacts (mesh + axis-aligned slice plots).

        Args:
            epoch_dir: optional sub-directory under the logger's misc/mesh folders to write artifacts into; if None,
                artifacts are written at the top level.
        """
        bound_min = self.cfg.bound_min
        bound_max = self.cfg.bound_max

        if self.cfg.save_mesh:
            mesh_prior, mesh = self.evaluator.extract_mesh(
                bound_min=bound_min,
                bound_max=bound_max,
                grid_resolution=self.cfg.mesh_resolution,
                fields=["sdf_prior", "sdf"],
                iso_value=self.cfg.mesh_iso_value,
            )
            if epoch_dir is not None:
                self.logger.log_mesh(mesh_prior, f"{epoch_dir}/mesh_prior.ply")
                self.logger.log_mesh(mesh, f"{epoch_dir}/mesh.ply")
            else:
                self.logger.log_mesh(mesh_prior, "mesh_prior.ply")
                self.logger.log_mesh(mesh, "mesh.ply")

        if self.cfg.save_slice:
            slice_configs = [
                {
                    "axis_name": "x",
                    "xlabel": "y (m)",
                    "ylabel": "z (m)",
                },
                {
                    "axis_name": "y",
                    "xlabel": "x (m)",
                    "ylabel": "z (m)",
                },
                {
                    "axis_name": "z",
                    "xlabel": "x (m)",
                    "ylabel": "y (m)",
                },
            ]
            fontsize = 12
            for axis in range(3):
                if self.cfg.slice_center is None:
                    pos = 0.5 * (bound_min[axis] + bound_max[axis])
                else:
                    pos = self.cfg.slice_center[axis]
                slice_result = self.evaluator.extract_slice(
                    axis=axis,
                    pos=pos,
                    resolution=self.cfg.mesh_resolution,
                    bound_min=bound_min,
                    bound_max=bound_max,
                )

                slice_config = slice_configs[axis]
                axis_name = slice_config["axis_name"]
                slice_bound = slice_result["slice_bound"].tolist()  # (bound_min, bound_max) for the two axes

                for slice_name in ["sdf_prior", "sdf_residual", "sdf"]:
                    slice_values = slice_result[slice_name].cpu().numpy()
                    plt.figure()
                    im = plt.imshow(
                        slice_values,
                        extent=(
                            slice_bound[0][0],
                            slice_bound[1][0],
                            slice_bound[0][1],
                            slice_bound[1][1],
                        ),
                        origin="lower",
                        cmap="jet",
                    )
                    plt.colorbar(im, shrink=0.8)
                    plt.xlabel(slice_config["xlabel"], fontsize=fontsize)
                    plt.ylabel(slice_config["ylabel"], fontsize=fontsize)
                    plt.title(f"At {axis_name} = {pos:.2f} m", fontsize=fontsize)
                    plt.tight_layout()
                    img_path = f"slice_{axis_name}_{slice_name}.png"
                    if epoch_dir is not None:
                        img_path = os.path.join(self.logger.misc_dir, epoch_dir, img_path)
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    else:
                        img_path = os.path.join(self.logger.misc_dir, img_path)
                    plt.savefig(img_path, dpi=300)
                    plt.close()

        self.logger.info("Evaluation completed.")


def main():
    parser = TrainerConfig.get_argparser()
    cfg: TrainerConfig = parser.parse_args()
    trainer = SdfTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
