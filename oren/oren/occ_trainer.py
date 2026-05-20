import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oren import torch
from oren.evaluator_occ import OccEvaluator
from oren.frame import Frame
from oren.occ_criterion import OccCriterion
from oren.occ_network import OccNetwork
from oren.trainer_base import TrainerBase
from oren.trainer_config import TrainerConfig
from oren.utils.registry import get_identifier, register_trainer
from oren.utils.sampling import generate_sdf_samples


@register_trainer
class OccTrainer(TrainerBase):
    """Trainer for OccNetwork. Binary-occupancy loss driven by depth ray-casting; no input gradients are needed
    (criterion.needs_grad == False). Mesh/slice extraction goes through OccEvaluator, which negates logits before
    marching cubes so the SDF-style sign convention (positive = outside) drives the iso-surface."""

    model: OccNetwork
    criterion: OccCriterion
    evaluator: OccEvaluator

    def __init__(self, cfg: TrainerConfig, data_stream=None):
        assert cfg.model_identifier == get_identifier(
            OccNetwork
        ), f"OccTrainer requires OccNetwork, got {cfg.model_identifier}"
        assert cfg.criterion_identifier == get_identifier(
            OccCriterion
        ), f"OccTrainer requires OccCriterion, got {cfg.criterion_identifier}"
        assert cfg.mode == "optimize", f"OccTrainer only supports optimize mode, got {cfg.mode}"

        super().__init__(cfg, data_stream=data_stream)

    def _create_evaluator(self):
        return OccEvaluator(
            batch_size=self.cfg.batch_size,
            clean_mesh=self.cfg.clean_mesh,
            model_cfg=self.cfg.model,
            model=self.model,
            device=self.cfg.device,
        )

    def train_with_frame(self, frame: Frame | None):
        """Run `num_iterations` occupancy training iterations against the selected key frames and `frame`.

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
                return False

        with self.timer_select_key_frames:
            self.selected_key_frame_indices = self.key_frame_set.select_key_frames()
        with self.timer_sample_rays:
            rays_o_all, rays_d_all, depth_samples_all = self.key_frame_set.sample_by_num(
                total_num_samples=self.cfg.num_rays_total,
                sample_frame_fn=Frame.sample_rays,
                key_frame_indices=self.selected_key_frame_indices,
                current_frame=frame,
                device=self.cfg.device,
            )

            if self.cfg.extra_surface_sample:
                self.extra_surface_pcd = self.key_frame_set.sample_by_ratio(
                    ratio=1.0 / self.cfg.frame_downsample,
                    sample_frame_fn=Frame.sample_points,
                    key_frame_indices=self.selected_key_frame_indices,
                    current_frame=frame,
                )

        with self.timer_generate_sdf_samples:
            self.samples = generate_sdf_samples(
                rays_d_all=rays_d_all,
                rays_o_all=rays_o_all,
                depth_samples_all=depth_samples_all,
                cfg=self.cfg.sample_rays,
                extra_surface_pcd=self.extra_surface_pcd,
                device=self.cfg.device,
                compute_sdf_labels=False,  # OccCriterion ignores SDF labels; skip the NN call
            )

            mask = torch.ones_like(self.samples.sampled_xyz[..., 0], dtype=torch.bool)
            if not self.cfg.data.apply_bound:
                mask = mask & (self.samples.sampled_xyz >= self.bound_min).all(dim=-1)
                mask = mask & (self.samples.sampled_xyz <= self.bound_max).all(dim=-1)

        # self.samples.sampled_xyz: (n, m, 3)
        num_rays = self.samples.sampled_xyz.shape[0]
        with self.timer_find_voxel_indices_sampled_xyz:
            voxel_indices = self.find_voxel_indices(self.samples.sampled_xyz)  # (n, m)
            mask = mask & (voxel_indices != -1)

        bs = int(self.cfg.batch_size / self.samples.sampled_xyz.shape[1])

        smoothness_weight = self.cfg.criterion.smoothness_weight
        smoothness_eps = self.cfg.criterion.smoothness_eps

        for _ in range(self.num_iterations):
            self.model.train()
            with self.timer_training_iteration, torch.enable_grad():
                self.optimizer.zero_grad()
                occ_pred_all = []
                occ_prior_all = []
                occ_pred_perturb_all = [] if smoothness_weight > 0 else None
                for i in range(0, num_rays, bs):
                    j = min(i + bs, num_rays)
                    points = self.samples.sampled_xyz[i:j]  # (b, m, 3)
                    voxel_indices_batch = voxel_indices[i:j]
                    out = self.model(points, voxel_indices_batch)
                    occ_pred = out.pred.squeeze(-1)
                    # Hybrid mode is the default; substitute zeros for prior when running in implicit-only mode so
                    # the criterion's prior-based regularizers degrade to no-ops instead of crashing.
                    occ_prior = out.prior.squeeze(-1) if out.prior is not None else torch.zeros_like(occ_pred)
                    occ_pred_all.append(occ_pred)
                    occ_prior_all.append(occ_prior)

                    if smoothness_weight > 0:
                        # Fresh ε per iteration, in metric coords; voxel indices are recomputed because the
                        # perturbation may cross a leaf boundary.
                        eps = (torch.rand_like(points) * 2 - 1) * smoothness_eps
                        points_perturb = points + eps
                        voxel_indices_perturb = self.find_voxel_indices(points_perturb)
                        out_perturb = self.model(points_perturb, voxel_indices_perturb)
                        occ_pred_perturb = out_perturb.pred.squeeze(-1)
                        occ_pred_perturb_all.append(occ_pred_perturb)

                if len(occ_pred_all) == 1:
                    occ_pred_all = occ_pred_all[0]
                    occ_prior_all = occ_prior_all[0]
                    if smoothness_weight > 0:
                        occ_pred_perturb_all = occ_pred_perturb_all[0]
                else:
                    occ_pred_all = torch.cat(occ_pred_all, dim=0)
                    occ_prior_all = torch.cat(occ_prior_all, dim=0)
                    if smoothness_weight > 0:
                        occ_pred_perturb_all = torch.cat(occ_pred_perturb_all, dim=0)

                loss, self.loss_dict = self.criterion(
                    pred_occ=occ_pred_all,
                    pred_prior=occ_prior_all,
                    positive_perturbation_mask=self.samples.positive_perturbation_mask,
                    n_stratified=self.cfg.sample_rays.n_stratified,
                    n_perturbed=self.cfg.sample_rays.n_perturbed,
                    valid_mask=mask,
                    pred_occ_perturb=occ_pred_perturb_all,
                )
                loss.backward()
                self.optimizer.step()
            self.global_step += 1

            self.logger.info(f"loss_dict: {self.loss_dict}")
            for k, v in self.loss_dict.items():
                self.logger.tb.add_scalar(f"loss/{k}", v, self.global_step)

            if self.training_iteration_end_callback is not None:
                self.training_iteration_end_callback(self)
        return True

    def save_mesh(
        self,
        path: str,
        prior: bool = False,
        bound_min: list | None = None,
        bound_max: list | None = None,
        grid_resolution: float | None = None,
        iso_value: float | None = None,
    ) -> None:
        """Extract and write a mesh of the occupancy (or occupancy prior) iso-surface.

        Args:
            path: target file name; treated as absolute if it starts with `/`, otherwise placed under the logger's
                mesh directory.
            prior: if True, extract the `occ_prior` field instead of the full `occ`.
            bound_min: optional (3,) lower bound of the extraction grid; defaults to `cfg.bound_min`.
            bound_max: optional (3,) upper bound of the extraction grid; defaults to `cfg.bound_max`.
            grid_resolution: optional grid spacing; defaults to `cfg.mesh_resolution`.
            iso_value: optional iso-value (in occupancy-logit sign) for marching cubes; defaults to
                `cfg.mesh_iso_value`.
        """
        field = "occ_prior" if prior else "occ"
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

    def query_occ(self, points: torch.Tensor, return_grad: bool = False, prior_only: bool = False) -> dict:
        """Forward the model on the given points. Returns dict with occupancy logits and (optional) gradients. Logits:
        positive=occupied, negative=free, zero=surface.

        Args:
            points: (..., 3) world-coordinate query points.
            return_grad: if True, compute gradients via autograd.
            prior_only: if True, only evaluate the occupancy prior from the octree (skip the implicit head).

        Returns:
            dict with `voxel_indices`, `occ_prior`, `occ_residual`, `occ`, and (when `return_grad`) a nested `grad`
            dict containing `occ_prior` and `occ` gradients.
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
                fields=["occ_prior", "occ"],
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
                {"axis_name": "x", "xlabel": "y (m)", "ylabel": "z (m)"},
                {"axis_name": "y", "xlabel": "x (m)", "ylabel": "z (m)"},
                {"axis_name": "z", "xlabel": "x (m)", "ylabel": "y (m)"},
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
                slice_bound = slice_result["slice_bound"].tolist()

                # Diverging colormap centered at 0 highlights the surface (logit=0 boundary).
                for slice_name in ["occ_prior", "occ_residual", "occ"]:
                    slice_values = slice_result[slice_name]
                    if slice_values is None:
                        self.logger.warn(f"Slice {slice_name} is None, skipping visualization.")
                        continue
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
                        cmap="RdBu_r",
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
    trainer = OccTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
