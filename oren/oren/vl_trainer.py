"""VlTrainer: train (or just fuse) a vision-language FieldStorage over a VL feature bundle.

A thin :class:`TrainerBase` subclass that mirrors :class:`SdfTrainer.train_with_frame` structurally: select key
frames via `key_frame_set.select_key_frames()`, sample `(points, features)` via
`key_frame_set.sample_points_and_features(...)`, then run the inner optimisation loop. Scatter mode short-circuits
the loop -- it just folds the current frame's `(points, features)` into the field's `prior_fuser` via
`model.update(...)`.

Frame storage, key-frame management, optimizer wiring, octree insertion, and the streaming / online / offline
outer loops all live in `TrainerBase` -- this trainer adds only the per-frame work that's VL-specific.
"""

import torch

from oren.frame import Frame, VlFrame
from oren.trainer_base import TrainerBase
from oren.trainer_config import TrainerConfig
from oren.utils.registry import get_identifier, register_trainer
from oren.vl_criterion import VlCriterion
from oren.vl_network import VlNetwork


@register_trainer
class VlTrainer(TrainerBase):
    """VL trainer built on :class:`TrainerBase`.

    Construction default-swaps the TrainerConfig SDF defaults to their VL equivalents so `python -m oren.vl_trainer`
    without a YAML still produces a sensible VL run.
    """

    model: VlNetwork
    criterion: VlCriterion

    def __init__(self, cfg: TrainerConfig, data_stream=None):
        assert cfg.model_identifier == get_identifier(
            VlNetwork
        ), f"VlTrainer requires VlNetwork, got {cfg.model_identifier}"
        assert cfg.criterion_identifier == get_identifier(
            VlCriterion
        ), f"VlTrainer requires VlCriterion, got {cfg.criterion_identifier}"
        super().__init__(cfg, data_stream=data_stream)

    # ------------------------------------------------------------------
    # TrainerBase abstract hooks
    # ------------------------------------------------------------------

    def _create_evaluator(self):
        """VL has no mesh / slice extraction; the evaluator slot is unused."""
        return None

    def evaluate(self, epoch_dir: str | None = None) -> None:
        """No-op. VL features don't have a mesh or slice to extract."""
        return None

    def save_mesh(self, path: str, prior: bool = False, **kwargs) -> None:
        """No-op. VL features don't have a mesh."""
        return None

    # ------------------------------------------------------------------
    # Per-frame work
    # ------------------------------------------------------------------

    def train_with_frame(self, frame: Frame | None) -> bool:
        """Mirror :meth:`SdfTrainer.train_with_frame` for VL.

        Scatter mode: fold the current frame's `(points, features)` into the field via `model.update`. No optimizer,
        no criterion, no inner loop.

        Optimize mode: pick key frames, sample paired (points, features) with the current frame, then run
        `num_iterations_per_frame` (or `init_frame_iterations` for the warmup epochs) Adam + VlCriterion steps.
        """
        self.num_iterations = self.cfg.num_iterations_per_frame
        if self.epoch < self.cfg.num_init_frames:
            self.num_iterations = self.cfg.init_frame_iterations

        if self.training_frame_start_callback is not None:
            if not self.training_frame_start_callback(self, frame):
                self.logger.info("Training interrupted by callback, exiting.")
                return False

        if self.cfg.mode == "scatter":
            if frame is not None:
                self._scatter_frame(frame)
            return True

        with self.timer_select_key_frames:
            self.selected_key_frame_indices = self.key_frame_set.select_key_frames()

        for _ in range(self.num_iterations):
            self._train_one_iteration(current_frame=frame)
        return True

    def train_with_frames(self, frames: list[Frame]) -> bool:
        """Offline-epoch step. Mirrors the SDF offline path's "all batch frames count as key frames for this step"
        semantic: synthesise a temporary `key_frame_indices` from the batch directly and reuse
        `key_frame_set.sample_points_and_features` over the live `key_frame_set.frames` list.

        Args:
            frames: a batch of frames produced by :class:`TrainerBase._train_bounded_offline`.

        Returns:
            True to keep training; False if a callback short-circuits (not currently invoked here).
        """
        if not frames:
            return True

        if self.cfg.mode == "scatter":
            for f in frames:
                self._scatter_frame(f)
            return True

        self.num_iterations = self.cfg.num_iterations_per_frame
        if self.epoch < self.cfg.num_init_frames:
            self.num_iterations = self.cfg.init_frame_iterations

        # Snapshot the key-frame list, splice the offline batch in, run iterations, then restore. Lets us reuse the
        # same `sample_points_and_features` call site without growing the persistent key-frame set with non-key frames.
        saved_frames = self.key_frame_set.frames
        self.key_frame_set.frames = list(frames)
        batch_indices = list(range(len(frames)))
        try:
            for _ in range(self.num_iterations):
                self._train_one_iteration(current_frame=None, key_frame_indices=batch_indices)
        finally:
            self.key_frame_set.frames = saved_frames
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _scatter_frame(self, frame: Frame) -> None:
        """Push one frame's `(world_points, features)` through the field's `prior_fuser.scatter`."""
        device = self.cfg.device
        pts = frame.get_points(to_world_frame=True, device=device)
        if pts.numel() == 0:
            return
        feats = frame.get_vl_features(valid_only=True, device=device)
        self.model.scatter_update(pts, feats, level=1)

    def _train_one_iteration(self, current_frame: Frame | None, key_frame_indices: list[int] | None = None) -> None:
        """One optimization step. Sampling is delegated to `key_frame_set.sample_points_and_features`.
        Unlike :class:`SdfTrainer`, each iteration runs over a fresh set of points and features sampled
        from the selected frames.
        """
        if key_frame_indices is None:
            key_frame_indices = self.selected_key_frame_indices

        with self.timer_sample_rays:
            result = self.key_frame_set.sample_by_num(
                total_num_samples=self.cfg.num_rays_total,
                sample_frame_fn=VlFrame.sample_points_and_features,
                key_frame_indices=key_frame_indices,
                current_frame=current_frame,
                device=self.cfg.device,
            )
            if result is None:
                return
            pts, gt = result
        if pts.numel() == 0:
            return

        bs = self.cfg.batch_size
        n = pts.shape[0]

        self.model.train()
        with self.timer_training_iteration, torch.enable_grad():
            self.optimizer.zero_grad()
            pred_all = []
            prior_all = []
            for i in range(0, n, bs):
                out = self.model(pts[i : i + bs])
                pred_all.append(out.pred)
                if out.prior is not None:
                    prior_all.append(out.prior)
            if len(pred_all) == 1:
                pred_all = pred_all[0]
                prior_all = prior_all[0] if prior_all else None
            else:
                pred_all = torch.cat(pred_all, dim=0)
                prior_all = torch.cat(prior_all, dim=0) if prior_all else None

            loss, self.loss_dict = self.criterion(pred_vl=pred_all, pred_prior=prior_all, gt_vl=gt)
            loss.backward()
            self.optimizer.step()
        self.global_step += 1

        self.logger.info(f"step {self.global_step} loss_dict: {self.loss_dict}")
        for k, v in self.loss_dict.items():
            self.logger.tb.add_scalar(f"loss/{k}", v, self.global_step)

        if self.training_iteration_end_callback is not None:
            self.training_iteration_end_callback(self)


def main() -> None:
    """CLI entry point. Uses :class:`TrainerConfig`'s argparser; the trainer's `__init__` swaps SDF defaults to VL
    when the user hasn't overridden `model_identifier` / `criterion_identifier` / `data.dataset_name`.
    """
    parser = TrainerConfig.get_argparser()
    cfg, _ = parser.parse_known_args()
    if cfg.trainer_identifier == "oren.sdf_trainer.SdfTrainer":
        cfg.trainer_identifier = get_identifier(VlTrainer)
    trainer = VlTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
