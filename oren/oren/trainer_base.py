import random
from typing import Callable, Optional

import numpy as np
from tqdm import tqdm

from oren import torch
from oren.frame import Frame
from oren.key_frame_set import KeyFrameSet
from oren.loggers import BasicLogger
from oren.trainer_config import TrainerConfig
from oren.utils.import_util import get_dataset
from oren.utils.profiling import GpuTimer
from oren.utils.registry import get_criterion, get_model
from oren.utils.sampling import SampleResults


class TrainerBase:
    """Shared scaffolding for field-specific trainers (SdfTrainer, OccTrainer, ...).

    Provides: data-stream wiring, key-frame management, octree insertion,
    streaming/bounded outer loops, optimizer + criterion construction, save_model,
    timers, and callback hooks. Subclasses implement train_with_frame, evaluate,
    save_mesh, and any field-specific query helpers, plus _create_evaluator.
    """

    def __init__(self, cfg: TrainerConfig, data_stream=None):
        self.cfg = cfg

        self.setup_seed(self.cfg.seed)

        if data_stream is None:
            self.data_stream = get_dataset(cfg.data.dataset_name, cfg.data.dataset_args)
        else:
            self.data_stream = data_stream

        # Streaming sources advertise themselves with a class attribute.
        # Python's built-in len() rejects negative returns from __len__, so a length-based sentinel can't be used.
        self.streaming = getattr(self.data_stream, "streaming", False)

        # set the bound automatically from the dataset if available.
        # the bound is used for evaluation and mesh extraction.
        # the training does not rely on the bound.
        if self.data_stream.bound_min is not None and self.data_stream.bound_max is not None:
            self.cfg.bound_min = (self.data_stream.bound_min - 0.1).cpu().tolist()
            self.cfg.bound_max = (self.data_stream.bound_max + 0.1).cpu().tolist()

        self.bound_min = torch.tensor(self.cfg.bound_min, dtype=torch.float32, device=self.cfg.device)
        self.bound_max = torch.tensor(self.cfg.bound_max, dtype=torch.float32, device=self.cfg.device)

        if not self.streaming:
            if self.cfg.data.end_frame < 0:
                self.cfg.data.end_frame = len(self.data_stream)
            self.cfg.data.start_frame = min(self.cfg.data.start_frame, len(self.data_stream) - 1)
            self.cfg.data.end_frame = min(self.cfg.data.end_frame, len(self.data_stream))
        self.current_frame_idx = self.cfg.data.start_frame

        self.key_frame_set = KeyFrameSet(
            cfg=self.cfg.key_frame_set,
            max_num_voxels=self.cfg.model.octree_cfg.init_voxel_num,
            device=self.cfg.device,
        )
        self.model = get_model(self.cfg.model_identifier)(self.cfg.model)
        self.model.to(self.cfg.device)

        self.logger = BasicLogger(cfg.log_dir, cfg.exp_name, cfg.as_dict())

        self.epoch = 0
        self.global_step = 0
        self.num_iterations = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.criterion = get_criterion(self.cfg.criterion_identifier)(
            cfg=self.cfg.criterion,
            n_stratified=self.cfg.sample_rays.n_stratified,
            n_perturbed=self.cfg.sample_rays.n_perturbed,
        )

        self.selected_key_frame_indices = []
        self.samples: Optional[SampleResults] = None
        self.extra_surface_pcd: Optional[torch.Tensor] = None
        self.loss_dict = dict()

        timer_on = self.cfg.profiling
        verbose = self.cfg.profiling_verbose
        self.timer_octree_insert = GpuTimer("octree insert", enable=timer_on, verbose=verbose)
        self.timer_key_frame_set_update = GpuTimer("key frame set update", enable=timer_on, verbose=verbose)
        self.timer_train_frame = GpuTimer("train with frame", enable=timer_on, verbose=verbose)
        self.timer_select_key_frames = GpuTimer("select key frames", enable=timer_on, verbose=verbose)
        self.timer_sample_rays = GpuTimer("sample rays", enable=timer_on, verbose=verbose)
        self.timer_generate_sdf_samples = GpuTimer("generate sdf samples", enable=timer_on, verbose=verbose)
        self.timer_compute_offset_points = GpuTimer("compute offset points", enable=timer_on, verbose=verbose)
        self.timer_find_voxel_indices_offset_points = GpuTimer(
            "find voxel indices for offset points", enable=timer_on, verbose=verbose
        )
        self.timer_find_voxel_indices_sampled_xyz = GpuTimer(
            "find voxel indices for sampled_xyz", enable=timer_on, verbose=verbose
        )
        self.timer_training_iteration = GpuTimer("training iteration", enable=timer_on, verbose=verbose)

        self.training_iteration_end_callback: Callable[["TrainerBase"], None] = None  # type: ignore
        self.training_frame_start_callback: Callable[["TrainerBase", Frame], bool] = None  # type: ignore
        self.training_end_callback: Callable[["TrainerBase"], None] = None  # type: ignore

        self.evaluator = self._create_evaluator()

    def _create_evaluator(self):
        raise NotImplementedError("Subclasses must implement _create_evaluator()")

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self):
        try:
            if self.streaming:
                self._train_streaming()
            else:
                self._train_bounded()
        finally:
            for _ in range(self.cfg.final_iterations):
                self.train_with_frame(None)

            self.logger.info("Training completed.")
            if self.training_end_callback is not None:
                self.training_end_callback(self)

            if self.cfg.final_evaluate:
                self.evaluate()
            if self.cfg.final_save_model:
                self.save_model("final.pth")

    def _train_streaming(self) -> None:
        pbar = tqdm(desc="Mapping (streaming)", ncols=120, leave=False)
        try:
            # init frame_id for streaming source, which is only used for logging and checkpoint naming.
            # self.current_frame_idx is the fetching counter for streaming source, which is increased
            # whenever we fetch a frame (even if it's None or has bad pose).
            frame_id = self.current_frame_idx
            while True:
                frame = self.fetch_one_frame()
                if frame is None:
                    # `None` from a streaming source means either shutdown or transient idle;
                    # the loader exposes which via `is_shutdown` (default True for sources without it,
                    # so non-ROS streaming sources keep non-streaming behavior).
                    if getattr(self.data_stream, "is_shutdown", True):
                        self.logger.info("No more frames (data stream closed), finish mapping.")
                        return
                    # Transient idle: keep optimizing on existing keyframes.
                    if not self.train_with_frame(None):
                        return
                    continue
                if not self._step_one_frame(frame, frame_id):
                    return
                frame_id += 1  # increase frame_id when frame is not None and processed successfully
                pbar.update(1)
        finally:
            pbar.close()

    def _train_bounded(self) -> None:
        for frame_id in tqdm(
            range(self.cfg.data.start_frame, self.cfg.data.end_frame),
            desc="Mapping",
            ncols=120,
            leave=False,
        ):
            frame = self.fetch_one_frame()
            if frame is None:
                self.logger.info("No more valid frames, finish mapping.")
                return
            if not self._step_one_frame(frame, frame_id):
                return

    def _step_one_frame(self, frame: Frame, frame_id: int) -> bool:
        """Run insertion + key-frame update + training for one frame. Returns False if interrupted by callback."""
        points = frame.get_points(to_world_frame=True, device=self.cfg.device)

        with self.timer_octree_insert:
            _, seen_voxels = self.insert_points_to_octree(points)

        with self.timer_key_frame_set_update:
            is_key_frame = self.update_key_frame_set(frame, seen_voxels)

        if is_key_frame:
            self.logger.info(f"Frame {frame_id} is selected as a key frame.")

        with self.timer_train_frame:
            if not self.train_with_frame(frame=frame):
                return False
        self.epoch += 1

        if self.cfg.ckpt_interval > 0 and self.epoch % self.cfg.ckpt_interval == 0:
            self.save_model(f"epoch_{self.epoch:04d}.pth")
        return True

    def fetch_one_frame(self) -> Optional[Frame]:
        frame = None
        if self.streaming:
            # Streaming source: index value is unused; loader blocks until a frame
            # is ready and returns None on shutdown. Skip frames with bad poses.
            while True:
                frame = self.data_stream[self.current_frame_idx]
                self.current_frame_idx += 1  # fetching counter for streaming source
                if frame is None:
                    return None
                if torch.all(frame.get_ref_pose().isfinite()):
                    return frame
        else:
            while self.current_frame_idx < self.cfg.data.end_frame:
                frame = self.data_stream[self.current_frame_idx]
                self.current_frame_idx += 1
                if not torch.all(frame.get_ref_pose().isfinite()):  # bad pose
                    continue
                break
            return frame

    @torch.no_grad()
    def insert_points_to_octree(self, points: torch.Tensor):
        voxels, seen_voxels = self.model.octree.insert_points(points)
        return voxels, seen_voxels

    @torch.no_grad()
    def find_voxel_indices(self, points: torch.Tensor):
        """
        Find the voxel indices for the given points.
        Args:
            points: (..., 3) points to find the voxel indices for

        Returns:
            (..., ) voxel indices for the given points, -1 if not exists
        """
        shape = points.shape
        voxel_indices = self.model.octree.find_voxel_indices(points.view(-1, 3), False)
        voxel_indices = voxel_indices.view(shape[:-1])
        return voxel_indices

    def update_key_frame_set(self, frame: Frame, seen_voxels: torch.Tensor) -> bool:
        return self.key_frame_set.add_key_frame(frame, seen_voxels)

    def select_key_frames(self) -> list[int]:
        return self.key_frame_set.select_key_frames()

    def train_with_frame(self, frame: Optional[Frame]) -> bool:
        raise NotImplementedError("Subclasses must implement train_with_frame()")

    @torch.no_grad()
    def save_model(self, path: str):
        self.logger.log_ckpt(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}.")

    def save_mesh(self, path: str, prior: bool = False, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement save_mesh()")

    def get_time_stats(self) -> dict:
        time_stats = {
            "train_frame": self.timer_train_frame.average_t,
            "octree_insert": self.timer_octree_insert.average_t,
            "key_frame_set_update": self.timer_key_frame_set_update.average_t,
            "select_key_frames": self.timer_select_key_frames.average_t,
            "sample_rays": self.timer_sample_rays.average_t,
            "generate_sdf_samples": self.timer_generate_sdf_samples.average_t,
            "compute_offset_points": self.timer_compute_offset_points.average_t,
            "find_voxel_indices_offset_points": self.timer_find_voxel_indices_offset_points.average_t,
            "find_voxel_indices_sampled_xyz": self.timer_find_voxel_indices_sampled_xyz.average_t,
            "training_iteration": self.timer_training_iteration.average_t,
        }
        return time_stats

    def evaluate(self, epoch_dir: Optional[str] = None) -> None:
        raise NotImplementedError("Subclasses must implement evaluate()")
