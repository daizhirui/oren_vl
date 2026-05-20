from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from oren.frame import Frame
from oren.utils.config_abc import ConfigABC
from oren.utils.frame_util import multiple_max_set_coverage, sample_from_frames


@dataclass
class KeyFrameSetConfig(ConfigABC):
    insert_method: str = "insert_method"  # naive | intersection
    insert_interval: int = 50  # number of frames between key frames
    insert_ratio: float = 0.85
    frame_selection: str = "multiple_max_set_coverage"  # multiple_max_set_coverage | random
    selection_window_size: int = 8
    frame_weight: str = "uniform"


class KeyFrameSet:
    def __init__(self, cfg: KeyFrameSetConfig, max_num_voxels: int, device: str):
        """Initialize an empty key-frame set.

        Args:
            cfg: key-frame-set configuration controlling insertion / selection strategy.
            max_num_voxels: octree voxel-buffer capacity, used to allocate the `kf_unoptimized_voxels` mask for
                `multiple_max_set_coverage` selection.
            device: torch device on which auxiliary tensors (voxel masks) live.
        """
        self.cfg = cfg
        self.max_num_voxels = max_num_voxels
        self.device = device

        self.frames: list[Frame] = []
        self.valid_indices: dict[int, torch.Tensor] = {}
        self.sample_counts: dict[int, int] = {}

        self.kf_indices: set[int] = set()
        self.kf_seen_voxel_indices: list[torch.Tensor] = []
        self.kf_seen_voxel_num: list[int] = []
        self.kf_unoptimized_voxels: Optional[torch.Tensor] = None
        self.kf_all_voxels: Optional[torch.Tensor] = None

    def add_key_frame(self, frame: Frame, seen_voxel_indices: torch.Tensor):
        """
        Adds a key frame to the set.
        Args:
            frame: RGBDFrame to be added.
            seen_voxel_indices: indices of voxels seen by the frame.
        Returns:
            bool: True if the frame is added as a key frame, False otherwise.
        """
        if self.is_key_frame(frame, seen_voxel_indices):
            self.add_frame(frame, seen_voxel_indices)
            return True
        return False

    def is_key_frame(self, frame: Frame, seen_voxel_indices: torch.Tensor):
        """
        Decide whether to add the frame as a key frame.
        If self.frames is empty, return True.
        If self.cfg.insert_method is "naive", return True if the frame index is greater than the last key frame index
        by self.cfg.insert_interval.
        If self.cfg.insert_method is "intersection", compute the IoU of the voxels seen by the frame and the last key
        frame. Return True if IoU < self.cfg.insert_ratio.

        Args:
            frame: Frame to be added.
            seen_voxel_indices: indices of voxels seen by the frame.

        Returns:
            True if the frame should be added as a key frame, False otherwise.
        """
        if len(self.frames) == 0:
            return True

        if self.cfg.insert_method == "naive":
            if frame.get_frame_index() - self.frames[-1].get_frame_index() >= self.cfg.insert_interval:
                return True
            return False

        voxels_unique, counts = torch.unique(
            torch.cat([self.kf_seen_voxel_indices[-1], seen_voxel_indices], dim=0),
            return_counts=True,
            sorted=False,
            dim=0,
        )
        n_intersection = torch.sum(counts > 1).item()
        n_union = voxels_unique.shape[0]
        iou = n_intersection / n_union
        if iou < self.cfg.insert_ratio:
            return True
        return False

    def add_frame(self, frame: Frame, seen_voxel_indices: torch.Tensor):
        """
        Add a frame to the set.
        1. Append the frame to self.frames.
        2. Append the indices of voxels seen by the frame to self.kf_seen_voxel_indices.
        3. Append the number of voxels seen by the frame to self.kf_seen_voxel_num.
        4. Append the valid indices of the frame to self.valid_indices.
        5. Initialize the sample count of the frame.
        6. Update self.kf_unoptimized_voxels if using "multiple_max_set_coverage" selection.

        Args:
            frame: Frame to be added.
            seen_voxel_indices: indices of voxels seen by the frame.

        Returns:

        """
        self.frames.append(frame)
        fid = frame.get_frame_index()
        self.kf_indices.add(fid)
        self.kf_seen_voxel_indices.append(seen_voxel_indices)
        self.kf_seen_voxel_num.append(seen_voxel_indices.shape[0])

        if fid not in self.valid_indices:
            self.valid_indices[fid] = torch.nonzero(frame.get_valid_mask().view(-1))
        if fid not in self.sample_counts:
            self.sample_counts[fid] = sum(self.sample_counts.values()) // (len(self.sample_counts) + 2)

        if self.cfg.frame_selection == "multiple_max_set_coverage" and self.kf_unoptimized_voxels is not None:
            self.kf_unoptimized_voxels.index_fill_(0, seen_voxel_indices.long().view(-1).to(self.device), True)

    def select_key_frames(self) -> list[int]:
        """
        Pick self.cfg.selection_window_size key frames from self.frames.
        The selection strategy is set by self.cfg.frame_selection.
        If the number of frames is less than or equal to selection_window_size, we return all frames.

        Returns:
            list of indices of selected key frames.
        """
        if len(self.frames) <= self.cfg.selection_window_size:
            return list(range(len(self.frames)))

        if self.cfg.frame_selection == "random":
            selected_frame_indices = torch.randperm(len(self.frames))[: self.cfg.selection_window_size].tolist()
            return selected_frame_indices

        if self.cfg.frame_selection == "multiple_max_set_coverage":
            selected_frame_indices, self.kf_unoptimized_voxels, self.kf_all_voxels = multiple_max_set_coverage(
                self.kf_seen_voxel_num,
                self.kf_seen_voxel_indices,
                self.kf_unoptimized_voxels,
                self.kf_all_voxels,
                self.cfg.selection_window_size,
                num_voxels=self.max_num_voxels,
                device=self.device,
            )
            return selected_frame_indices

        raise ValueError(f"Unknown frame selection method: {self.cfg.frame_selection}")

    def allocate_num_samples_to_frames(self, total_num_samples: int, frames: list[Frame]) -> list[int]:
        """Distribute `total_num_samples` across `frames` according to `self.cfg.frame_weight`.

        With `frame_weight == "uniform"` the budget is split evenly and any remainder is given out
        one-per-frame from the front. Otherwise each frame is weighted *inversely* by its accumulated
        per-frame sample count in `self.sample_counts` (frames sampled less in the past receive more
        this round); frames missing from the dict fall back to a smoothed default. The returned list
        always sums to `total_num_samples`. Assumes `frames` is non-empty.

        Args:
            total_num_samples: total number of samples to allocate.
            frames: frames the samples will be drawn from; allocation order matches input order.

        Returns:
            `samples_per_frame`: list of int with length `len(frames)`, summing to `total_num_samples`.
        """

        n_frames = len(frames)
        if n_frames == 1:
            fid = frames[0].get_frame_index()
            self.sample_counts[fid] = self.sample_counts.get(fid, 0) + total_num_samples
            return [total_num_samples]

        if self.cfg.frame_weight == "uniform":
            samples_per_frame = [total_num_samples // n_frames] * n_frames
            for i in range(total_num_samples % n_frames):
                samples_per_frame[i] += 1
            for frame, n in zip(frames, samples_per_frame):
                fid = frame.get_frame_index()
                self.sample_counts[fid] = self.sample_counts.get(fid, 0) + n
            return samples_per_frame

        # To achieve balanced sampling across frames, we allocate more samples to frames with lower sample counts so
        # far. However, if we allocate strictly inversely proportional to the sample counts, frames with zero samples
        # so far will get too many samples. To avoid this, we use a smoothed version of the sample counts where we
        # add a default count to each frame's sample count.
        default_count = sum(self.sample_counts.values()) // (len(self.sample_counts) + 2)
        sample_counts = [self.sample_counts.get(frame.get_frame_index(), default_count) for frame in frames]
        total_count = sum(sample_counts)
        if total_count == 0:
            samples_per_frame = [total_num_samples // n_frames] * n_frames
            for i in range(total_num_samples % n_frames):
                samples_per_frame[i] += 1
            return samples_per_frame

        m = total_count * (n_frames - 1)
        samples_per_frame = [max(1, int(total_num_samples * (total_count - count) / m)) for count in sample_counts]
        # adjust to make sum exactly total_num_samples
        diff = total_num_samples - sum(samples_per_frame)
        for i in range(abs(diff)):
            idx = i % len(self.frames)
            if diff > 0:
                samples_per_frame[idx] += 1
            elif samples_per_frame[idx] > 1:
                samples_per_frame[idx] -= 1
        for frame, n in zip(frames, samples_per_frame):
            fid = frame.get_frame_index()
            self.sample_counts[fid] = self.sample_counts.get(fid, default_count) + n
        return samples_per_frame

    def sample_by_ratio(
        self,
        ratio: float,
        sample_frame_fn: Callable,
        key_frame_indices: list[int],
        current_frame: Optional[Frame] = None,
        to_world_frame: bool = True,
        device: str | None = None,
    ) -> Any:
        """
        Sample from key frames based on a ratio.

        Args:
            ratio: fraction of each frame's valid points to sample.
            sample_frame_fn: function to sample from each frame. e.g. :meth:`Frame.sample_points`.
            key_frame_indices: indices into `self.frames` selecting which key frames to draw from.
            current_frame: extra frame to also sample from (skipped if it is already the last key frame).
            to_world_frame: whether to transform points to world coordinates.
            device: device to use for sampling.

        Returns:
            Sampled data from the selected frames.
        """
        frames: list[Frame] = [self.frames[i] for i in key_frame_indices]
        if current_frame is not None and current_frame.get_frame_index() not in self.kf_indices:
            frames.append(current_frame)
        if device is None:
            device = self.device
        return sample_from_frames(
            frames=frames,
            sample_frame_fn=sample_frame_fn,
            sample_frame_fn_kwargs=dict(ratio=ratio, to_world_frame=to_world_frame, device=device),
        )

    def sample_by_num(
        self,
        total_num_samples: int,
        sample_frame_fn: Callable,
        key_frame_indices: list[int],
        current_frame: Optional[Frame] = None,
        to_world_frame: bool = True,
        device: str | None = None,
    ) -> Any:
        frames: list[Frame] = [self.frames[i] for i in key_frame_indices]
        if current_frame is not None and current_frame.get_frame_index() not in self.kf_indices:
            frames.append(current_frame)
        if not frames:
            return None
        if device is None:
            device = self.device
        samples_per_frame = self.allocate_num_samples_to_frames(total_num_samples, frames)
        sample_frame_fn_kwargs = [
            dict(num_samples=n, to_world_frame=to_world_frame, device=device) for n in samples_per_frame
        ]
        return sample_from_frames(
            frames=frames,
            sample_frame_fn=sample_frame_fn,
            sample_frame_fn_kwargs=sample_frame_fn_kwargs,
        )

    # def sample_points(self, ratio: float, key_frame_indices: list, current_frame: Optional[Frame]) -> torch.Tensor:
    #     """Sample world-frame surface points from selected key frames plus an optional current frame.

    #     Args:
    #         ratio: fraction of each frame's valid points to sample.
    #         key_frame_indices: indices into `self.frames` selecting which key frames to draw from.
    #         current_frame: extra frame to also sample from (skipped if it is already the last key frame).

    #     Returns:
    #         (N, 3) concatenated sampled points in world coordinates.
    #     """
    #     frames: list[Frame] = [self.frames[i] for i in key_frame_indices]
    #     if current_frame is not None and current_frame != self.frames[-1]:
    #         frames.append(current_frame)
    #     points = [frame.sample_points(ratio=ratio, to_world_frame=True, device=self.device) for frame in frames]
    #     points = torch.cat(points, dim=0)
    #     return points

    # def sample_points_and_features(
    #     self,
    #     num_samples: int,
    #     key_frame_indices: list,
    #     current_frame: Optional[Frame],
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Sample paired `(world_points, vl_features)` from selected key frames plus an optional current frame.

    #     Companion to :meth:`sample_points`. Requires each frame to implement `sample_points_and_features` (e.g.
    #     :class:`oren.frame.VlFrame`); raises `AttributeError` otherwise. The total budget `num_samples` is split as
    #     evenly as possible across the participating frames.

    #     Args:
    #         num_samples: total number of (point, feature) pairs to draw across all participating frames.
    #         key_frame_indices: indices into `self.frames` selecting which key frames to draw from.
    #         current_frame: extra frame to also sample from (skipped if it is already the last key frame).

    #     Returns:
    #         Tuple `(points (N, 3), features (N, C))` in world coordinates, with rows aligned row-for-row.
    #     """
    #     frames: list[Frame] = [self.frames[i] for i in key_frame_indices]
    #     if current_frame is not None and (len(self.frames) == 0 or current_frame != self.frames[-1]):
    #         frames.append(current_frame)
    #     if not frames:
    #         return torch.empty((0, 3), device=self.device), torch.empty((0, 0), device=self.device)

    #     n_per_frame = max(1, num_samples // len(frames))
    #     pts_chunks: list[torch.Tensor] = []
    #     feat_chunks: list[torch.Tensor] = []
    #     for frame in frames:
    #         pts, feats = frame.sample_points_and_features(
    #             num_points=n_per_frame,
    #             to_world_frame=True,
    #             device=self.device,
    #         )
    #         if pts.numel() == 0:
    #             continue
    #         pts_chunks.append(pts)
    #         feat_chunks.append(feats)
    #     if not pts_chunks:
    #         return torch.empty((0, 3), device=self.device), torch.empty((0, 0), device=self.device)
    #     return torch.cat(pts_chunks, dim=0), torch.cat(feat_chunks, dim=0)

    # def sample_rays(
    #     self,
    #     num_samples: int,
    #     key_frame_indices: list,
    #     current_frame: Frame | None,
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Sample rays from the key frames. The sampling strategy is set by self.cfg.frame_weight.
    #     When the strategy is "uniform", we sample uniformly from each frame.
    #     Otherwise, we do:
    #         1. Distribute num_samples to each frame based on the sample counts. Higher sample count -> fewer samples.
    #         2. Sample the rays from each frame.
    #         3. Update the sample counts for next sampling.
    #     Args:
    #         num_samples: number of rays to sample.
    #         key_frame_indices: indices of key frames to sample from.
    #         current_frame: the current frame, if not None, we also sample from it.

    #     Returns:
    #         (num_samples, 3) ray origins in world coordinates.
    #         (num_samples, 3) ray directions in world coordinates.
    #         (num_samples,) depth values in meter.
    #     """

    #     # distribute num_samples to each frame based on the sample counts
    #     # higher sample count -> fewer samples

    #     frames: list[Frame] = [self.frames[i] for i in key_frame_indices]
    #     sample_counts = [self.sample_counts[i] for i in key_frame_indices]
    #     if current_frame is not None and current_frame != self.frames[-1]:
    #         frames.append(current_frame)
    #         sample_counts.append(sum(sample_counts) // (len(sample_counts) + 2))

    #     return self.sample_rays_from_frames(num_samples, frames)

    # def sample_rays_from_frames(
    #     self,
    #     num_samples: int,
    #     frames: list[Frame],
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    #     n_frames = len(frames)

    #     if n_frames == 0 or num_samples == 0:
    #         return None, None, None, None

    #     samples_per_frame = self.allocate_num_samples_to_frames(num_samples, frames)

    #     # rays_o, rays_d, depth_samples = self.sample_from_frames(
    #     #     frames=frames,
    #     #     num_samples=samples_per_frame,
    #     #     sample_frame_fn=lambda frame, n: frame.sample_rays(num_samples=n, to_world_frame=True, device=self.device),
    #     # )

    #     rays_o_all = []
    #     rays_d_all = []
    #     depth_samples_all = []
    #     for frame_idx, frame in enumerate(frames):
    #         n_frame_samples = samples_per_frame[frame_idx]

    #         if frame.fid in self.valid_indices:
    #             valid_idx = self.valid_indices[frame.fid]
    #         else:
    #             valid_idx = torch.nonzero(frame.get_valid_mask().view(-1))

    #         if frame.fid in self.sample_counts:
    #             self.sample_counts[frame.fid] += n_frame_samples
    #         else:
    #             self.sample_counts[frame.fid] = n_frame_samples

    #         sample_idx = valid_idx[torch.randint(0, valid_idx.shape[0], (n_frame_samples,))]
    #         sample_idx = sample_idx.view(-1)

    #         pose = frame.get_ref_pose()
    #         rotation = pose[:3, :3]
    #         sampled_rays_d = frame.get_rays_direction().view(-1, 3)[sample_idx]  # (n_frame_samples, 3)
    #         sampled_rays_d = sampled_rays_d @ rotation.T  # (n_frame_samples, 3)
    #         sampled_rays_o = pose[:3, 3].view(1, 3).expand_as(sampled_rays_d)  # (n_frame_samples, 3)
    #         rays_o_all.append(sampled_rays_o)
    #         rays_d_all.append(sampled_rays_d)

    #         sampled_depth = frame.get_depth().view(-1)[sample_idx]  # (n_frame_samples,)
    #         depth_samples_all.append(sampled_depth)

    #     rays_o_all = torch.cat(rays_o_all, dim=0)  # (num_samples, 3)
    #     rays_d_all = torch.cat(rays_d_all, dim=0)  # (num_samples, 3)
    #     depth_samples_all = torch.cat(depth_samples_all, dim=0)  # (num_samples,)
    #     return rays_o_all, rays_d_all, depth_samples_all
