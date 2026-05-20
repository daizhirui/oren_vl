import json
import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset

from oren.frame import VlFrame
from oren.utils.depth_utils import depth_to_camera_points


class DataLoader(Dataset):
    """Loads a VL feature bundle written by `generate_vl_features`.

    Returns one :class:`VlFrame` per index: depth-backprojected camera-frame points paired with their per-pixel VL
    features, plus the frame's `T_wc` pose. Features and depth are read from memory-mapped binaries; poses are loaded
    eagerly. Backprojection happens lazily on each `__getitem__` call (cheap relative to the feature dim x H x W copy).

    Constructor surface mirrors :class:`oren.dataset.replica.DataLoader` so this drops in via
    :class:`oren.dataset.data_config.DataConfig` -> :func:`oren.utils.import_util.get_dataset` exactly like the SDF /
    OCC loaders:

      - `min_depth` / `max_depth` filter the raw depth map (replica zeroes out-of-range pixels; we do the same so the
        downstream `depth > 0` filter handles exclusion uniformly). `max_depth = -1` disables the upper bound.
      - `apply_bound` controls whether the returned :class:`VlFrame` is pre-narrowed to `[bound_min, bound_max]`.
      - `bound_min` / `bound_max` are loaded from the manifest (`bound_min`, `bound_max` keys, each a list of 3 floats)
        when not provided explicitly. The manifest is the canonical source -- callers can override per-run by passing
        the kwargs (e.g. via `DataConfig.dataset_args`).
    """

    streaming: bool = False  # consumed by TrainerBase to pick between streaming / bounded loops.

    def __init__(
        self,
        data_path: str,
        min_depth: float = 0.0,
        max_depth: float = -1.0,
        apply_bound: bool = False,
        bound_min: torch.Tensor | None = None,
        bound_max: torch.Tensor | None = None,
    ):
        """Memory-map the feature / depth binaries, load poses, and resolve bounds.

        Args:
            data_path: Path to the directory containing the VL feature bundle (manifest, `.bin` files, poses).
            min_depth: minimum valid depth in meters; values below are zeroed (i.e. excluded by the `depth > 0`
                filter inside :func:`depth_to_camera_points`). 0 means "no lower cap beyond zero".
            max_depth: maximum valid depth in meters; values above are zeroed. Use `-1` to disable.
            apply_bound: if True, crop each returned :class:`VlFrame` to `[bound_min, bound_max]` in world coordinates.
            bound_min: optional (3,) lower bound; if None, loaded from `manifest["bound_min"]`.
            bound_max: optional (3,) upper bound; if None, loaded from `manifest["bound_max"]`.
        """
        data_path = osp.expanduser(data_path)
        data_path = osp.abspath(data_path)
        data_path = data_path.rstrip("/")

        self.data_path = data_path
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.apply_bound = apply_bound
        self.bound_min = bound_min
        self.bound_max = bound_max

        with open(os.path.join(data_path, "vl_features_manifest.json")) as f:
            self.manifest = json.load(f)

        feat_meta = self.manifest["features"]
        depth_meta = self.manifest["depth_resampled"]
        traj_meta = self.manifest["trajectory"]

        # Features are stored on disk in NHWC layout (see `generate_vl_features.py`'s memmap write): shape is
        # (n_frames, h_feat, w_feat, channels). Older bundles with `manifest["features"]["layout"] != "NHWC"` (or no
        # `layout` key, which used to mean NCHW) are explicitly rejected -- regenerate them with the current
        # `generate_vl_features.py`.
        feat_shape = tuple(feat_meta["shape"])
        depth_shape = tuple(depth_meta["shape"])
        layout = feat_meta.get("layout", "NCHW")
        assert layout == "NHWC", (
            f"DataLoader expects NHWC feature layout (n_frames, h_feat, w_feat, channels); manifest says "
            f"layout={layout!r}, shape={feat_shape}. Regenerate the bundle with the current generate_vl_features.py."
        )
        self.n_frames, self.h_feat, self.w_feat, self.channels = feat_shape

        self.features = np.memmap(
            os.path.join(data_path, feat_meta["file"]),
            dtype=np.dtype(feat_meta["dtype"]),
            mode="r",
            shape=feat_shape,
        )
        self.depth = np.memmap(
            os.path.join(data_path, depth_meta["file"]),
            dtype=np.dtype(depth_meta["dtype"]),
            mode="r",
            shape=depth_shape,
        )
        self.poses = np.load(os.path.join(data_path, traj_meta["file"]))
        assert self.poses.shape == (
            self.n_frames,
            4,
            4,
        ), f"poses.npy has shape {self.poses.shape}, expected ({self.n_frames}, 4, 4)"
        self.K_feat = np.array(feat_meta["K_feat"], dtype=np.float32)
        # Torch view of the intrinsics, reused by every `__getitem__` call. Lives on CPU; backprojection happens on CPU
        # and the consumer (`VlFrame`) is responsible for moving the tensors to GPU.
        self._K_feat_t = torch.from_numpy(self.K_feat)

        # Bounds: explicit kwargs win; otherwise fall back to the manifest.
        if self.bound_min is None or self.bound_max is None:
            self.bound_min = self.manifest["bound_min"]
            self.bound_max = self.manifest["bound_max"]

        self.bound_min = torch.tensor(self.bound_min).float()
        self.bound_max = torch.tensor(self.bound_max).float()

    def load_depth(self, index: int) -> torch.Tensor:
        """Load and depth-filter the per-frame depth map.

        Mirrors :meth:`oren.dataset.replica.DataLoader.load_depth`: out-of-range pixels are zeroed so the downstream
        `depth > 0` filter inside :func:`depth_to_camera_points` excludes them with no extra mask plumbing.

        Args:
            index: frame index (zero-based).

        Returns:
            depth: (h_feat, w_feat) float depth in meters with out-of-range pixels zeroed.
        """
        depth = torch.from_numpy(self.depth[index].copy()).float()
        if self.min_depth >= 0:
            depth[depth < self.min_depth] = 0
        if self.max_depth > 0:
            depth[depth > self.max_depth] = 0
        return depth

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> VlFrame:
        """Build a :class:`VlFrame` for frame `idx`.

        Backproject the depth map to camera-frame points and flatten the `(h, w, C)` feature grid into per-valid-pixel
        feature vectors aligned row-for-row with the points. When `apply_bound=True` the returned frame's `valid_mask`
        is pre-narrowed to `[self.bound_min, self.bound_max]`.
        """
        feat = torch.from_numpy(self.features[idx].copy()).float()  # (h, w, C)
        depth = self.load_depth(idx)  # (h, w) meters, out-of-range pixels zeroed
        pose = torch.from_numpy(self.poses[idx]).float()  # (4, 4) T_wc

        valid_mask = depth > 0  # (h, w)
        pts_cam = depth_to_camera_points(depth, self._K_feat_t)[valid_mask]  # (N, 3)
        feat_per_pixel = feat.reshape(-1, self.channels)[valid_mask.flatten()]
        frame = VlFrame(fid=idx, points=pts_cam, vl_features=feat_per_pixel, ref_pose=pose)
        if self.apply_bound:
            frame.apply_bound(self.bound_min, self.bound_max)
        return frame
