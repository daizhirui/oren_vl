import json
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class VLFeaturesDataset(Dataset):
    """Loads a VL feature bundle written by `generate_vl_features`.

    Returns per-frame (idx, features, depth, T_wc) tuples. Features and depth
    are read from memory-mapped binaries; poses are loaded eagerly.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

        with open(os.path.join(output_dir, "vl_features_manifest.json")) as f:
            self.manifest = json.load(f)

        feat_meta = self.manifest["features"]
        depth_meta = self.manifest["depth_resampled"]
        traj_meta = self.manifest["trajectory"]

        feat_shape = tuple(feat_meta["shape"])
        depth_shape = tuple(depth_meta["shape"])
        self.n_frames, self.channels, self.h_feat, self.w_feat = feat_shape

        self.features = np.memmap(
            os.path.join(output_dir, feat_meta["file"]),
            dtype=np.dtype(feat_meta["dtype"]),
            mode="r",
            shape=feat_shape,
        )
        self.depth = np.memmap(
            os.path.join(output_dir, depth_meta["file"]),
            dtype=np.dtype(depth_meta["dtype"]),
            mode="r",
            shape=depth_shape,
        )
        self.poses = np.load(os.path.join(output_dir, traj_meta["file"]))
        assert self.poses.shape == (self.n_frames, 4, 4), (
            f"poses.npy has shape {self.poses.shape}, expected ({self.n_frames}, 4, 4)"
        )
        self.K_feat = np.array(feat_meta["K_feat"], dtype=np.float32)

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int):
        feat = torch.from_numpy(self.features[idx].copy())  # (C, h, w), original dtype
        depth = torch.from_numpy(self.depth[idx].copy())    # (h, w) float32, meters
        pose = torch.from_numpy(self.poses[idx])            # (4, 4) float32, T_wc
        return idx, feat, depth, pose


def depth_to_world_points(
    depth: torch.Tensor,
    pose: torch.Tensor,
    K: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backproject a depth map to world-space points.

    Args:
        depth: (H, W) meters.
        pose: (4, 4) cam->world (T_wc).
        K: (3, 3) intrinsics at the depth/feature scale.

    Returns:
        world_points: (M, 3) where M is the count of pixels with depth > 0.
        valid_mask: (H, W) bool.
    """
    H, W = depth.shape
    device = depth.device
    dtype = depth.dtype
    u = torch.arange(W, device=device, dtype=dtype) + 0.5
    v = torch.arange(H, device=device, dtype=dtype) + 0.5
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (uu - cx) * depth / fx
    y_cam = (vv - cy) * depth / fy
    z_cam = depth
    pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (H, W, 3)
    R = pose[:3, :3]
    t = pose[:3, 3]
    pts_world = pts_cam @ R.T + t  # (H, W, 3)
    valid_mask = depth > 0
    return pts_world[valid_mask].view(-1, 3), valid_mask
