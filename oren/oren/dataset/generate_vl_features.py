import importlib
import json
import os
import pathlib
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor, to_tensor
from tqdm import tqdm

from oren.utils.config_abc import ConfigABC
from oren.utils.depth_utils import depth_to_world_points


@dataclass
class VLFeatureConfig(ConfigABC):
    extracter_name: str = "clip"
    extracter_kwargs: dict | None = None  # genuinely optional: None == {}
    input_dir: str = None
    depth_img_prefix: str = "depth/"
    depth_img_ext: str = "png"
    rgb_img_prefix: str = "rgb/"
    rgb_img_ext: str = "jpg"
    img_name_format: str = "{prefix}{idx:06d}.{ext}"
    traj_filename: str = "traj.txt"  # one 4x4 row-major matrix (T_wc) per line
    intrinsics_fx: float = None
    intrinsics_fy: float = None
    intrinsics_cx: float = None
    intrinsics_cy: float = None
    depth_scale: float = 0.001  # raw depth value -> meters
    batch_size: int = 16
    num_workers: int = 4
    output_dir: str = None


def _frame_path(cfg: VLFeatureConfig, kind: str, idx: int) -> str:
    assert cfg.input_dir is not None
    if kind == "rgb":
        prefix, ext = cfg.rgb_img_prefix, cfg.rgb_img_ext
    elif kind == "depth":
        prefix, ext = cfg.depth_img_prefix, cfg.depth_img_ext
    else:
        raise ValueError(kind)
    return os.path.join(cfg.input_dir, cfg.img_name_format.format(prefix=prefix, idx=idx, ext=ext))


def _count_frames(cfg: VLFeatureConfig) -> int:
    n = 0
    while os.path.exists(_frame_path(cfg, "rgb", n)):
        n += 1
    return n


def _load_traj(path: str, expected_n: int) -> np.ndarray:
    """Replica-style traj.txt: each line is a flattened 4x4 row-major matrix (16 numbers)."""
    arr = np.loadtxt(path).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    assert arr.shape[1] == 16, f"Expected 16 numbers per row in {path}, got {arr.shape[1]}"
    assert arr.shape[0] == expected_n, f"{path} has {arr.shape[0]} poses but found {expected_n} RGB frames"
    return arr.reshape(-1, 4, 4)


class _FrameDataset(Dataset):
    def __init__(self, cfg: VLFeatureConfig, n_frames: int, rgb_transform, depth_transform):
        """Lazily load RGB / depth image pairs from disk for a single capture sequence.

        Args:
            cfg: Capture-time configuration with file naming and intrinsics info.
            n_frames: Number of RGB / depth pairs available under ``cfg.input_dir``.
            rgb_transform: Torchvision transform applied to each loaded RGB tensor.
            depth_transform: Torchvision transform applied to each loaded depth tensor.
        """
        self.cfg = cfg
        self.n_frames = n_frames
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        rgb = to_tensor(Image.open(_frame_path(self.cfg, "rgb", idx)).convert("RGB"))
        depth = pil_to_tensor(Image.open(_frame_path(self.cfg, "depth", idx))).float()
        return idx, self.rgb_transform(rgb), self.depth_transform(depth)


def generate_vl_features(cfg: VLFeatureConfig):
    """Extract VL features and resampled depth for every frame in a Replica-style capture and write a bundle to disk.

    Args:
        cfg: Configuration with input dir, output dir, intrinsics, extractor name, and DataLoader parameters.
    """
    assert cfg.input_dir is not None, "VLFeatureConfig.input_dir is required"
    assert cfg.output_dir is not None, "VLFeatureConfig.output_dir is required"

    output_dir = pathlib.Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracter_module = importlib.import_module(f"oren.dataset.vl_features.{cfg.extracter_name}")
    extracter = extracter_module.Extracter(**(cfg.extracter_kwargs or {}))
    extracter.eval()

    n_frames = _count_frames(cfg)
    assert n_frames > 0, f"No RGB frames found under {cfg.input_dir} with prefix '{cfg.rgb_img_prefix}'"

    poses = _load_traj(os.path.join(cfg.input_dir, cfg.traj_filename), n_frames)
    np.save(output_dir / "poses.npy", poses)

    with Image.open(_frame_path(cfg, "rgb", 0)) as im:
        orig_w, orig_h = im.size

    fx, fy, cx, cy = (
        cfg.intrinsics_fx,
        cfg.intrinsics_fy,
        cfg.intrinsics_cx,
        cfg.intrinsics_cy,
    )

    rgb_tf, depth_tf, fx_feat, fy_feat, cx_feat, cy_feat = extracter.get_transform_and_intrinsics(
        orig_w, orig_h, fx, fy, cx, cy
    )

    dataset = _FrameDataset(cfg, n_frames, rgb_tf, depth_tf)

    # Probe feature shape with a single forward pass. The extractor returns (B, C, h, w); we persist the per-frame
    # feature map in HWC layout (h_feat, w_feat, channels) so the dataset reader's `(h*w, C)` reshape is a single
    # contiguous view -- no extra permute / non-contiguous copy on the hot path.
    _, rgb0, depth0 = dataset[0]
    with torch.no_grad():
        feat0 = extracter.model_forward_fn(rgb0.unsqueeze(0))
    _, channels, h_feat, w_feat = feat0.shape
    assert depth0.shape[-2:] == (h_feat, w_feat), (
        f"depth_transform produced {tuple(depth0.shape[-2:])}, but feature map is {(h_feat, w_feat)}"
    )

    K_feat = [
        [float(fx_feat), 0.0, float(cx_feat)],
        [0.0, float(fy_feat), float(cy_feat)],
        [0.0, 0.0, 1.0],
    ]
    K_feat_t = torch.tensor(K_feat, dtype=torch.float32)

    feat_mm = np.memmap(
        output_dir / "features.bin",
        dtype=np.float16,
        mode="w+",
        shape=(n_frames, h_feat, w_feat, channels),
    )
    depth_mm = np.memmap(
        output_dir / "depth.bin",
        dtype=np.float32,
        mode="w+",
        shape=(n_frames, h_feat, w_feat),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    depth0_np = depth0.squeeze(0).numpy()
    valid0 = depth0_np > 0
    points0 = depth_to_world_points(
        torch.from_numpy(depth0_np),
        torch.from_numpy(poses[0]),
        K_feat_t,
    ).numpy()[valid0]  # (M, 3)
    bound_min = np.min(points0, axis=0)
    bound_max = np.max(points0, axis=0)

    with torch.no_grad():
        for idxs, rgbs, depths in tqdm(loader, desc="Extracting VL features", ncols=80):
            feats = extracter.model_forward_fn(rgbs)  # (B, C, h_feat, w_feat)
            # Persist as HWC: permute then contiguous so the memmap write is a flat copy and the on-disk layout matches
            # the dataset reader's expectation. `.half()` already on GPU keeps the GPU->CPU transfer at fp16.
            feats_hwc = feats.detach().half().permute(0, 2, 3, 1).contiguous()  # (B, h_feat, w_feat, C)
            depths_feat = depths.squeeze(1).float() * cfg.depth_scale  # (B, h_feat, w_feat)

            feats_np = feats_hwc.cpu().numpy()
            depths_np = depths_feat.detach().float().cpu().numpy()
            for i, idx in enumerate(idxs.tolist()):
                feat_mm[idx] = feats_np[i]
                depth_mm[idx] = depths_np[i]

                valid = depths_np[i] > 0
                points = depth_to_world_points(
                    torch.from_numpy(depths_np[i]),
                    torch.from_numpy(poses[idx]),
                    K_feat_t,
                ).numpy()[valid]  # (M, 3)
                bound_min = np.minimum(bound_min, points.min(axis=0))
                bound_max = np.maximum(bound_max, points.max(axis=0))

    feat_mm.flush()
    depth_mm.flush()

    # add padding
    bound_min -= 0.5
    bound_max += 0.5

    manifest = {
        "n_frames": n_frames,
        "extracter": {
            "name": cfg.extracter_name,
            "kwargs": cfg.extracter_kwargs or {},
        },
        "capture": {
            "rgb_size": [orig_w, orig_h],
            "K": [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            "depth_scale": cfg.depth_scale,
        },
        "features": {
            "file": "features.bin",
            "shape": [n_frames, h_feat, w_feat, channels],
            "dtype": "float16",
            "layout": "NHWC",
            "K_feat": K_feat,
        },
        "depth_resampled": {
            "file": "depth.bin",
            "shape": [n_frames, h_feat, w_feat],
            "dtype": "float32",
            "unit": "meters",
        },
        "trajectory": {
            "file": "poses.npy",
            "shape": [n_frames, 4, 4],
            "dtype": "float32",
            "convention": "T_wc",
            "format": "matrix_4x4_row_major",
            "source": cfg.traj_filename,
        },
        "bound_min": bound_min.tolist(),
        "bound_max": bound_max.tolist(),
    }
    with open(output_dir / "vl_features_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    parser = VLFeatureConfig.get_argparser()
    cfg, _ = parser.parse_known_args()
    generate_vl_features(cfg)
