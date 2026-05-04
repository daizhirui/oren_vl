# pyright: reportPrivateImportUsage=none, reportAttributeAccessIssue=none
"""PCA-visualize a VL feature bundle as a single colorized point cloud.

For every frame in the bundle, backproject depth into world points, then
project the per-pixel VL feature through a PCA (fit on a subsample of all
features) to get an RGB color. The result is one Open3D point cloud.

Two-pass design keeps peak memory bounded:
  1. Fit PCA on ~`pca_fit_samples` features sampled across frames.
  2. Stream over the dataset: backproject + transform per frame, accumulate
     only (xyz, rgb) — never the full feature matrix.
"""

import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from oren.utils.config_abc import ConfigABC
from oren_vl.dataset.vl_features_dataset import VLFeaturesDataset, depth_to_world_points


@dataclass
class VisualizePcdConfig(ConfigABC):
    vl_features_dir: str = None  # required
    frame_stride: int = 1
    pca_fit_samples: int = 30000
    voxel_downsample: float = -1.0  # >0 enables o3d voxel_down_sample at this size (m)
    save_screenshot: Optional[str] = None  # PNG path for headless render
    save_pcd: Optional[str] = None         # write the PCA-colored cloud (.ply / .pcd / ...)
    interactive: bool = False              # open an Open3D window (default if no screenshot)
    seed: int = 0


def fit_pca(dataset: VLFeaturesDataset, stride: int, n_samples: int, seed: int) -> PCA:
    """Pass 1: fit PCA on features sampled approximately uniformly across frames."""
    indices = list(range(0, len(dataset), stride))
    per_frame = max(1, n_samples // max(1, len(indices)))
    rng = np.random.default_rng(seed)

    chunks = []
    for idx in tqdm(indices, desc="PCA fit pass", ncols=80):
        _, feat, depth, _ = dataset[idx]
        feat = feat.float()  # (C, h, w)
        valid = (depth > 0).flatten()
        feat_flat = feat.permute(1, 2, 0).reshape(-1, feat.shape[0])[valid]  # (M, C)
        if feat_flat.shape[0] == 0:
            continue
        take = min(per_frame, feat_flat.shape[0])
        sel = rng.choice(feat_flat.shape[0], take, replace=False)
        chunks.append(feat_flat[sel].numpy())

    if not chunks:
        raise RuntimeError("No valid pixels found for PCA fitting.")
    fit_data = np.concatenate(chunks, axis=0)
    print(f"Fitting PCA on {fit_data.shape[0]} samples of {fit_data.shape[1]}-d features.")
    pca = PCA(n_components=3)
    pca.fit(fit_data)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    return pca


def stream_points_and_rgb(
    dataset: VLFeaturesDataset,
    K_feat: torch.Tensor,
    stride: int,
    pca: PCA,
):
    """Pass 2: backproject + apply PCA per-frame, accumulating only (xyz, rgb_pca)."""
    pts_chunks = []
    rgb_chunks = []
    indices = list(range(0, len(dataset), stride))
    for idx in tqdm(indices, desc="Backproject + transform", ncols=80):
        _, feat, depth, pose = dataset[idx]
        feat = feat.float()
        pts_world, valid = depth_to_world_points(depth, pose, K_feat)
        if pts_world.numel() == 0:
            continue
        feat_flat = feat.permute(1, 2, 0).reshape(-1, feat.shape[0])
        feat_flat = feat_flat[valid.flatten()].numpy()
        rgb = pca.transform(feat_flat)  # (M, 3)
        pts_chunks.append(pts_world.numpy())
        rgb_chunks.append(rgb)
    pts = np.concatenate(pts_chunks, axis=0)
    rgb_pca = np.concatenate(rgb_chunks, axis=0)
    return pts, rgb_pca


def main(cfg: VisualizePcdConfig) -> None:
    assert cfg.vl_features_dir is not None, "VisualizePcdConfig.vl_features_dir is required"

    dataset = VLFeaturesDataset(cfg.vl_features_dir)
    K_feat = torch.from_numpy(dataset.K_feat)
    print(f"Loaded {len(dataset)} frames; feature {dataset.channels}-d at "
          f"{dataset.h_feat}x{dataset.w_feat}.")

    pca = fit_pca(dataset, cfg.frame_stride, cfg.pca_fit_samples, cfg.seed)
    pts, rgb_pca = stream_points_and_rgb(dataset, K_feat, cfg.frame_stride, pca)
    print(f"Collected {pts.shape[0]} points.")

    # Rank-normalize per channel so each color axis spans [0, 1].
    ranks = rgb_pca.argsort(axis=0).argsort(axis=0)
    rgb = ranks.astype(np.float64) / max(1, rgb_pca.shape[0] - 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    if cfg.voxel_downsample > 0:
        pcd = pcd.voxel_down_sample(cfg.voxel_downsample)
        print(f"Voxel-downsampled to {len(pcd.points)} points at {cfg.voxel_downsample} m.")

    if cfg.save_pcd is not None:
        pathlib.Path(cfg.save_pcd).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(cfg.save_pcd, pcd)
        print(f"Point cloud saved to {cfg.save_pcd}")

    do_screenshot = cfg.save_screenshot is not None
    do_interactive = cfg.interactive or not do_screenshot

    if do_screenshot:
        from open3d.visualization import rendering as o3dr

        renderer = o3dr.OffscreenRenderer(1920, 1080)
        mat = o3dr.MaterialRecord()
        mat.shader = "defaultUnlit"  # point clouds use unlit
        mat.point_size = 4.0
        renderer.scene.add_geometry("pcd", pcd, mat)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        # Robust bbox from points: trim 1st/99th percentiles to ignore stray
        # backprojections from bad depth.
        lo = np.percentile(pts, 1, axis=0)
        hi = np.percentile(pts, 99, axis=0)
        center = (lo + hi) * 0.5
        diag = float(np.linalg.norm(hi - lo))
        direction = np.array([0.6, -0.6, 0.5])
        direction /= np.linalg.norm(direction)
        eye = center + direction * diag * 1.6
        renderer.setup_camera(60.0, center.tolist(), eye.tolist(), [0.0, 0.0, 1.0])
        img = renderer.render_to_image()
        o3d.io.write_image(cfg.save_screenshot, img)
        print(f"Screenshot saved to {cfg.save_screenshot} (robust extent={hi - lo}, diag={diag:.2f}m)")

    if do_interactive:
        o3d.visualization.draw_geometries([pcd], window_name="VL features point cloud (PCA)")


if __name__ == "__main__":
    parser = VisualizePcdConfig.get_argparser()
    cfg, _ = parser.parse_known_args()
    main(cfg)
