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
from sklearn.decomposition import PCA
from tqdm import tqdm

from oren.dataset.vl_feature_dataset import DataLoader as VlFeatureDataset
from oren.utils.config_abc import ConfigABC


@dataclass
class VisualizePcdConfig(ConfigABC):
    vl_features_dir: str = None  # required
    frame_stride: int = 1
    pca_fit_samples: int = 30000
    pca_path: Optional[str] = None  # pca.npz from generate_vl_features; "" forces a fresh fit
    voxel_downsample: float = -1.0  # >0 enables o3d voxel_down_sample at this size (m)
    save_screenshot: Optional[str] = None  # PNG path for headless render
    save_pcd: Optional[str] = None  # write the PCA-colored cloud (.ply / .pcd / ...)
    interactive: bool = False  # open an Open3D window (default if no screenshot)
    seed: int = 0


class _SavedPCA:
    """Minimal duck-typed PCA replacement: ``transform`` matches sklearn's ``(X - mean_) @ components_.T``."""

    def __init__(self, components: np.ndarray, mean: np.ndarray, explained_variance_ratio: np.ndarray):
        self.components_ = components.astype(np.float32)
        self.mean_ = mean.astype(np.float32)
        self.explained_variance_ratio_ = explained_variance_ratio

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X.astype(self.components_.dtype) - self.mean_) @ self.components_.T


def _load_pca(path: pathlib.Path) -> _SavedPCA:
    """Load a PCA saved by ``generate_vl_features`` into a sklearn-compatible projector."""
    data = np.load(path)
    print(f"Loaded PCA from {path}; explained variance ratio: {data['explained_variance_ratio']}")
    return _SavedPCA(data["components"], data["mean"], data["explained_variance_ratio"])


def _resolve_pca_path(vl_features_dir: str, override: Optional[str]) -> Optional[pathlib.Path]:
    """Pick the PCA file to load. Explicit override wins; empty string disables. Otherwise auto-locate."""
    if override == "":
        return None
    if override is not None:
        path = pathlib.Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"pca_path override does not exist: {path}")
        return path
    candidate = pathlib.Path(vl_features_dir) / "pca.npz"
    return candidate if candidate.is_file() else None


def fit_pca(dataset: VlFeatureDataset, stride: int, n_samples: int, seed: int) -> PCA:
    """Pass 1: fit PCA on features sampled approximately uniformly across frames.

    Args:
        dataset: VL feature bundle to draw samples from.
        stride: Only every ``stride``-th frame is sampled.
        n_samples: Approximate total number of feature vectors to feed into ``PCA.fit``.
        seed: Seed for the per-frame uniform sampler.

    Returns:
        Fitted ``sklearn.decomposition.PCA`` with ``n_components=3``.
    """
    indices = list(range(0, len(dataset), stride))
    per_frame = max(1, n_samples // max(1, len(indices)))
    rng = np.random.default_rng(seed)

    chunks = []
    for idx in tqdm(indices, desc="PCA fit pass", ncols=80):
        frame = dataset[idx]
        feat_flat = frame.get_vl_features(valid_only=True).float()  # (M, C)
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
    dataset: VlFeatureDataset,
    stride: int,
    pca: PCA,
):
    """Pass 2: backproject + apply PCA per-frame, accumulating only (xyz, rgb_pca).

    Args:
        dataset: VL feature bundle providing per-frame :class:`VlFrame` objects with pre-backprojected
            camera-frame points and per-point VL features.
        stride: Only every ``stride``-th frame is processed.
        pca: PCA fitted by ``fit_pca``; used to project features to 3-D RGB-space coordinates.

    Returns:
        pts: (M, 3) world-frame point coordinates.
        rgb_pca: (M, 3) PCA-projected feature coordinates aligned with ``pts``.
    """
    pts_chunks = []
    rgb_chunks = []
    indices = list(range(0, len(dataset), stride))
    for idx in tqdm(indices, desc="Backproject + transform", ncols=80):
        frame = dataset[idx]
        pts_world = frame.get_points(to_world_frame=True, device="cpu")  # (M, 3)
        if pts_world.numel() == 0:
            continue
        feat_flat = frame.get_vl_features(valid_only=True, device="cpu").float().numpy()  # (M, C)
        rgb = pca.transform(feat_flat)  # (M, 3)
        pts_chunks.append(pts_world.numpy())
        rgb_chunks.append(rgb)
    pts = np.concatenate(pts_chunks, axis=0)
    rgb_pca = np.concatenate(rgb_chunks, axis=0)
    return pts, rgb_pca


def main(cfg: VisualizePcdConfig) -> None:
    """Fit a PCA over the bundle's VL features, build a PCA-colored point cloud, and render / save it.

    Args:
        cfg: Visualization configuration with the input bundle path and optional save / interactive flags.
    """
    assert cfg.vl_features_dir is not None, "VisualizePcdConfig.vl_features_dir is required"

    dataset = VlFeatureDataset(cfg.vl_features_dir)
    print(f"Loaded {len(dataset)} frames; feature {dataset.channels}-d at " f"{dataset.h_feat}x{dataset.w_feat}.")

    pca_file = _resolve_pca_path(cfg.vl_features_dir, cfg.pca_path)
    pca = _load_pca(pca_file) if pca_file is not None else fit_pca(
        dataset, cfg.frame_stride, cfg.pca_fit_samples, cfg.seed
    )
    pts, rgb_pca = stream_points_and_rgb(dataset, cfg.frame_stride, pca)
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
