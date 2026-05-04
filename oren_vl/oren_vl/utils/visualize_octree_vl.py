# pyright: reportPrivateImportUsage=none, reportAttributeAccessIssue=none
"""PCA-visualize an octree with scattered VL features as a colorized voxel grid.

Loads the file produced by `demo_semi_sparse_octree_vl`, computes a per-voxel
feature (mean over the up to 8 vertex slots), reduces to 3 components via
PCA, and renders the voxels as a single Open3D triangle mesh with one cube
per voxel and per-cube color from the PCA projection.
"""

import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d
import torch
from sklearn.decomposition import PCA

from oren.utils.config_abc import ConfigABC


@dataclass
class VisualizeOctreeConfig(ConfigABC):
    octree_path: str = None  # required: .pt file produced by demo_semi_sparse_octree_vl
    save_screenshot: Optional[str] = None  # PNG path for headless render
    save_mesh: Optional[str] = None        # write the PCA-colored cube mesh (.ply / .obj / ...)
    interactive: bool = False              # open an Open3D window (default if no screenshot)


def per_voxel_features(vertex_indices: torch.Tensor, residual_features: torch.Tensor) -> torch.Tensor:
    """Average residual features over the (up to 8) valid vertices of each voxel."""
    valid = vertex_indices >= 0  # (V, 8)
    safe = torch.where(valid, vertex_indices, torch.zeros_like(vertex_indices)).long()
    vert_feats = residual_features[safe]  # (V, 8, C)
    vert_feats = vert_feats * valid.unsqueeze(-1).float()
    counts = valid.sum(dim=1, keepdim=True).clamp(min=1).float()
    return vert_feats.sum(dim=1) / counts  # (V, C)


def make_voxel_cubes_mesh(centers: np.ndarray, sizes_m: np.ndarray, colors: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Build a single TriangleMesh with one cube per voxel, vertex-colored."""
    V = centers.shape[0]
    base_v = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    base_t = np.array(
        [
            [0, 2, 1], [1, 2, 3],   # -z face
            [4, 5, 6], [5, 7, 6],   # +z face
            [0, 1, 4], [1, 5, 4],   # -y face
            [2, 6, 3], [3, 6, 7],   # +y face
            [0, 4, 2], [2, 4, 6],   # -x face
            [1, 3, 5], [3, 7, 5],   # +x face
        ],
        dtype=np.int64,
    )

    sizes_m = sizes_m.reshape(-1, 1, 1)
    vertices = (base_v[None] * sizes_m + centers[:, None, :]).reshape(-1, 3)
    triangles = (base_t[None] + (np.arange(V, dtype=np.int64)[:, None, None] * 8)).reshape(-1, 3)
    vertex_colors = np.repeat(colors, 8, axis=0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.compute_vertex_normals()
    return mesh


def main(cfg: VisualizeOctreeConfig) -> None:
    assert cfg.octree_path is not None, "VisualizeOctreeConfig.octree_path is required"

    state = torch.load(cfg.octree_path, map_location="cpu", weights_only=False)

    voxels = state["voxels"]                       # (N, 4) [x, y, z, discrete_size]
    voxel_centers = state["voxel_centers"]         # (N, 3) meters
    vertex_indices = state["vertex_indices"]       # (N, 8)
    residual_features = state["residual_features"] # (Vmax, C)
    octree_cfg = state["octree_cfg"]
    resolution = float(octree_cfg["resolution"])

    N = int(voxels.shape[0])
    # Leaves at the finest level have discrete voxel_size == 1.
    leaf_mask = (voxels[:, -1] == 1).numpy()
    print(
        f"Loaded octree: {N} buffer rows, {int(leaf_mask.sum())} finest leaves; "
        f"residual features {tuple(residual_features.shape)}."
    )

    # Per-node feature (averaged over the up-to-8 valid vertex slots).
    feat = per_voxel_features(vertex_indices, residual_features).numpy()  # (N, C)

    # Render only leaves with non-zero features. Internal nodes overlap their
    # children, so we drop them; nodes that never got a feature scatter are
    # also skipped (zero-norm rows).
    norms = np.linalg.norm(feat, axis=1)
    keep = leaf_mask & (norms > 1e-8)
    print(f"Finest leaves with non-zero VL features: {int(keep.sum())} / {int(leaf_mask.sum())}.")
    if not keep.any():
        raise RuntimeError("No finest-level leaf received any VL feature; nothing to visualize.")

    feat = feat[keep]
    centers = voxel_centers.numpy()[keep].astype(np.float64)
    sizes_m = voxels[:, -1].numpy()[keep].astype(np.float64) * resolution

    # PCA -> 3 components. Use rank-based (quantile) normalization per channel
    # so each color axis spans [0, 1] regardless of the variance ratios.
    # Min-max would collapse low-variance channels to a thin band when one
    # principal direction dominates (typical for semantically uniform scenes).
    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(feat)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    ranks = feat_pca.argsort(axis=0).argsort(axis=0)
    rgb = ranks.astype(np.float64) / max(1, feat_pca.shape[0] - 1)

    mesh = make_voxel_cubes_mesh(centers, sizes_m, rgb.astype(np.float64))

    if cfg.save_mesh is not None:
        pathlib.Path(cfg.save_mesh).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(cfg.save_mesh, mesh)
        print(f"Mesh saved to {cfg.save_mesh}")

    do_screenshot = cfg.save_screenshot is not None
    do_interactive = cfg.interactive or not do_screenshot

    if do_screenshot:
        from open3d.visualization import rendering as o3dr

        renderer = o3dr.OffscreenRenderer(1920, 1080)
        mat = o3dr.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("octree", mesh, mat)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        # Robust bbox from voxel centers: trim 1st/99th percentiles so a stray
        # leaf at an octree extreme doesn't blow up the framing.
        lo = np.percentile(centers, 1, axis=0)
        hi = np.percentile(centers, 99, axis=0)
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
        o3d.visualization.draw_geometries([mesh], window_name="Octree VL features (PCA)")


if __name__ == "__main__":
    parser = VisualizeOctreeConfig.get_argparser()
    cfg, _ = parser.parse_known_args()
    main(cfg)
