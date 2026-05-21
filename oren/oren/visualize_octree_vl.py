# pyright: reportPrivateImportUsage=none, reportAttributeAccessIssue=none
"""PCA-visualize an octree with scattered VL features as a colorized voxel grid.

Loads a VlTrainer checkpoint (`ckpt/final.pth` + sibling `bak/config.yaml`),
computes a per-voxel feature (mean over the up-to-8 vertex slots) for each
finest-level leaf, reduces to 3 components via PCA, and renders the voxels
as a single Open3D triangle mesh with one cube per voxel and per-cube color
from the PCA projection.
"""

import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d
import torch
from ruamel import yaml
from sklearn.decomposition import PCA

from oren.utils.config_abc import ConfigABC


@dataclass
class VisualizeOctreeConfig(ConfigABC):
    octree_path: str = None  # required: path to ckpt/final.pth (state_dict) saved by VlTrainer
    config_path: Optional[str] = None  # optional override; default = <ckpt>/../../bak/config.yaml
    field_name: str = "vl"  # which field_bank entry to visualize
    pca_path: Optional[str] = None  # optional pca.npz from generate_vl_features; "" disables auto-detect
    save_screenshot: Optional[str] = None  # PNG path for headless render
    save_mesh: Optional[str] = None        # write the PCA-colored cube mesh (.ply / .obj / ...)
    interactive: bool = False              # open an Open3D window (default if no screenshot)


def per_voxel_features(vertex_indices: torch.Tensor, implicit_features: torch.Tensor) -> torch.Tensor:
    """Average implicit features over the (up to 8) valid vertices of each voxel.

    Args:
        vertex_indices: (V, 8) int tensor of per-voxel vertex ids, with -1 marking absent slots.
        implicit_features: (Vmax, C) per-vertex feature tensor indexed by ``vertex_indices``.

    Returns:
        (V, C) tensor of per-voxel mean features over the valid vertex slots.
    """
    valid = vertex_indices >= 0  # (V, 8)
    safe = torch.where(valid, vertex_indices, torch.zeros_like(vertex_indices)).long()
    vert_feats = implicit_features[safe]  # (V, 8, C)
    vert_feats = vert_feats * valid.unsqueeze(-1).float()
    counts = valid.sum(dim=1, keepdim=True).clamp(min=1).float()
    return vert_feats.sum(dim=1) / counts  # (V, C)


def make_voxel_cubes_mesh(centers: np.ndarray, sizes_m: np.ndarray, colors: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Build a single TriangleMesh with one cube per voxel, vertex-colored.

    Args:
        centers: (V, 3) per-voxel center coordinates in meters.
        sizes_m: (V,) per-voxel side lengths in meters.
        colors: (V, 3) RGB triples in [0, 1] applied uniformly to each cube's 8 vertices.

    Returns:
        Open3D TriangleMesh with 8*V vertices, 12*V triangles, and per-vertex colors.
    """
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


def _resolve_config_path(ckpt_path: pathlib.Path, override: Optional[str]) -> pathlib.Path:
    """Locate the trainer config.yaml that pairs with a final.pth checkpoint."""
    if override is not None:
        return pathlib.Path(override)
    # Trainer layout: <run>/ckpt/final.pth and <run>/bak/config.yaml.
    candidate = ckpt_path.parent.parent / "bak" / "config.yaml"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Could not auto-locate config.yaml next to {ckpt_path} (looked at {candidate}); "
            "pass --config-path explicitly."
        )
    return candidate


def _resolve_pca_path(trainer_cfg: dict, override: Optional[str]) -> Optional[pathlib.Path]:
    """Pick the PCA file to load. Explicit override wins; empty string disables. Otherwise auto-locate."""
    if override == "":
        return None
    if override is not None:
        path = pathlib.Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"pca_path override does not exist: {path}")
        return path
    try:
        data_path = trainer_cfg["data"]["dataset_args"]["data_path"]
    except (KeyError, TypeError):
        return None
    candidate = pathlib.Path(data_path) / "pca.npz"
    return candidate if candidate.is_file() else None


def _project_feat(feat: np.ndarray, pca_path: Optional[pathlib.Path]) -> np.ndarray:
    """Project per-voxel features to 3-D either via a saved PCA (mean + components) or a freshly fit one."""
    if pca_path is not None:
        data = np.load(pca_path)
        components = data["components"].astype(np.float64)  # (k, C)
        mean = data["mean"].astype(np.float64)              # (C,)
        print(
            f"Loaded PCA from {pca_path}; "
            f"explained variance ratio: {data['explained_variance_ratio']}"
        )
        return (feat.astype(np.float64) - mean) @ components.T
    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(feat)
    print(f"PCA explained variance ratio (fit on the fly): {pca.explained_variance_ratio_}")
    return feat_pca


def main(cfg: VisualizeOctreeConfig) -> None:
    """Load a VlTrainer checkpoint, PCA-color its VL features per voxel, and render / save the result.

    Args:
        cfg: Visualization configuration with the input checkpoint path and optional save / interactive flags.
    """
    assert cfg.octree_path is not None, "VisualizeOctreeConfig.octree_path is required"

    ckpt_path = pathlib.Path(cfg.octree_path)
    config_path = _resolve_config_path(ckpt_path, cfg.config_path)
    with open(config_path) as f:
        trainer_cfg = yaml.YAML(typ="safe").load(f)
    resolution = float(trainer_cfg["model"]["octree_cfg"]["resolution"])

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    voxels = state["octree.voxels"]                # (N, 4) [x, y, z, discrete_size]
    voxel_centers = state["octree.voxel_centers"]  # (N, 3) meters
    vertex_indices = state["octree.vertex_indices"]  # (N, 8)
    values_key = f"field_bank.fields.{cfg.field_name}.values"
    weight_key = f"field_bank.fields.{cfg.field_name}.prior_fuser.weight_sum"
    if values_key not in state:
        raise KeyError(
            f"Field '{cfg.field_name}' not found in checkpoint. Available field keys: "
            f"{sorted(k for k in state if k.startswith('field_bank.fields.'))}"
        )
    vertex_values = state[values_key]              # (Vmax, C) per-vertex explicit features
    vertex_weights = state.get(weight_key)         # (Vmax,) or None

    N = int(voxels.shape[0])
    # Leaves at the finest level have discrete voxel_size == 1.
    leaf_mask = (voxels[:, -1] == 1).numpy()
    print(
        f"Loaded checkpoint: {N} buffer rows, {int(leaf_mask.sum())} finest leaves; "
        f"field '{cfg.field_name}' values {tuple(vertex_values.shape)}; resolution={resolution} m."
    )

    # Restrict to leaves up front so we don't materialize the (N, 8, C) gather for internal nodes.
    leaf_idx = np.nonzero(leaf_mask)[0]
    leaf_vertex_indices = vertex_indices[leaf_idx]                              # (L, 8)
    feat = per_voxel_features(leaf_vertex_indices, vertex_values).numpy()       # (L, C)

    # Drop leaves that never received a scatter. Prefer the explicit weight_sum
    # when available (zero-norm features and "never updated" are different in
    # principle); fall back to feature norm otherwise.
    if vertex_weights is not None:
        safe = leaf_vertex_indices.clamp(min=0).long()
        valid = (leaf_vertex_indices >= 0).float()
        per_voxel_w = (vertex_weights[safe] * valid).sum(dim=1).numpy()
        keep_leaf = per_voxel_w > 0.0
    else:
        keep_leaf = np.linalg.norm(feat, axis=1) > 1e-8
    print(f"Finest leaves with non-zero VL features: {int(keep_leaf.sum())} / {int(leaf_mask.sum())}.")
    if not keep_leaf.any():
        raise RuntimeError("No finest-level leaf received any VL feature; nothing to visualize.")

    feat = feat[keep_leaf]
    kept_idx = leaf_idx[keep_leaf]
    centers = voxel_centers.numpy()[kept_idx].astype(np.float64)
    sizes_m = voxels[:, -1].numpy()[kept_idx].astype(np.float64) * resolution

    # PCA -> 3 components. Prefer the PCA saved at feature-extraction time so
    # colors are consistent across visualizations of the same scene; fall back
    # to fitting on the leaf features when no saved PCA is available.
    # Rank-normalize per channel so each color axis spans [0, 1] regardless of
    # variance ratios (min-max would collapse low-variance channels).
    pca_file = _resolve_pca_path(trainer_cfg, cfg.pca_path)
    feat_pca = _project_feat(feat, pca_file)
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
