# pyright: reportPrivateImportUsage=none
"""Build a SemiSparseOctree whose per-vertex storage is a scattered VL field.

This demo:
  1. Loads a VL feature bundle (written by `generate_vl_features`).
  2. Backprojects each frame's depth into world points.
  3. Inserts those points into a SemiSparseOctree (geometry only).
  4. Scatters per-pixel VL features into a `FieldStorage(name="vl", D=C,
     mode="explicit", gradient_augmentation=False)` at the nearest vertex
     of the containing leaf voxel, either by overwrite or running average.
  5. Saves the populated field's `values` tensor + octree state to disk.

Phase 1 of the FieldStorage refactor (DESIGN.md): VL becomes a first-class
explicit field stored in a `FieldStorage`, not a side-load into the
octree's legacy `implicit_features` parameter. The octree itself carries
no per-vertex field state in this demo - those toggles (`enable_sdf` /
`enable_occupancy` / `enable_implicit`) are gone from `OctreeConfig`;
per-vertex storage now lives entirely on `FieldStorage`.
"""

import pathlib
from dataclasses import dataclass
from typing import Literal

import torch
from oren.dataset.vl_feature_dataset import DataLoader
from oren.field_storage import FieldStorage
from oren.field_storage_config import FieldStorageConfig
from oren.frame import VlFrame
from oren.octree_config import OctreeConfig
from oren.semi_sparse_octree import SemiSparseOctree
from oren.utils.config_abc import ConfigABC
from tqdm import tqdm


@dataclass
class DemoConfig(ConfigABC):
    vl_features_dir: str = None  # required
    output_path: str = None  # required
    resolution: float = 0.1
    tree_depth: int = 8
    semi_sparse_depth: int = 5
    init_voxel_num: int = 100000
    # Per-vertex parameter pre-allocation seed; matches OctreeConfig.init_vertex_num,
    # which sizes FieldStorage.values (and any FeatureBank.features) rows so the
    # first few insertions don't churn through pow-2 resize boundaries.
    init_vertex_num: int = 65536
    insertion_threshold: int = 3
    scatter_mode: Literal["overwrite", "running_average"] = "running_average"
    insertion_stride: int = 1
    device: str = "cuda"


@torch.no_grad()
def insert_phase(octree: SemiSparseOctree, dataset: DataLoader, device: str, stride: int) -> None:
    """Pass 1: build octree geometry from depth maps.

    Args:
        octree: Geometry-only octree to grow in place with new voxels.
        dataset: Source of :class:`VlFrame` per index (camera-frame points + pose pre-paired with features).
        device: Torch device for backprojected points and the octree.
        stride: Only every ``stride``-th frame is inserted (1 = all frames).
    """
    for idx in tqdm(range(0, len(dataset), stride), desc="Insert points", ncols=80):
        frame: VlFrame = dataset[idx]
        pts_world = frame.get_points(to_world_frame=True, device=device)
        if pts_world.numel() == 0:
            continue
        octree.insert_points(pts_world)


@torch.no_grad()
def scatter_phase(
    octree: SemiSparseOctree,
    vl_field: FieldStorage,
    dataset: DataLoader,
    device: str,
    mode: str,
):
    """Pass 2: scatter per-pixel VL features into the VL field's `values` tensor.

    `vl_field.values` is sized at `octree.capacity` (the pow-2-rounded vertex
    high-water mark - the FieldStorage was grown in lockstep via the resize
    observer after the insert phase). The trailing rows past
    `octree.sso.num_vertices` stay at `explicit_prior_init` (zero here).

    Args:
        octree: Geometry octree previously populated by ``insert_phase``.
        vl_field: Explicit FieldStorage whose ``values`` tensor is written to.
        dataset: Source of :class:`VlFrame` per index (world points + per-point features + pose).
        device: Torch device used for accumulation tensors.
        mode: ``"overwrite"`` to keep the last feature per slot, ``"running_average"`` to mean-pool.

    Returns:
        vl_values: (V, C) per-vertex scattered features on ``device``.
        n_touched: Count of slots that received at least one feature (``-1`` for overwrite mode).
    """
    feat_dim = vl_field.cfg.output_dim
    n_vertices = vl_field.values.shape[0]

    # We scatter into a local tensor and copy back at the end so the running-
    # average division is one shot (rather than re-reading partial sums on
    # every iteration through the dataset).
    implicit = torch.zeros((n_vertices, feat_dim), dtype=torch.float32, device=device)
    if mode == "running_average":
        count = torch.zeros((n_vertices,), dtype=torch.long, device=device)
    else:
        count = None

    for idx in tqdm(range(len(dataset)), desc="Scatter features", ncols=80):
        frame: VlFrame = dataset[idx]
        pts_world = frame.get_points(to_world_frame=True, device=device)
        if pts_world.numel() == 0:
            continue
        feat_per_pixel = frame.get_vl_features(device=device)  # (M, C) aligned with pts_world

        voxel_indices = octree.find_voxel_indices(pts_world, are_voxels=False, level=1)
        in_octree = voxel_indices >= 0
        if not in_octree.any():
            continue
        pts = pts_world[in_octree]
        f = feat_per_pixel[in_octree]
        vidx = voxel_indices[in_octree]

        # Locate nearest vertex of the leaf voxel containing each point.
        v_centers = octree.voxel_centers[vidx]  # (M, 3) meters
        v_size_m = octree.voxels[vidx, -1:].float() * octree.cfg.resolution  # (M, 1) meters
        rel = (pts - v_centers) / v_size_m  # in [-0.5, 0.5]
        bits = (rel >= 0).long()  # (M, 3)
        # Little-endian: vertex local index = bx + 2*by + 4*bz.
        vlocal = bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2]
        vert_indices_8 = octree.vertex_indices[vidx].long()  # (M, 8)
        vertex_idx = vert_indices_8.gather(1, vlocal.unsqueeze(1)).squeeze(1)  # (M,)

        has_vertex = (vertex_idx >= 0) & (vertex_idx < n_vertices)
        f = f[has_vertex]
        vertex_idx = vertex_idx[has_vertex]
        if vertex_idx.numel() == 0:
            continue

        if mode == "overwrite":
            implicit[vertex_idx] = f
        else:
            assert count is not None
            implicit.index_add_(0, vertex_idx, f)
            count.index_add_(0, vertex_idx, torch.ones_like(vertex_idx))

    if mode == "running_average":
        assert count is not None
        touched = count > 0
        implicit[touched] /= count[touched].unsqueeze(1).float()
        implicit[~touched].zero_()
        return implicit, int(touched.sum().item())
    # In overwrite mode we don't track which slots were touched; return -1.
    return implicit, -1


def main(cfg: DemoConfig) -> None:
    """Run insert + scatter passes against the dataset and save the resulting octree + VL field to disk.

    Args:
        cfg: Demo configuration with input/output paths, octree parameters, and the scatter mode.
    """
    assert cfg.vl_features_dir is not None, "DemoConfig.vl_features_dir is required"
    assert cfg.output_path is not None, "DemoConfig.output_path is required"

    dataset = DataLoader(cfg.vl_features_dir)
    print(f"Loaded {len(dataset)} frames; feature {dataset.channels}-d at {dataset.h_feat}x{dataset.w_feat}.")

    # Octree is geometry-only: OctreeConfig no longer carries enable_sdf /
    # enable_occupancy / enable_implicit / gradient_augmentation toggles -
    # all per-vertex field state lives on the FieldStorage below.
    octree_cfg = OctreeConfig(
        resolution=cfg.resolution,
        tree_depth=cfg.tree_depth,
        semi_sparse_depth=cfg.semi_sparse_depth,
        init_voxel_num=cfg.init_voxel_num,
        init_vertex_num=cfg.init_vertex_num,
        insertion_threshold=cfg.insertion_threshold,
        skip_insertion_if_exists=True,
        independent_smallest_leaf_vertex=False,
    )
    octree = SemiSparseOctree(octree_cfg).to(cfg.device)

    # One FieldStorage holding the scattered VL field. D = feature channels
    # from the dataset (typically 512 for CLIP). Explicit mode with no GA:
    # values are point-trilinear interpolations of vertex features. Phase 1
    # only exposes this mode; matches the demo's scatter-then-store flow.
    vl_field_cfg = FieldStorageConfig(
        name="vl",
        output_dim=dataset.channels,
        mode="explicit",
        gradient_augmentation=False,
        explicit_prior_init=0.0,
    )
    vl_field = FieldStorage(vl_field_cfg, octree).to(cfg.device)

    insert_phase(octree, dataset, cfg.device, cfg.insertion_stride)
    n_nodes = int(octree.sso.number_of_nodes)
    n_leaves = int(octree.sso.number_of_leaf_nodes)
    n_vertices = int(octree.sso.num_vertices)
    print(
        f"Octree built: {n_nodes} nodes ({n_leaves} leaves), {n_vertices} vertices "
        f"(buffer capacity: {octree.voxels.shape[0]}; VL field rows: {vl_field.values.shape[0]})."
    )

    vl_values, n_touched = scatter_phase(octree, vl_field, dataset, cfg.device, cfg.scatter_mode)
    if cfg.scatter_mode == "running_average":
        print(f"Vertices with VL features: {n_touched} / {vl_values.shape[0]}")

    # Copy the scattered values into the field's parameter so consumers reading
    # the saved state see the same FieldStorage state that lives in memory.
    with torch.no_grad():
        vl_field.values.data.copy_(vl_values)

    output_path = pathlib.Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The semi-sparse octree stores ALL nodes (root + internals + leaves) in
    # the per-node buffers; voxel_size differs per level. Slice by n_nodes
    # (total populated rows) to drop unused buffer tail. The visualizer
    # filters to leaves via `(structure == -1).all(dim=1)` before rendering.
    # Key rename: `implicit_features` -> `vl_values` (the field's `values`
    # tensor), reflecting the architectural shift to FieldStorage.
    state = {
        "voxels": octree.voxels[:n_nodes].cpu(),
        "voxel_centers": octree.voxel_centers[:n_nodes].cpu(),
        "vertex_indices": octree.vertex_indices[:n_nodes].cpu(),
        "structure": octree.structure[:n_nodes].cpu(),
        "vl_values": vl_values.cpu(),
        # Back-compat alias so existing visualizer scripts that read
        # `state["implicit_features"]` keep working until they're migrated.
        "implicit_features": vl_values.cpu(),
        "n_nodes": n_nodes,
        "n_leaves": n_leaves,
        "n_vertices": n_vertices,
        "octree_cfg": octree_cfg.as_dict(),
        "vl_field_cfg": vl_field_cfg.as_dict(),
        "demo_cfg": cfg.as_dict(),
        "manifest": dataset.manifest,
        "scatter_mode": cfg.scatter_mode,
    }
    torch.save(state, output_path)
    print(f"Saved octree to {output_path}")


if __name__ == "__main__":
    parser = DemoConfig.get_argparser()
    cfg, _ = parser.parse_known_args()
    main(cfg)
