# pyright: reportPrivateImportUsage=none
"""Build a SemiSparseOctree whose implicit features are scattered VL features.

This demo:
  1. Loads a VL feature bundle (written by `generate_vl_features`).
  2. Backprojects each frame's depth into world points.
  3. Inserts those points into a SemiSparseOctree.
  4. Scatters per-pixel VL features into the nearest vertex of the
     containing leaf voxel, either by overwrite or running average.
  5. Saves the populated octree state to disk for visualization.

SDF priors and gradient priors are left at their zero initialization —
this demo is purely about using the octree as a sparse VL feature store.
"""

import pathlib
from dataclasses import dataclass
from typing import Literal

import torch
from tqdm import tqdm

from oren.octree_config import OctreeConfig
from oren.semi_sparse_octree_v2 import SemiSparseOctree
from oren.utils.config_abc import ConfigABC
from oren_vl.dataset.vl_features_dataset import VLFeaturesDataset, depth_to_world_points


@dataclass
class DemoConfig(ConfigABC):
    vl_features_dir: str = None  # required
    output_path: str = None  # required
    resolution: float = 0.1
    tree_depth: int = 8
    semi_sparse_depth: int = 5
    init_voxel_num: int = 100000
    insertion_threshold: int = 3
    scatter_mode: Literal["overwrite", "running_average"] = "running_average"
    insertion_stride: int = 1
    device: str = "cuda"


@torch.no_grad()
def insert_phase(
    octree: SemiSparseOctree, dataset: VLFeaturesDataset, K_feat: torch.Tensor, device: str, stride: int
) -> None:
    """Pass 1: build octree geometry from depth maps."""
    for idx in tqdm(range(0, len(dataset), stride), desc="Insert points", ncols=80):
        _, _, depth, pose = dataset[idx]
        pts_world, _ = depth_to_world_points(depth.to(device), pose.to(device), K_feat)
        if pts_world.numel() == 0:
            continue
        octree.insert_points(pts_world)


@torch.no_grad()
def scatter_phase(octree: SemiSparseOctree, dataset: VLFeaturesDataset, K_feat: torch.Tensor, device: str, mode: str):
    """Pass 2: scatter per-pixel VL features into vertex implicit features.

    Returns (implicit_features, n_touched). implicit_features is sized to
    octree.sso.num_vertices (which can exceed init_voxel_num — the underlying
    erl_geometry tree grows past its initial buffer, and using
    octree.implicit_features directly would index out of bounds).
    n_touched counts vertex slots that received at least one feature
    (only meaningful for `running_average`).
    """
    n_vertices = int(octree.sso.num_vertices)
    feat_dim = octree.implicit_features.shape[1]

    implicit = torch.zeros((n_vertices, feat_dim), dtype=torch.float32, device=device)
    if mode == "running_average":
        count = torch.zeros((n_vertices,), dtype=torch.long, device=device)
    else:
        count = None

    for idx in tqdm(range(len(dataset)), desc="Scatter features", ncols=80):
        _, feat, depth, pose = dataset[idx]
        feat = feat.to(device).float()  # (C, h, w)
        depth = depth.to(device)
        pose = pose.to(device)

        pts_world, valid_mask = depth_to_world_points(depth, pose, K_feat)
        if pts_world.numel() == 0:
            continue

        feat_per_pixel = feat.permute(1, 2, 0).reshape(-1, feat_dim)  # (h*w, C)
        feat_per_pixel = feat_per_pixel[valid_mask.flatten()]  # (M, C)

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
    assert cfg.vl_features_dir is not None, "DemoConfig.vl_features_dir is required"
    assert cfg.output_path is not None, "DemoConfig.output_path is required"

    dataset = VLFeaturesDataset(cfg.vl_features_dir)
    print(f"Loaded {len(dataset)} frames; feature {dataset.channels}-d at " f"{dataset.h_feat}x{dataset.w_feat}.")

    K_feat = torch.from_numpy(dataset.K_feat).to(cfg.device)

    octree_cfg = OctreeConfig(
        resolution=cfg.resolution,
        tree_depth=cfg.tree_depth,
        semi_sparse_depth=cfg.semi_sparse_depth,
        init_voxel_num=cfg.init_voxel_num,
        insertion_threshold=cfg.insertion_threshold,
        skip_insertion_if_exists=True,
        gradient_augmentation=True,
        implicit_feature_dim=dataset.channels,
        implicit_num_levels=1,
        independent_smallest_leaf_vertex=False,
    )
    octree = SemiSparseOctree(octree_cfg).to(cfg.device)

    insert_phase(octree, dataset, K_feat, cfg.device, cfg.insertion_stride)
    n_nodes = int(octree.sso.number_of_nodes)
    n_leaves = int(octree.sso.number_of_leaf_nodes)
    n_vertices = int(octree.sso.num_vertices)
    print(
        f"Octree built: {n_nodes} nodes ({n_leaves} leaves), {n_vertices} vertices "
        f"(buffer capacity: {octree.voxels.shape[0]})."
    )

    implicit_features, n_touched = scatter_phase(octree, dataset, K_feat, cfg.device, cfg.scatter_mode)
    if cfg.scatter_mode == "running_average":
        print(f"Vertices with VL features: {n_touched} / {implicit_features.shape[0]}")

    output_path = pathlib.Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The semi-sparse octree stores ALL nodes (root + internals + leaves) in
    # the per-node buffers; voxel_size differs per level. Slice by n_nodes
    # (total populated rows) to drop unused buffer tail. The visualizer
    # filters to leaves via `(structure == -1).all(dim=1)` before rendering.
    state = {
        "voxels": octree.voxels[:n_nodes].cpu(),
        "voxel_centers": octree.voxel_centers[:n_nodes].cpu(),
        "vertex_indices": octree.vertex_indices[:n_nodes].cpu(),
        "structure": octree.structure[:n_nodes].cpu(),
        "implicit_features": implicit_features.cpu(),
        "n_nodes": n_nodes,
        "n_leaves": n_leaves,
        "n_vertices": n_vertices,
        "octree_cfg": octree_cfg.as_dict(),
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
