from dataclasses import dataclass

from oren.utils.config_abc import ConfigABC


@dataclass
class OctreeConfig(ConfigABC):
    """Geometry-only octree configuration."""

    resolution: float = 0.1
    tree_depth: int = 8
    semi_sparse_depth: int = 5
    init_voxel_num: int = 200000
    # Initial per-vertex-parameter capacity seed (the V dim of every FieldStorage / FeatureBank tensor). The octree
    # announces `max(num_vertices, init_vertex_num)` rounded up to the next power of two as `octree.capacity` on
    # construction — so per-vertex tensors are pre-allocated at this size and the first insertions don't trigger
    # resizes until the vertex high-water mark crosses the next pow-2 boundary above this value. Unrelated to
    # `init_voxel_num`, which is the C++ voxel-node buffer's initial capacity (nodes, not vertices).
    init_vertex_num: int = 65536
    insertion_threshold: int = 3  # Minimum number of points to insert a voxel
    # If True, skip insertion if voxel of size 1 already exists. Speeds up insertion when the underlying search is
    # fast enough.
    skip_insertion_if_exists: bool = True
    independent_smallest_leaf_vertex: bool = False
