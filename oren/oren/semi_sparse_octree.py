"""Geometry-only semi-sparse octree.

The underlying C++ tree is provided by `erl_geometry::SemiSparseOctreeF`.

The octree owns geometry buffers (voxels, voxel_centers, vertex_indices,
structure) and exposes the per-call `query(points) -> OctreeGeometry` cache
that FieldStorage consumes.

Per-vertex field state, such as SDF/OCC priors, gradient priors, implicit
features now lives on `FieldStorage` / `FeatureBank` modules.

Buffers (registered as torch buffers, not parameters):
  - voxels:         (N, 4) [x, y, z, voxel_size]
  - voxel_centers:  (N, 3) in meters
  - vertex_indices: (N, 8) per-leaf vertex ids; -1 if absent
  - structure:      (N, 8) per-node children indices

Resize observers (FieldStorage / FeatureBank) register with this class and
get notified after every `insert_points` if the C++ tree's vertex count
grows past the last seen value.
"""

from typing import Callable
from erl_geometry import SemiSparseOctreeF, find_voxel_indices, morton_encode

from oren import torch
from oren.field_output import LevelGeometry, OctreeGeometry
from oren.ga_trilinear import normalize_to_voxel_unit_cube
from oren.octree_config import OctreeConfig


def _round_up_pow2(n: int) -> int:
    """Smallest power of two >= n. `_round_up_pow2(0) == 1`.

    Used to round per-vertex parameter capacity up before announcing a resize to observers, so allocations align with
    GPU page sizes and reduce VRAM fragmentation from many narrow `torch.zeros` calls during early training.
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class SemiSparseOctree(torch.nn.Module):

    def __init__(self, cfg: OctreeConfig):
        """Build the C++-backed octree and register Python-side geometry buffers.

        Args:
            cfg: octree configuration (resolution, depth, initial capacities, etc.).
        """
        super().__init__()
        self.cfg = cfg

        self.ever_inserted = False

        n = self.cfg.init_voxel_num
        self.register_buffer("voxels", torch.zeros((n, 4), dtype=torch.float32))
        self.register_buffer("voxel_centers", torch.zeros((n, 3), dtype=torch.float32))
        self.register_buffer("vertex_indices", torch.zeros((n, 8), dtype=torch.int32))
        self.register_buffer("structure", torch.zeros((n, 8), dtype=torch.int32))

        # (N, 4) [x, y, z, voxel_size].
        self.voxels: torch.Tensor
        # (N, 3) in meter.
        self.voxel_centers: torch.Tensor
        # (N, 8) index of vertices, -1 if not exists.
        self.vertex_indices: torch.Tensor
        # (N, 8) [children(8)].
        self.structure: torch.Tensor

        # erl_geometry backing tree.
        sso_setting = SemiSparseOctreeF.Setting()
        sso_setting.resolution = cfg.resolution
        sso_setting.tree_depth = cfg.tree_depth
        sso_setting.semi_sparse_depth = cfg.semi_sparse_depth
        sso_setting.init_voxel_num = cfg.init_voxel_num
        sso_setting.independent_smallest_leaf_vertex = cfg.independent_smallest_leaf_vertex
        sso_setting.cache_voxel_centers = True
        self.sso = SemiSparseOctreeF(sso_setting)
        self.key_offset = 1 << (self.cfg.tree_depth - 1)

        # Resize-observer machinery.
        # FeatureBank, FieldStorage, and any optimizer-state migrator register here; callbacks fire after
        # `insert_points` when the C++ tree's `num_vertices` (the unique-vertex-ID high-water mark) has grown past the
        # last announced capacity. The announced capacity is rounded up to the next power of two for VRAM efficiency,
        # so observers only fire on pow-2 boundary crossings.
        # Python-side polling — no C++ callback re-entry on the insertion fast path.
        self._resize_observers: list[Callable[[int], None]] = []
        # Seed with the pow-2 rounding of `max(num_vertices, init_vertex_num)`.
        # `init_vertex_num` pre-allocates per-vertex parameter capacity so the early-training insertions don't churn
        # through pow-2 boundaries; the `max()` ensures a checkpoint-loaded tree with more vertices than the config's
        # seed wins.
        self._last_known_capacity: int = _round_up_pow2(max(int(self.sso.num_vertices), int(self.cfg.init_vertex_num)))

    # --- state_dict ---
    # The four geometry buffers (voxels / voxel_centers / vertex_indices / structure) are already in state_dict via
    # `register_buffer`. They are derived from the C++ tree, so on their own they cannot rebuild the SSO's continuous
    # node buffers, vertex maps, recycled slots, or setting. We bolt three extra keys onto state_dict so a checkpoint
    # is fully self-contained:
    #   - `sso_blob`             : uint8 1-D tensor; byte-equivalent to `Serialization<SemiSparseOctreeF>::Write`.
    #   - `ever_inserted`        : bool scalar; gates the `skip_insertion_if_exists` fast path.
    #   - `last_known_capacity`  : int64 scalar; the pow-2 capacity already announced to resize observers.
    # The four geometry buffers are sliced to `sso.buf_head` on save — entries beyond the live high-water mark are
    # uninitialized noise (the C++ side reserves the last column of `m_voxels_` as a sentinel) and would otherwise
    # bloat the checkpoint by roughly `init_voxel_num / buf_head` (often >100x). The load path already pre-resizes
    # the local buffers to match the saved shape, so smaller saved shapes load fine. After load, the buffers are
    # exactly `buf_head`-sized; the next `insert_voxels` call rebinds them to `sso.*_tensor`, which is the full
    # C++ `buf_size`, so the shrunken state is transient.

    _SLICED_BUFFERS = ("voxels", "voxel_centers", "vertex_indices", "structure")

    def _save_to_state_dict(self, destination, prefix: str, keep_vars: bool) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        buf_head = int(self.sso.buf_head)
        for name in self._SLICED_BUFFERS:
            key = prefix + name
            if key in destination:
                destination[key] = destination[key][:buf_head].contiguous().clone()
        # `sso.write()` returns numpy uint8; zero-length on failure (the C++ side emits its own ERL_WARN).
        blob = self.sso.write()
        destination[prefix + "sso_blob"] = torch.from_numpy(blob).clone()
        destination[prefix + "ever_inserted"] = torch.tensor(self.ever_inserted, dtype=torch.bool)
        destination[prefix + "last_known_capacity"] = torch.tensor(self._last_known_capacity, dtype=torch.int64)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix: str,
        local_metadata,
        strict: bool,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # 1. Restore the C++ tree FIRST. Any observer that wakes up later (e.g. from a sibling module loaded after us)
        # must see the correct `num_vertices`.
        sso_key = prefix + "sso_blob"
        if sso_key in state_dict:
            blob = state_dict.pop(sso_key)
            np_blob = torch.as_tensor(blob, dtype=torch.uint8).detach().cpu().contiguous().numpy()
            if np_blob.size == 0 or not self.sso.read(np_blob):
                error_msgs.append(f"{sso_key}: SemiSparseOctreeF.read() failed; the C++ tree was not restored.")
        elif strict:
            missing_keys.append(sso_key)

        # 2. Python-side scalars.
        inserted_key = prefix + "ever_inserted"
        if inserted_key in state_dict:
            v = state_dict.pop(inserted_key)
            self.ever_inserted = bool(v.item() if isinstance(v, torch.Tensor) else v)
        elif strict:
            missing_keys.append(inserted_key)

        capacity_key = prefix + "last_known_capacity"
        if capacity_key in state_dict:
            v = state_dict.pop(capacity_key)
            self._last_known_capacity = int(v.item() if isinstance(v, torch.Tensor) else v)
        elif strict:
            missing_keys.append(capacity_key)

        # 3. The four geometry buffers can have grown between init and checkpoint time. Super's default loader does an
        # in-place `copy_` that requires shape match, so we rebind any size-mismatched buffer to an empty tensor of the
        # saved shape (and the local dtype/device) before delegating.
        for name in ("voxels", "voxel_centers", "vertex_indices", "structure"):
            key = prefix + name
            if key not in state_dict:
                continue
            saved = state_dict[key]
            local = getattr(self, name)
            if tuple(saved.shape) != tuple(local.shape):
                setattr(self, name, torch.empty(saved.shape, dtype=local.dtype, device=local.device))

        # 4. Fire registered resize observers so per-vertex tensors owned by sibling modules (FieldStorage.values,
        # FeatureBank.features, fuser counts / weight_sum, etc.) grow to the loaded capacity *before* their own
        # `_load_from_state_dict` runs the in-place `copy_`. Without this, freshly-constructed buffers stay at
        # `init_vertex_num` and the subsequent load throws a shape-mismatch error. Each observer's grow path is a
        # no-op when the buffer is already large enough, so re-firing here is safe.
        for fn in self._resize_observers:
            fn(self._last_known_capacity)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @property
    def capacity(self) -> int:
        """The current per-vertex-parameter capacity announced to observers.

        Always a power of two >= `max(num_vertices, cfg.init_vertex_num)`. Use this — not `cfg.init_voxel_num` — when
        sizing per-vertex parameters on the Python side: `init_voxel_num` is the C++ voxel-node buffer's initial
        capacity, which counts *nodes*, not unique vertices. `cfg.init_vertex_num` is the per-vertex-parameter
        pre-allocation seed.
        """
        return self._last_known_capacity

    @torch.no_grad()
    def points_to_voxels(self, points: torch.Tensor) -> torch.Tensor:
        """Convert (..., 3) world points -> (..., 3) voxel coordinates.

        Args:
            points: (..., 3) point cloud in world coordinates.

        Returns:
            (..., 3) integer voxel coordinates shifted by `self.key_offset`.
        """
        voxels = torch.div(points, self.cfg.resolution, rounding_mode="floor").long()
        voxels += self.key_offset
        return voxels

    @torch.no_grad()
    def insert_voxels(self, voxels: torch.Tensor) -> torch.Tensor:
        """Insert voxels into the octree, update buffers, return per-voxel indices.

        Args:
            voxels: (n_voxels, 3) integer voxel coordinates to insert.

        Returns:
            (n_voxels,) per-voxel indices into the octree's voxel buffer, on CPU.
        """
        self.ever_inserted = True
        svo_idx = self.sso.insert_keys(voxels.cpu().to(torch.uint32))  # on CPU

        device = self.voxels.device
        self.voxels = self.sso.voxels_tensor.long().to(device)
        self.voxel_centers = self.sso.voxel_centers_tensor.to(device)
        self.vertex_indices = self.sso.vertices_tensor.to(device)
        self.structure = self.sso.children_tensor.to(device)
        return svo_idx

    @torch.no_grad()
    def insert_points(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Insert points into the octree.
        Args:
            points: (n_points, 3) point cloud in world coordinates
        Returns:
            voxels_unique: (n_unique, 3) unique voxel coordinates inserted
            voxel_indices: (n_unique,) per-voxel indices, on CPU.
        """
        voxels = self.points_to_voxels(points)
        voxels_raw, counts = torch.unique(voxels, dim=0, return_inverse=False, return_counts=True)
        voxels_valid = voxels_raw[counts > self.cfg.insertion_threshold]
        voxels_unique = torch.unique(voxels_valid, dim=0)
        if voxels_unique.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device="cpu"), torch.empty((0,), dtype=torch.long, device="cpu")
        if self.cfg.skip_insertion_if_exists and self.ever_inserted:
            device = self.voxels.device
            voxel_indices = self.find_voxel_indices(voxels_unique.to(device), True, level=1)
            voxel_sizes = self.get_voxel_discrete_size(voxel_indices)
            # only insert the voxels that are not present or have discrete size > 1 (not the finest level)
            mask = voxel_sizes != 1
            voxels_to_insert = voxels_unique[mask]
            if voxels_to_insert.numel() == 0:
                return voxels_unique, voxel_indices.cpu()
            voxel_indices[mask] = self.insert_voxels(voxels_to_insert).to(device)
        else:
            voxel_indices = self.insert_voxels(voxels_unique)
        self._notify_resize_if_grown()
        return voxels_unique, voxel_indices.cpu()

    def register_resize_observer(self, fn: Callable[[int], None]) -> None:
        """Register a callback fired when the C++ tree's vertex count grows.

        `fn(new_capacity: int)` is invoked from `insert_points` after the C++ insertion returns; observers see only
        monotonic growth. Used by FeatureBank / FieldStorage to keep their (V, *) parameters in lockstep with
        `self.sso.num_vertices`.

        Catch-up: the callback is invoked once at registration with the current `_last_known_capacity`. Observers must
        treat a fire with `new == self.capacity` as a no-op (FeatureBank.grow_to and FieldStorage._on_octree_resize
        already do, via `if new <= old: return`). This makes it safe to construct FieldStorage / FeatureBank instances
        AFTER the octree has already grown past its initial capacity.

        Args:
            fn: callback taking the new pow-2-rounded vertex capacity.
        """
        self._resize_observers.append(fn)
        fn(self._last_known_capacity)

    def unregister_resize_observer(self, fn: Callable[[int], None]) -> None:
        """Remove a previously registered resize observer; no-op if `fn` is not registered.

        Args:
            fn: callback previously passed to `register_resize_observer`.
        """
        try:
            self._resize_observers.remove(fn)
        except ValueError:
            pass

    def _notify_resize_if_grown(self) -> None:
        """Poll the C++ tree's vertex count and fire observers if it grew past the last announced (pow-2) capacity.
        Observers receive the new rounded capacity, not the raw `num_vertices`, so they resize per-vertex tensors to a
        power-of-two size."""
        current_vertices = int(self.sso.num_vertices)
        if current_vertices <= self._last_known_capacity:
            return
        rounded = _round_up_pow2(current_vertices)
        for fn in list(self._resize_observers):
            fn(rounded)
        self._last_known_capacity = rounded

    @torch.no_grad()
    def get_voxel_discrete_size(self, voxel_indices: torch.Tensor) -> torch.Tensor:
        """Return (..., ) discrete sizes for the given voxel indices.

        Args:
            voxel_indices: (...) long tensor of voxel indices; -1 entries are tolerated and return -1.

        Returns:
            (...) discrete voxel sizes (in voxel-grid units); -1 where `voxel_indices == -1`.
        """
        assert self.voxels is not None, "Octree is empty. Please insert points first."
        assert voxel_indices.dtype == torch.long, "voxel_indices must be of type torch.long"

        voxel_sizes = self.voxels[voxel_indices.view(-1), -1]
        voxel_sizes = voxel_sizes.view(voxel_indices.shape)
        voxel_sizes[voxel_indices < 0] = -1
        return voxel_sizes

    @torch.no_grad()
    def find_voxel_indices(self, points: torch.Tensor, are_voxels: bool, level: int = 1) -> torch.Tensor:
        """Find per-point voxel indices at the requested level. -1 -> not present.

        Args:
            points: (..., 3) world points or integer voxel coordinates (see `are_voxels`).
            are_voxels: if True, treat `points` as already-quantized voxel coordinates; otherwise quantize first.
            level: octree level to query; 1 is the leaf level, higher walks toward the root.

        Returns:
            (...,) long tensor of voxel indices; -1 marks out-of-bounds or missing voxels.
        """
        if are_voxels:
            voxels = points
        else:
            voxels = self.points_to_voxels(points)
        morton_codes = morton_encode(voxels.to(torch.uint32))
        voxel_indices = find_voxel_indices(
            codes=morton_codes,
            dims=3,
            n_levels=self.cfg.tree_depth - level,
            children=self.structure,
        ).long()
        mask = ((voxels < 0) | (voxels >= (1 << self.cfg.tree_depth))).any(dim=-1)
        voxel_indices[mask] = -1  # Out of bounds
        return voxel_indices

    def _compute_level_geometry(self, points: torch.Tensor, level: int = 1) -> LevelGeometry:
        """Build a single-level geometry snapshot for `points` at the given `level`.

        Level 1 is the leaf level; higher levels walk up the octree. Used by the public `query()` accessor and the
        per-level cache on `OctreeGeometry`. Pure geometry — no field math.
        """
        voxel_indices = self.find_voxel_indices(points, False, level)
        voxel_centers = self.voxel_centers[voxel_indices]  # (n_points, 3)
        voxel_sizes = self.voxels[voxel_indices, -1:]  # (n_points, 1)
        vertex_indices = self.vertex_indices[voxel_indices]  # (n_points, 8)
        voxel_offsets = normalize_to_voxel_unit_cube(points, voxel_centers, voxel_sizes, self.cfg.resolution)
        return LevelGeometry(
            level=level,
            points=points,
            voxel_indices=voxel_indices,
            voxel_centers=voxel_centers,
            voxel_sizes=voxel_sizes,
            vertex_indices=vertex_indices,
            voxel_offsets=voxel_offsets,  # (n_points, 3) in [0, 1]^3 within the voxel
        )

    def query(self, points: torch.Tensor, voxel_indices: torch.Tensor = None) -> OctreeGeometry:
        """Per-call multi-level geometry cache.

        Level 1 is populated eagerly (every field needs it); higher levels are filled lazily via
        `OctreeGeometry.at_level(k)` and cached for the rest of this forward pass. The caller must guarantee the
        correctness of `voxel_indices` if provided; no safety check.

        Args:
            points: (n_points, 3) world-coordinate query points.
            voxel_indices: optional (n_points,) long tensor of leaf-level voxel indices; if provided, skips the
                level-1 lookup and is used as-is.

        Returns:
            OctreeGeometry with level 1 pre-populated and higher levels lazily computed on demand.
        """
        geom = OctreeGeometry(points=points, compute_level=self._compute_level_geometry)
        if voxel_indices is not None:
            assert voxel_indices.dtype == torch.long
            voxel_centers = self.voxel_centers[voxel_indices]
            voxel_sizes = self.voxels[voxel_indices, -1:]
            vertex_indices = self.vertex_indices[voxel_indices]
            voxel_offsets = normalize_to_voxel_unit_cube(points, voxel_centers, voxel_sizes, self.cfg.resolution)
            geom._levels[1] = LevelGeometry(
                level=1,
                points=points,
                voxel_indices=voxel_indices,
                voxel_centers=voxel_centers,
                voxel_sizes=voxel_sizes,
                vertex_indices=vertex_indices,
                voxel_offsets=voxel_offsets,
            )
        else:
            geom._levels[1] = self._compute_level_geometry(points, level=1)
        return geom

    @torch.no_grad()
    def grid_vertex_filter(
        self,
        grid_points: torch.Tensor,
        min_voxel_size: int = 1,
        max_voxel_size: int = 2,
        dilation_iters: int = 1,
        batch_size: int = 204800,
        device: str | None = None,
    ) -> torch.Tensor:
        """
        Filter out grid vertices that are in voxels that are too big.
        Args:
            grid_points: (nx, ny, nz, 3) grid points in world coordinates
            min_voxel_size: minimum voxel size to keep
            max_voxel_size: maximum voxel size to keep
            dilation_iters: number of dilation iterations to fill small holes
            batch_size: number of points to process in a batch
            device: device to use, if None, use the device of grid_points

        Returns:
            (nx, ny, nz) boolean mask, True if the vertex is valid (in a voxel that is not too big)
        """
        assert grid_points.ndim == 4 and grid_points.shape[-1] == 3

        if batch_size <= 0:
            bs = grid_points.shape[0] * grid_points.shape[1] * grid_points.shape[2]
        else:
            bs = batch_size

        grid_shape = grid_points.shape
        grid_points = grid_points.view(-1, 3)

        model_device = self.structure.device
        if device is None:
            device = grid_points.device

        valid_mask = []
        for start in range(0, grid_points.shape[0], bs):
            end = min(start + bs, grid_points.shape[0])
            indices = self.find_voxel_indices(grid_points[start:end].to(model_device), False).view(-1)
            sizes = self.get_voxel_discrete_size(indices)
            valid_mask.append(((sizes >= min_voxel_size) & (sizes <= max_voxel_size)).to(device))

        if len(valid_mask) == 1:
            valid_mask = valid_mask[0]
        else:
            valid_mask = torch.cat(valid_mask, dim=0)
        valid_mask = valid_mask.view(grid_shape[:-1])

        # Dilation: any vertex valid -> keep its 8-neighborhood, so the cube counts.
        kernel = torch.ones((3, 3, 3), dtype=torch.float32, device=valid_mask.device).view(1, 1, 3, 3, 3)
        for _ in range(dilation_iters):
            valid_mask = (
                torch.nn.functional.conv3d(
                    input=valid_mask.view(1, 1, *valid_mask.shape).to(torch.float32),
                    weight=kernel,
                    padding=1,
                ).view(*valid_mask.shape)
                >= 1
            ).to(torch.bool)

        return valid_mask

    @property
    def little_endian_vertex_order(self) -> bool:
        """True iff the C++ implementation uses little-endian vertex ordering (vertex local id 1 -> (1, 0, 0))."""
        return True
