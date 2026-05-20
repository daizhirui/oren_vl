"""Fusion policies for writing point-wise features into per-vertex parameters and reading them back.

Each `Fuser` is an `nn.Module` bound to a `SemiSparseOctree` at construction.
The *target* `nn.Parameter` (`field.values`, `field.bank.features`, or any other `(capacity, D)` tensor sized to
the same octree) is passed in at every call. This lets the same fuser instance operate on different per-vertex
parameters of the same octree without re-binding.

Subclasses provided here (Phase 1, no learnable parameters of their own):

    - `NearestVertexFuser` (modes: `overwrite` / `running_average` / `ema`). No learnable mode.
    - `TrilinearFuser`. Optional Hermite gradient-augmented gather via the `grads` argument.
    - `InverseDistanceFuser`, `RbfFuser`, `OrnsteinUhlenbeckFuser`. Same 8-corner splat as `TrilinearFuser` but
        with kernel weights instead of the trilinear basis.

Stateful fusers (`NearestVertexFuser.running_average` and the four 8-corner kernel fusers) own per-vertex auxiliary
buffers (`counts` or `weight_sum`) and register a resize observer on the bound octree so those buffers grow in
lockstep with `octree.sso.num_vertices` -- mirroring how `FieldStorage._on_octree_resize` and
`FeatureBank.grow_to` handle their own per-vertex tensors. The catch-up fire at registration brings new fusers up to
the current octree capacity, so it is safe to construct a fuser after the octree has already grown.

`scatter` is wrapped with `@torch.no_grad()` and writes into the target tensor's `.data` in place. `gather` is
differentiable in the target tensor and is the only path used in the learnable mode (the optimizer updates the
target end-to-end against a reconstruction loss between gathered and input features).

Vertex local IDs follow the bound octree's `little_endian_vertex_order`. Each `_CornerWeightFuser` caches its
own corner-offset tables as non-persistent buffers (`_offsets11` in `{-1, +1}^3`, `_offsets01_flip` = `1 - offsets01`
in `{0, 1}^3`) so corner positions and trilinear weights can be computed without any global state.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from oren.field_output import LevelGeometry
from oren.semi_sparse_octree import SemiSparseOctree
from oren.utils.config_abc import ConfigABC
from oren.utils.param_resize import grow_param_first_dim

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FuserConfig(ConfigABC):
    """Base class for fuser configs. Subclasses select the mechanism via `cfg_identifier`.

    YAML round-tripping goes through `ConfigABC._registry`: a serialised `FuserConfig` field stores the chosen
    subclass's identifier and `from_dict` re-dispatches to that subclass on load.
    """

    pass


@dataclass
class NearestVertexFuserConfig(FuserConfig):
    """Config for :class:`NearestVertexFuser`.

    `mode` selects the update rule applied at the nearest vertex of each point's leaf voxel:

        - `overwrite`: each point's feature replaces the vertex's stored value (last write wins; stateless).
        - `running_average`: maintain `values[v]` as the streaming mean of all features assigned to `v` across
            every `scatter` call. A `counts` buffer tracks the per-vertex sample count.
        - `ema`: `values[v] <- (1 - ema_alpha) * values[v] + ema_alpha * f`. Stateless (the running update folds
            into `values` directly).
    """

    mode: Literal["overwrite", "running_average", "ema"] = "running_average"
    ema_alpha: float = 0.1


@dataclass
class TrilinearFuserConfig(FuserConfig):
    """Config for :class:`TrilinearFuser`.

    `TrilinearFuser` doubles as the canonical trilinear-gather implementation: `FieldStorage._prior` delegates to
    `TrilinearFuser.gather` with the field's `values` (and `grads` when `cfg.gradient_augmentation`), so the
    default `FieldStorageConfig.prior_fuser_cfg` is a plain `TrilinearFuserConfig` and existing SDF / OCC fields
    keep their behavior with no config change.
    """

    pass


@dataclass
class InverseDistanceFuserConfig(FuserConfig):
    """Config for :class:`InverseDistanceFuser`.

    Kernel weights are `w_i = 1 / (||p - c_i|| + epsilon)` in metric units. `epsilon` guards against
    division-by-zero when a query point coincides exactly with a corner; it has the same length units as
    `octree.cfg.resolution`.
    """

    epsilon: float = 1e-6


@dataclass
class RbfFuserConfig(FuserConfig):
    """Config for :class:`RbfFuser`.

    Kernel weights are `w_i = exp(-||p - c_i||^2 / (2 * bandwidth^2))` in metric units. `bandwidth` is a length
    scale in the same units as `octree.cfg.resolution`.
    """

    bandwidth: float = 0.1


@dataclass
class OrnsteinUhlenbeckFuserConfig(FuserConfig):
    """Config for :class:`OrnsteinUhlenbeckFuser`.

    Kernel weights are `w_i = exp(-||p - c_i|| / bandwidth)` (Matern-1/2, linear distance in the exponent). The
    kernel is C^0 but not C^1 at voxel vertices, so the *learnable* gather path is discouraged here -- prefer
    :class:`TrilinearFuser` or :class:`RbfFuser` for reconstruction training. The direct-update scatter path is
    unaffected.
    """

    bandwidth: float = 0.1


def build_fuser(cfg: FuserConfig, octree: SemiSparseOctree) -> "Fuser":
    """Construct a concrete :class:`Fuser` from its config + bound octree.

    The dispatch lives outside the config classes so the config module stays a pure data carrier (no `nn.Module`
    imports there).
    """
    if isinstance(cfg, NearestVertexFuserConfig):
        return NearestVertexFuser(octree=octree, mode=cfg.mode, ema_alpha=cfg.ema_alpha)
    if isinstance(cfg, TrilinearFuserConfig):
        return TrilinearFuser(octree=octree)
    if isinstance(cfg, InverseDistanceFuserConfig):
        return InverseDistanceFuser(octree=octree, epsilon=cfg.epsilon)
    if isinstance(cfg, RbfFuserConfig):
        return RbfFuser(octree=octree, bandwidth=cfg.bandwidth)
    if isinstance(cfg, OrnsteinUhlenbeckFuserConfig):
        return OrnsteinUhlenbeckFuser(octree=octree, bandwidth=cfg.bandwidth)
    raise TypeError(f"Unknown FuserConfig subtype: {type(cfg).__name__}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vertex_offsets(little_endian: bool, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build an `(8, 3)` tensor of voxel-corner offsets in `{-1, +1}^3` ordered by vertex local ID.

    With `little_endian=True` the LSB encodes x: vertex 1 -> `(+1, -1, -1)`, vertex 2 -> `(-1, +1, -1)`,
    vertex 4 -> `(-1, -1, +1)`. This matches the convention the C++ octree exposes through
    `vertex_indices[..., i]`.

    Equivalent code for big-endian:
    ```python
    cut = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    xx, yy, zz = torch.meshgrid(cut, cut, cut, indexing="ij")  # big-endian
    offsets = torch.stack([xx, yy, zz], dim=-1).reshape(1, 8, 3)  # (1,8,3)
    ```
    """

    ids = torch.arange(8, device=device, dtype=torch.long)
    if little_endian:
        # Bit 0 (LSB) encodes x, bit 1 encodes y, bit 2 encodes z. Vertex local ID = 1*bit0 + 2*bit1 + 4*bit2.
        bx = (ids & 1).to(dtype)
        by = ((ids >> 1) & 1).to(dtype)
        bz = ((ids >> 2) & 1).to(dtype)
    else:
        # Bit 0 (LSB) encodes z, bit 1 encodes y, bit 2 encodes x. Vertex local ID = 1*bit0 + 2*bit1 + 4*bit2.
        bz = (ids & 1).to(dtype)
        by = ((ids >> 1) & 1).to(dtype)
        bx = ((ids >> 2) & 1).to(dtype)
    offsets01 = torch.stack([bx, by, bz], dim=-1)  # (8, 3) in {0, 1}^3
    return 2.0 * offsets01 - 1.0  # (8, 3) in {-1, +1}^3


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Fuser(nn.Module):
    """Policy for writing point-wise features into a per-vertex parameter and reading them back.

    Subclasses implement one fusion mechanism each. `scatter` and `gather` must use the same mechanism so a
    feature written in is recovered (within numerical error) by gathering at the same point.

    A Fuser is bound to one :class:`SemiSparseOctree` at construction so its state buffers can register a resize
    observer and grow in lockstep with `octree.sso.num_vertices`. The octree itself is not stored on the fuser --
    only the derived knobs (endianness, resolution) are cached, and the resize-observer callback the octree retains
    is what keeps the fuser alive on that side.

    The *target* tensor -- whatever `(capacity, D)` `nn.Parameter` to read/write -- is passed in at each call, not
    configured up front. This lets the same fuser instance operate on `FieldStorage.values`,
    `FeatureBank.features`, or any other compatible per-vertex parameter on the same octree. Query geometry comes
    in via a :class:`LevelGeometry` (the per-level snapshot returned by `octree.query(points).at_level(level)`),
    which carries both the world-space `points` and the per-level voxel/vertex indices; callers are responsible
    for building the `OctreeGeometry` cache (typically once per forward pass and shared across fields).
    """

    def __init__(self, octree: SemiSparseOctree):
        super().__init__()
        # Cache the derived knobs the subclasses need at call time. We deliberately do NOT keep a back-reference to
        # `octree`: the resize-observer callback registered below holds a bound-method ref to `self`, which keeps
        # the fuser alive as long as the octree is, and the public scatter/gather take a LevelGeometry directly,
        # so we never need to call `octree.query` from the fuser.
        self._little_endian = octree.little_endian_vertex_order
        self._resolution = float(octree.cfg.resolution)

    # Subclasses override.
    def scatter(self, level_geom: LevelGeometry, values: nn.Parameter, feats: torch.Tensor) -> None:
        """Update `values` in place from the (per-level geometry, feats) pair at `level_geom.level`."""
        raise NotImplementedError

    def gather(self, level_geom: LevelGeometry, values: nn.Parameter) -> torch.Tensor:
        """Read features at the query points cached in `level_geom` using the same mechanism as :meth:`scatter`."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Nearest-vertex fuser
# ---------------------------------------------------------------------------


class NearestVertexFuser(Fuser):
    """Assign each point to the nearest vertex of its leaf voxel and update only that vertex.

    Modes (selected at construction):

        - `overwrite`: each scatter writes `values[v] = f` for the assigned vertex. Last-write-wins; stateless.
        - `running_average`: maintain a streaming mean -- `values[v]` is always the average of all features ever
            assigned to `v` across every scatter call. A per-vertex `counts` buffer (long, `(capacity,)`) tracks
            sample counts and is resized by the octree resize observer.
        - `ema`: `values[v] <- (1 - ema_alpha) * values[v] + ema_alpha * f`. Stateless beyond `ema_alpha`.

    Gather looks up the nearest vertex's value directly (no interpolation). Learnable mode is *not* supported (the
    discrete vertex assignment breaks gradient sharing across neighbors); see DESIGN-VL-Fusion.md for the rationale.
    """

    def __init__(
        self,
        octree: SemiSparseOctree,
        mode: Literal["overwrite", "running_average", "ema"] = "running_average",
        ema_alpha: float = 0.1,
    ):
        super().__init__(octree)
        if mode not in ("overwrite", "running_average", "ema"):
            raise ValueError(f"NearestVertexFuser: unknown mode {mode!r}")
        self.mode = mode
        self.ema_alpha = float(ema_alpha)

        if mode == "running_average":
            # Long buffer; resized in lockstep with the octree via _grow_counts.
            self.register_buffer("counts", torch.zeros((octree.capacity,), dtype=torch.long))
            octree.register_resize_observer(self._grow_counts)
        else:
            # Sentinel; lets isinstance-free code paths check `self.counts is None` to know they are stateless.
            self.counts: Optional[torch.Tensor] = None

    def _grow_counts(self, new_capacity: int) -> None:
        """Resize observer: grow the per-vertex `counts` buffer in place."""
        if self.counts is None:
            return
        grow_param_first_dim(self.counts, new_capacity, fill_value=0)

    def _nearest_vertex_indices(self, level_geom: LevelGeometry) -> torch.Tensor:
        """For each point return the global index of the nearest vertex of its containing voxel at `level_geom`.

        Each axis bit is 1 iff the point is on the upper half of the voxel along that axis.
        The local-ID-from-bits formula depends on the octree's endianness convention.
        """
        bits = (level_geom.voxel_offsets >= 0.5).long()  # (N, 3), entries in {0, 1}
        if self._little_endian:
            vlocal = bits[:, 0] | (bits[:, 1] << 1) | (bits[:, 2] << 2)
        else:
            vlocal = bits[:, 2] | (bits[:, 1] << 1) | (bits[:, 0] << 2)
        return level_geom.vertex_indices.long().gather(1, vlocal.unsqueeze(1)).squeeze(1)  # (N,)

    @torch.no_grad()
    def scatter(self, level_geom: LevelGeometry, values: nn.Parameter, feats: torch.Tensor) -> None:
        vidx = self._nearest_vertex_indices(level_geom)  # (N,)
        valid = (vidx >= 0) & (vidx < values.shape[0])
        if not valid.any():
            return
        vidx = vidx[valid]
        f = feats[valid].to(dtype=values.dtype)

        if self.mode == "overwrite":
            values.data[vidx] = f
            return

        if self.mode == "ema":
            current = values.data[vidx]
            values.data[vidx] = (1.0 - self.ema_alpha) * current + self.ema_alpha * f
            return

        # running_average: streaming mean update. We accumulate the batch's per-vertex sums + counts, then fold them
        # into the existing per-vertex mean so multiple scatter calls compose to the cumulative mean.
        assert self.counts is not None
        batch_sums = torch.zeros_like(values.data)
        batch_sums.index_add_(0, vidx, f)
        batch_counts = torch.zeros_like(self.counts)
        batch_counts.index_add_(0, vidx, torch.ones_like(vidx))

        touched = batch_counts > 0
        if not touched.any():
            return
        new_counts = self.counts + batch_counts
        # Welford-style streaming update: new_mean = old_mean + (batch_sum - batch_count * old_mean) / new_count.
        # The numerator stays small (it's the "new deviation from running mean"), so the final `old + delta`
        # addition keeps full precision even when old_count * old_mean would have swamped batch_sum in the direct
        # `(old_count * old_mean + batch_sum) / new_count` form. Matters for long-running fusion sessions or
        # reduced-precision storage on `values`; safe default otherwise.
        # Untouched rows keep their old mean; clamp(min=1) on the denominator avoids div-by-zero for those rows.
        denom = new_counts.clamp(min=1).to(values.dtype).unsqueeze(-1)
        delta = (batch_sums - batch_counts.to(values.dtype).unsqueeze(-1) * values.data) / denom
        values.data[touched] = values.data[touched] + delta[touched]
        self.counts.copy_(new_counts)

    def gather(self, level_geom: LevelGeometry, values: nn.Parameter) -> torch.Tensor:
        vidx = self._nearest_vertex_indices(level_geom)  # (N,)
        # Indexing with -1 wraps to the last row; the validity mask zeroes that bogus read.
        valid = (vidx >= 0) & (vidx < values.shape[0])
        out = values[vidx]  # (N, D)
        return out * valid.to(values.dtype).unsqueeze(-1)


# ---------------------------------------------------------------------------
# 8-corner kernel-weight fusers (Trilinear / InverseDistance / Rbf / OU)
# ---------------------------------------------------------------------------


class _CornerWeightFuser(Fuser):
    """Shared scaffolding for the 8-corner splat fusers.

    Subclasses override :meth:`_compute_weights` to return `(N, 8)` per-corner weights and override
    :meth:`gather` only if their gather signature deviates from the canonical normalized weighted average
    (only :class:`TrilinearFuser` does, to add the optional Hermite `grads` path).

    Scatter maintains a per-vertex `weight_sum` buffer so the cumulative weighted mean composes across calls:
    `values[v]` is always equal to `sum_p w_p(v) * f_p / sum_p w_p(v)` over every point ever scattered through
    this fuser.
    """

    def __init__(self, octree: SemiSparseOctree):
        super().__init__(octree)
        # Vertex-corner offset tables for the bound octree's endianness. `_offsets11` is `(8, 3)` in `{-1, +1}^3`
        # (per-corner sign on each axis); `_offsets01_flip` is `(8, 3)` in `{0, 1}^3` and equals `1 - offsets01` (the
        # per-corner complement of the `{0, 1}^3` table). The combination lets `_trilinear_weights` compute the per-
        # axis factor as one multiply + one add (see the identity in that method).
        offsets11 = _make_vertex_offsets(self._little_endian, device=torch.device("cpu"), dtype=torch.float32)
        # offsets01 = (offsets11 + 1.0) * 0.5  # (8, 3) in {0, 1}^3
        offsets01_flip = 1.0 - (offsets11 + 1.0) * 0.5  # = 1 - offsets01; see _trilinear_weights for the identity
        self.register_buffer("_offsets11", offsets11, persistent=False)  # (8, 3) in {-1, +1}^3
        self.register_buffer("_offsets01_flip", offsets01_flip, persistent=False)  # (8, 3) in {0, 1}^3, = 1 - offsets01
        self.register_buffer("weight_sum", torch.zeros((octree.capacity,), dtype=torch.float32))
        octree.register_resize_observer(self._grow_weight_sum)

    def _grow_weight_sum(self, new_capacity: int) -> None:
        """Resize observer for the per-vertex `weight_sum` buffer. Same in-place storage swap as
        `FieldStorage._on_octree_resize` -- `grow_param_first_dim` no-ops on the catch-up fire."""
        grow_param_first_dim(self.weight_sum, new_capacity, fill_value=0.0)

    def _corner_positions(self, level_geom: LevelGeometry) -> torch.Tensor:
        """Compute `(N, 8, 3)` metric corner positions for the points' containing voxels at `level_geom`.

        Returns positions ordered by vertex local ID matching the octree's ordering, so `corners[:, i, :]` is the
        metric position of the corner whose global index is `level_geom.vertex_indices[:, i]`.
        """
        # voxel_sizes is in grid units (e.g. 1 for the finest level); convert to metric half-sizes.
        half_metric = (level_geom.voxel_sizes.float() * 0.5 * self._resolution).view(-1, 1, 1)  # (N, 1, 1)
        return level_geom.voxel_centers.view(-1, 1, 3) + half_metric * self._offsets11.view(1, 8, 3)  # (N, 8, 3)

    def _trilinear_weights(self, voxel_offsets: torch.Tensor) -> torch.Tensor:
        """Compute the 8 trilinear interpolation weights for each point.

        `voxel_offsets`: `(N, 3)` point coordinates normalized to the unit voxel `[0, 1]^3`. Returns `(N, 8)`
        weights that sum to 1 across the 8 corners, ordered to match the octree's vertex-local-ID convention.

        Algebraic identity used to collapse each per-axis factor to one multiply + one add:
            (p * q + (1 - p) * (1 - q)).prod()  where q in {0, 1} = offsets01, p = voxel_offsets
          = (1 - q - p + 2 * p * q).prod()
          = ((1 - q) + p * (2q - 1)).prod()
          = (offsets01_flip + p * offsets11).prod()
        """
        p = voxel_offsets.unsqueeze(1)  # (N, 1, 3)
        return (self._offsets01_flip.view(1, 8, 3) + p * self._offsets11.view(1, 8, 3)).prod(dim=-1)  # (N, 8)

    def _compute_weights(self, level_geom: LevelGeometry) -> torch.Tensor:
        """Return per-corner weights `(N, 8)` ordered to match `level_geom.vertex_indices`. Override in subclasses."""
        raise NotImplementedError

    def _corner_valid_mask(self, level_geom: LevelGeometry, values_capacity: int) -> torch.Tensor:
        """`(N, 8)` bool mask: True where the corner's global vertex index is in range. Used to zero out the
        weight (and downstream contribution) for absent vertices.
        """
        vidx = level_geom.vertex_indices.long()  # (N, 8)
        return (vidx >= 0) & (vidx < values_capacity)

    @torch.no_grad()
    def scatter(self, level_geom: LevelGeometry, values: nn.Parameter, feats: torch.Tensor) -> None:
        weights = self._compute_weights(level_geom)  # (N, 8)
        valid = self._corner_valid_mask(level_geom, values.shape[0])  # (N, 8)
        weights = weights * valid.to(weights.dtype)  # zero contributions from absent corners

        # Flatten the per-point-per-corner contributions into per-vertex index_add accumulations.
        # vidx_flat: (N * 8,), w_flat: (N * 8,), wf_flat: (N * 8, D).
        N = level_geom.points.shape[0]
        D = values.shape[1]
        vidx_flat = level_geom.vertex_indices.long().clamp(min=0).reshape(-1)  # clamp -1 to 0; valid mask zeroes weight
        w_typed = weights.to(values.dtype)  # (N, 8)
        w_flat = w_typed.reshape(-1)  # (N * 8,)
        # Broadcast (N, 1, D) * (N, 8, 1) -> (N, 8, D) without materializing an explicit expand of feats.
        wf_flat = (feats.to(values.dtype).unsqueeze(1) * w_typed.unsqueeze(-1)).reshape(N * 8, D)

        batch_weight_sum = torch.zeros_like(self.weight_sum)
        batch_weight_sum.index_add_(0, vidx_flat, w_flat.to(self.weight_sum.dtype))
        batch_weighted_features = torch.zeros_like(values.data)
        batch_weighted_features.index_add_(0, vidx_flat, wf_flat)

        touched = batch_weight_sum > 0
        if not touched.any():
            return
        new_weight_sum = self.weight_sum + batch_weight_sum
        # new_mean = (old_weight_sum * old_mean + batch_weighted_features) / new_weight_sum, touched rows only.
        denom = new_weight_sum.clamp(min=torch.finfo(values.dtype).tiny).to(values.dtype).unsqueeze(-1)
        numer = values.data * self.weight_sum.to(values.dtype).unsqueeze(-1) + batch_weighted_features
        values.data[touched] = (numer / denom)[touched]
        self.weight_sum.copy_(new_weight_sum)

    def gather(self, level_geom: LevelGeometry, values: nn.Parameter) -> torch.Tensor:
        weights = self._compute_weights(level_geom)  # (N, 8)
        valid = self._corner_valid_mask(level_geom, values.shape[0])  # (N, 8)
        weights = weights * valid.to(weights.dtype)
        norm = weights.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(weights.dtype).tiny)
        weights = weights / norm
        gathered = values[level_geom.vertex_indices.long()]  # (N, 8, D)
        return torch.einsum("ni,nik->nk", weights.to(values.dtype), gathered)  # (N, D)


class TrilinearFuser(_CornerWeightFuser):
    """8-corner splat with trilinear-basis weights.

    Trilinear weights are a partition of unity (sum to 1 across the 8 corners at every interior point) so the
    scatter spreads each feature with total weight 1, and the gather is a true linear interpolation. Gather supports
    the gradient-augmented (Hermite) variant via the optional `grads` argument: when `grads` is provided, each
    per-corner value is replaced by its first-order Taylor extrapolation toward the query point
    (`value_i + grad_i . (point - corner_i)`) before the standard trilinear blend. Returns a scalar field with the
    trailing `D=1` axis preserved.

    Scatter never touches `grads`: a single scattered feature does not define a unique Hermite gradient update, so
    GA stays purely a gather-side concern. The trainer is responsible for optimizing `grads` separately.
    """

    def _compute_weights(self, level_geom: LevelGeometry) -> torch.Tensor:
        return self._trilinear_weights(level_geom.voxel_offsets)

    def gather(
        self,
        level_geom: LevelGeometry,
        values: nn.Parameter,
        *,
        grads: Optional[nn.Parameter] = None,
    ) -> torch.Tensor:
        if grads is None:
            return super().gather(level_geom, values)

        # Gradient-augmented (Hermite) trilinear gather. Only valid for scalar fields; we restore the trailing D=1
        # axis at the end so callers see a uniform (N, D) shape.
        assert values.shape[-1] == 1, (
            "TrilinearFuser.gather with grads is gradient-augmented and requires a scalar field "
            f"(values.shape[-1] == 1); got shape {tuple(values.shape)}"
        )
        # No validity mask on the GA path. A -1 in `vertex_indices` wraps to the last row of `values` / `grads`.
        # Callers needing strict boundary handling should mask their loss at the points returned with any missing corner.
        vidx = level_geom.vertex_indices.long()  # (N, 8)
        vertex_values = values[vidx].squeeze(-1)  # (N, 8)
        vertex_grad = grads[vidx]  # (N, 8, 3)

        # Per-corner Hermite extrapolation: augmented_i = value_i + grad_i . (point - corner_i). Corner positions
        # come from the cached offset buffers so the projection has no autograd path through geometry.
        corners = self._corner_positions(level_geom)  # (N, 8, 3)
        diffs = level_geom.points.unsqueeze(1) - corners  # (N, 8, 3)
        projection = torch.einsum("nik,nik->ni", vertex_grad, diffs)  # (N, 8)
        augmented = vertex_values + projection  # (N, 8)

        # Standard trilinear blend on the augmented corner values; uses the same identity as the non-GA gather.
        weights = self._trilinear_weights(level_geom.voxel_offsets).to(values.dtype)  # (N, 8)
        return (weights * augmented).sum(dim=1, keepdim=True)  # (N, 1)


class InverseDistanceFuser(_CornerWeightFuser):
    """8-corner splat with inverse-distance kernel weights.

    `w_i = 1 / (||p - c_i|| + epsilon)` in metric units. The bias `epsilon` keeps the kernel finite when a query
    point lands exactly on a corner; it has the same length units as `octree.cfg.resolution`.
    """

    def __init__(self, octree: SemiSparseOctree, epsilon: float = 1e-6):
        super().__init__(octree)
        self.epsilon = float(epsilon)

    def _compute_weights(self, level_geom: LevelGeometry) -> torch.Tensor:
        corners = self._corner_positions(level_geom)  # (N, 8, 3)
        dists = (level_geom.points.unsqueeze(1) - corners).norm(dim=-1)  # (N, 8)
        return 1.0 / (dists + self.epsilon)


class RbfFuser(_CornerWeightFuser):
    """8-corner splat with Gaussian RBF kernel weights.

    `w_i = exp(-||p - c_i||^2 / (2 * bandwidth^2))` in metric units. Smooth (C^infinity) at vertices, which makes
    the learnable-mode gather well-behaved for query points sitting near voxel corners.
    """

    def __init__(self, octree: SemiSparseOctree, bandwidth: float = 0.1):
        super().__init__(octree)
        self.bandwidth = float(bandwidth)

    def _compute_weights(self, level_geom: LevelGeometry) -> torch.Tensor:
        corners = self._corner_positions(level_geom)
        sqdists = ((level_geom.points.unsqueeze(1) - corners) ** 2).sum(dim=-1)  # (N, 8)
        return torch.exp(-sqdists / (2.0 * self.bandwidth * self.bandwidth))


class OrnsteinUhlenbeckFuser(_CornerWeightFuser):
    """8-corner splat with Ornstein-Uhlenbeck (Matern-1/2) kernel weights.

    `w_i = exp(-||p - c_i|| / bandwidth)` in metric units. Heavier tails than :class:`RbfFuser` but the kernel is
    C^0 (not C^1) at voxel corners; the learnable-mode gather acquires a non-smooth crease at every vertex and
    produces unstable gradients on `values` for query points sitting on or very near a vertex (a common case for
    points-on-surface data). Prefer this fuser only for direct-update scatter.
    """

    def __init__(self, octree: SemiSparseOctree, bandwidth: float = 0.1):
        super().__init__(octree)
        self.bandwidth = float(bandwidth)

    def _compute_weights(self, level_geom: LevelGeometry) -> torch.Tensor:
        corners = self._corner_positions(level_geom)
        dists = (level_geom.points.unsqueeze(1) - corners).norm(dim=-1)  # (N, 8)
        return torch.exp(-dists / self.bandwidth)


__all__ = [
    "Fuser",
    "FuserConfig",
    "NearestVertexFuser",
    "NearestVertexFuserConfig",
    "TrilinearFuser",
    "TrilinearFuserConfig",
    "InverseDistanceFuser",
    "InverseDistanceFuserConfig",
    "RbfFuser",
    "RbfFuserConfig",
    "OrnsteinUhlenbeckFuser",
    "OrnsteinUhlenbeckFuserConfig",
    "build_fuser",
]
