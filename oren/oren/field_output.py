"""Per-call geometry cache + per-field output types used by FieldStorage.

`LevelGeometry` is the immutable per-level snapshot the field math actually consumes (positions, voxel indices, vertex
indices, voxel offsets). One `OctreeGeometry` wraps a level-1 snapshot together with a cache for higher levels,
populated lazily through `at_level(k)`. The cache is built once per forward pass and shared by every `FieldStorage` on
a `FieldBank` so multiple fields on the same octree don't redo geometry work.

This module is intentionally octree-agnostic — `OctreeGeometry` stores a `compute_level` callable rather than a
back-reference to `SemiSparseOctree`, so `field_output.py` has no import on the octree side and there is no structural
cycle between the two modules.
"""

from dataclasses import InitVar, dataclass, field
from typing import Callable, Optional

import torch


@dataclass
class LevelGeometry:
    """Per-level snapshot for a single query batch.

    All tensors are aligned along the leading point dimension. The `level` attribute is informational; identical-shape
    tensors at different levels are *not* interchangeable because `voxel_indices` and `vertex_indices` index into
    different rows of the octree's per-node buffers.

    `points` is a back-reference to the world-space query points that produced this snapshot; identical (by
    reference) across every level returned from the same `OctreeGeometry`. Carrying it here lets per-level
    consumers (e.g. fusers) work from a single self-contained object instead of an `(OctreeGeometry, level)` pair.
    """

    # Octree level (1 = leaf, increases upward); informational only.
    level: int
    # (..., 3) world-space query points; shared by reference with the parent OctreeGeometry.
    points: torch.Tensor
    # (...,) leaf-or-deeper index per point at this level.
    voxel_indices: torch.Tensor
    # (..., 3) metric voxel centers.
    voxel_centers: torch.Tensor
    # (..., 1) grid-unit voxel sizes.
    voxel_sizes: torch.Tensor
    # (..., 8) 8 vertex indices per voxel.
    vertex_indices: torch.Tensor
    # (..., 3) point coords normalized to [0, 1]^3 within the voxel.
    voxel_offsets: torch.Tensor


@dataclass
class OctreeGeometry:
    """Per-call multi-level geometry cache.

    Built once by `octree.query(points)` and handed to every `FieldStorage` on the bank. Level 1 is populated eagerly
    (every field needs it). Higher levels are filled on first `at_level(k)` call and cached for the rest of this
    forward pass. The cache lifetime is exactly one forward pass; no global state.

    `compute_level` is a callable `(points, level) -> LevelGeometry` injected by the producer (typically
    `SemiSparseOctree._compute_level_geometry`). Storing a callable instead of a back-reference to the octree keeps
    this module free of any import on the octree side.
    """

    # (n_points, 3) flat query batch this cache is built for.
    points: torch.Tensor
    # Callable producing a LevelGeometry for a given level (injected; see class docstring).
    compute_level: Callable[[torch.Tensor, int], LevelGeometry]
    # Lazy per-level cache; populated by at_level().
    _levels: dict[int, LevelGeometry] = field(default_factory=dict)
    # Batch shape of the user-provided points before FieldBank flattened them (e.g. `(b, m)` for `(b, m, 3)` input).
    # Empty tuple means the caller passed flat `(n_points, 3)` or chunked processing is in progress (FieldBank applies
    # the reshape once on the final concatenated output). FieldStorage forwards this through to the FieldOutput it
    # constructs, whose `__post_init__` reshapes its per-point tensors back to `(*batch_shape, D)` / `(*batch_shape,)`.
    batch_shape: tuple[int, ...] = ()

    def at_level(self, level: int = 1) -> LevelGeometry:
        """Return the `LevelGeometry` for `level`, computing and caching it on first access.

        Args:
            level: octree level (1 = leaf, increases upward).

        Returns:
            The cached or freshly-computed `LevelGeometry` for `points` at `level`.
        """
        if level not in self._levels:
            self._levels[level] = self.compute_level(self.points, level)
        return self._levels[level]


@dataclass
class FieldOutput:
    """Per-field output for one forward pass.

    `prior` is `None` when the field's mode is `"implicit"` (no explicit branch); `implicit` is `None` when the mode is
    `"explicit"` (no MLP head); both populated in `"hybrid"` mode. `pred` is always the final prediction the field's
    criterion consumes.

    `__post_init__` normalizes the per-point tensors so callers see a uniform `(..., D)` convention: if `pred`,
    `prior`, or `implicit` arrives as `(n_points,)` (i.e. a producer squeezed away the scalar `D=1` axis), it's
    promoted to `(n_points, 1)` so `pred.shape[-1]` is always the feature dim. `voxel_indices` stays a plain index
    tensor and is not unsqueezed.

    `batch_shape` is an init-only knob (does NOT become an instance attribute). When non-empty, `__post_init__`
    then replaces the leading `(n_points,)` axis with `(*batch_shape,)` and preserves every trailing dim
    (`shape[1:]`), so pred/prior/implicit end at `(*batch_shape, D, ...)` and voxel_indices at `(*batch_shape,)`.
    This pushes the user's original batched leading dims back onto the outputs so callers (SdfNetwork / OccNetwork
    / MultiFieldTrainer) can return the FieldOutput directly without post-processing. FieldBank.forward sets
    `geom.batch_shape` from the user-provided `points.shape[:-1]` for single-shot processing, or applies it once
    after concatenating per-chunk outputs in the VRAM-bounded chunked path.
    """

    # (n_points,) flat; reshaped to (*batch_shape,) by __post_init__.
    voxel_indices: torch.Tensor
    # (n_points, D, ...) or (n_points,); reshaped to (*batch_shape, D, ...).
    prior: Optional[torch.Tensor]
    # Same shape convention as `prior`.
    implicit: Optional[torch.Tensor]
    # Same shape convention as `prior`.
    pred: torch.Tensor
    # Init-only; not stored on the instance. See class docstring.
    batch_shape: InitVar[tuple[int, ...]] = ()

    def __post_init__(self, batch_shape: tuple[int, ...]):
        # Promote `(n,)` to `(n, 1)` so callers can always rely on a trailing feature axis. Cheap because
        # `unsqueeze` is a view, not a copy.
        if self.pred.dim() == 1:
            self.pred = self.pred.unsqueeze(-1)
        if self.prior is not None and self.prior.dim() == 1:
            self.prior = self.prior.unsqueeze(-1)
        if self.implicit is not None and self.implicit.dim() == 1:
            self.implicit = self.implicit.unsqueeze(-1)

        if not batch_shape:
            return
        self.voxel_indices = self.voxel_indices.view(batch_shape)
        self.pred = self.pred.view(batch_shape + self.pred.shape[1:])
        if self.prior is not None:
            self.prior = self.prior.view(batch_shape + self.prior.shape[1:])
        if self.implicit is not None:
            self.implicit = self.implicit.view(batch_shape + self.implicit.shape[1:])

    @staticmethod
    def concatenate(outputs: list["FieldOutput"], batch_shape: tuple[int, ...]) -> "FieldOutput":
        """Concatenate a list of per-chunk FieldOutputs into one for the whole batch.

        Args:
            outputs: list of FieldOutputs for each chunk;
            batch_shape: original leading batch shape of the user-provided points.
        Returns:
            A single FieldOutput for the whole batch, with the per-chunk outputs concatenated along the leading point
            dimension and reshaped to `(*batch_shape, D, ...)` convention.
        """
        if len(outputs) == 1:
            return outputs[0]
        # Narrow `list[Tensor | None]` to `list[Tensor]` via the filter; `first.{prior,implicit}` not-None implies all
        # chunks agree (same field mode), so the filter is redundant at runtime and only there for the type checker.
        prior_chunks = [o.prior for o in outputs if o.prior is not None]
        implicit_chunks = [o.implicit for o in outputs if o.implicit is not None]
        if prior_chunks:
            assert len(prior_chunks) == len(outputs), "All chunks must have `prior` if any chunk has `prior`"
        if implicit_chunks:
            assert len(implicit_chunks) == len(outputs), "All chunks must have `implicit` if any chunk has `implicit`"
        return FieldOutput(
            voxel_indices=torch.cat([o.voxel_indices for o in outputs], dim=0),
            prior=torch.cat(prior_chunks, dim=0) if prior_chunks else None,
            implicit=torch.cat(implicit_chunks, dim=0) if implicit_chunks else None,
            pred=torch.cat([o.pred for o in outputs], dim=0),
            batch_shape=batch_shape,
        )
