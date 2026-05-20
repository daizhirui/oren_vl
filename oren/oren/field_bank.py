"""Owner of one octree + N FieldStorages + optional shared FeatureBanks.

`FieldBank` is what trainers consume. Each call: build the per-call `OctreeGeometry` once via `octree.query(points)`,
dispatch the same cached geometry to every field, and gather their `FieldOutput`s into a dict keyed by field name.

Shared `FeatureBank` instances live on the FieldBank (not on any single field), so the `(V, F)` parameter for `"geo"`
appears exactly once in `state_dict()` regardless of how many fields read it. Private banks (the default when a field's
`shared_bank=None`) live on the field's submodule and are not visible here.
"""

from typing import Iterable, Optional

import torch
import torch.nn as nn

from oren.feature_bank import FeatureBank, FeatureBankConfig
from oren.field_output import FieldOutput, OctreeGeometry
from oren.field_storage import FieldStorage
from oren.field_storage_config import AuxiliaryBankSpec, FieldStorageConfig
from oren.ga_trilinear import trilinear_interpolation
from oren.semi_sparse_octree import SemiSparseOctree


class FieldBank(nn.Module):
    """One octree + a set of FieldStorages + optional shared FeatureBanks."""

    def __init__(
        self,
        octree: SemiSparseOctree,
        fields: Iterable[FieldStorageConfig],
        shared_banks: Iterable[FeatureBankConfig] = (),
    ):
        """Build a FieldBank around `octree` with the given fields and shared FeatureBanks.

        Args:
            octree: geometry provider shared across all FieldStorages. Held as a plain attribute (not a submodule) so
                its parameters do not appear under this bank's state_dict prefix.
            fields: per-field configs; one `FieldStorage` is constructed per entry, validated against `shared_banks`.
            shared_banks: declarations for cross-field-shared `FeatureBank`s. Each is registered under
                `banks.<name>` and wired as an octree resize observer.
        """
        super().__init__()
        # `octree` is shared across all FieldStorage instances; keep a plain attribute (not a submodule) so its
        # parameters/buffers don't appear under the bank's state_dict prefix.
        # FieldBank's state_dict is meant to be the union of its fields' state_dicts, so the octree's parameters/buffers
        # should be stored separately. e.g. in SdfNetwork, the octree is stored on the SdfNetwork itself, not on the
        # FieldBank submodule.
        object.__setattr__(self, "octree", octree)
        self.octree: SemiSparseOctree

        # Initial per-vertex-parameter capacity is the octree's announced capacity (pow-2-rounded vertex high-water
        # mark), NOT `octree.cfg.init_voxel_num` (which is the C++ node-buffer capacity, a different quantity that
        # conflates voxel count with vertex count).
        init_capacity = octree.capacity

        # ---- Shared FeatureBank construction + observer wiring ----
        # Each shared bank is registered as a submodule (state_dict scope: `banks.<name>.features`) AND as an octree
        # resize observer so its features grow with the C++ tree. The catch-up fire in `register_resize_observer`
        # brings each bank up to the current vertex count immediately, so banks created after insertion still work.
        self.banks: nn.ModuleDict = nn.ModuleDict()
        for bank_cfg in shared_banks:
            bank = FeatureBank(bank_cfg, init_capacity=init_capacity)
            self.banks[bank.name] = bank
            octree.register_resize_observer(bank.grow_to)

        # ---- FieldStorage construction ----
        # Validates `shared_bank` and `auxiliary_banks` references against the shared bank set declared above.
        # Constraints:
        #   - `shared_bank` must reference a *shared* bank with matching `implicit_feature_dim`.
        #   - Every `AuxiliaryBankSpec.name` must reference a *shared* bank (pointing at another field's private bank
        #     is rejected, since it would create coupling invisible at the call site).
        #   - Each field's head input dim is recomputed here to include any aux-bank channels concatenated downstream.
        self.fields: nn.ModuleDict = nn.ModuleDict()
        # Resolve each field's auxiliary bank list to actual FeatureBank references; we look these up on every forward
        # to gather the aux features upstream of the field's head.
        self._aux_specs: dict[str, list[AuxiliaryBankSpec]] = {}

        for f in fields:
            self._validate_field(f)

            # Aux-bank channels (single-level leaf gather per aux bank in phase 2; multi-level aux is a future
            # extension). Each aux bank's per-point channel count is its `feature_dim`, or its shared trunk's
            # `out_dim` if present (not exercised yet).
            aux_total = sum(self.banks[a.name].feature_dim for a in f.auxiliary_banks)

            shared_bank: Optional[FeatureBank] = self.banks[f.shared_bank] if f.shared_bank else None
            # FieldStorage handles the rest of the head-dim arithmetic (own-bank aggregation + hybrid-prior concat).
            # Passing `aux_extra_dim` here avoids the previous race where both FieldBank and FieldStorage tried to set
            # `implicit_net_cfg.implicit_feature_dim` independently.
            field = FieldStorage(f, octree, bank=shared_bank, aux_extra_dim=aux_total)
            self.fields[f.name] = field
            self._aux_specs[f.name] = list(f.auxiliary_banks)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_field(self, f: FieldStorageConfig) -> None:
        assert f.name, "FieldStorageConfig.name must be a non-empty string"

        if f.shared_bank is not None:
            assert f.shared_bank in self.banks, (
                f"field {f.name!r} declares shared_bank={f.shared_bank!r} "
                f"which is not in the FieldBank's shared_banks ({list(self.banks)})"
            )
            bank = self.banks[f.shared_bank]
            assert f.implicit_feature_dim == bank.feature_dim, (
                f"field {f.name!r} shared_bank={f.shared_bank!r}: "
                f"implicit_feature_dim={f.implicit_feature_dim} mismatches "
                f"bank.feature_dim={bank.feature_dim}"
            )

        for aux in f.auxiliary_banks:
            assert aux.name, (
                f"field {f.name!r}: auxiliary_banks entry has empty name; "
                "this is the sentinel default of AuxiliaryBankSpec and must be set"
            )
            assert aux.name in self.banks, (
                f"field {f.name!r}: auxiliary_banks references {aux.name!r}, which "
                "must be a *shared* bank declared on FieldBank.shared_banks "
                f"(known shared banks: {list(self.banks)}). Pointing at another "
                "field's private bank is rejected on purpose"
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _gather_aux_features(
        self,
        aux_specs: list[AuxiliaryBankSpec],
        geom: OctreeGeometry,
    ) -> Optional[torch.Tensor]:
        """Trilinear-interpolate per-point features from each named aux bank, apply `.detach()` per spec, concat.
        Single-level (leaf) lookup — multi-level aux aggregation is a follow-up extension. Returns None when
        `aux_specs` is empty so the field's forward skips the concat.
        """
        if not aux_specs:
            return None

        leaf = geom.at_level(1)
        outs: list[torch.Tensor] = []
        for aux in aux_specs:
            bank = self.banks[aux.name]
            per_point_vertex = bank.features[leaf.vertex_indices]
            feats = trilinear_interpolation(
                points=leaf.voxel_offsets,
                per_point_vertex_values=per_point_vertex,
                little_endian=self.octree.little_endian_vertex_order,
            )
            if aux.detach:
                feats = feats.detach()
            if bank.shared_net is not None:
                feats = bank.shared_net(feats)
            outs.append(feats)
        return torch.cat(outs, dim=-1)

    def forward(
        self,
        points: torch.Tensor,
        voxel_indices: Optional[torch.Tensor] = None,
        prior_only: bool = False,
        batch_size: Optional[int] = None,
    ) -> dict[str, FieldOutput]:
        """Build one geometry cache, dispatch to every field, return a dict keyed by field name. Higher-level lookups
        (>level 1) the fields request all hit the cache on `geom`, so a 3-field setup walking levels 1..3 does 3 C++
        trips (not 9).

        The user-provided `points` may be batched (e.g. `(b, m, 3)`). FieldBank flattens to `(b*m, 3)` for the octree
        query, stores the original `batch_shape` on `geom`, and `FieldStorage` forwards it to the returned
        `FieldOutput` whose `__post_init__` reshapes per-point tensors back to `(*batch_shape, D)` / `(*batch_shape,)`.

        When `batch_size` is given and smaller than the flat point count, the flat point batch is processed in
        contiguous chunks and the per-chunk `FieldOutput` tensors are concatenated along the point axis before the
        single final reshape. Chunking keeps memory bounded for large grids / dense queries; the result is
        numerically identical to the single-shot path (per-point math has no cross-point dependency).

        Args:
            points: (..., 3) query points in world coordinates.
            voxel_indices: optional (...,) precomputed leaf voxel indices. When None, the octree resolves them.
            prior_only: if True, every field skips its implicit branch and returns only the explicit prior.
            batch_size: optional chunk size for the flat point axis. None or >= n_points uses a single forward pass.

        Returns:
            dict mapping each field name to its `FieldOutput` (prior / implicit / pred populated per that field's
            mode); tensor fields are reshaped back to the caller's `batch_shape`.
        """
        batch_shape = tuple(points.shape[:-1])
        points = points.view(-1, 3)
        if voxel_indices is not None:
            voxel_indices = voxel_indices.view(-1)
        n_points = points.shape[0]

        # Single-shot fast path: hand the geom its batch_shape so FieldStorage's FieldOutput reshapes in __post_init__.
        if batch_size is None or batch_size >= n_points:
            geom = self.octree.query(points, voxel_indices)
            geom.batch_shape = batch_shape
            out: dict[str, FieldOutput] = {}
            for name, field in self.fields.items():
                extra = self._gather_aux_features(self._aux_specs[name], geom)
                out[name] = field(geom, prior_only=prior_only, extra_feats=extra)
            return out

        # Chunked path: per-chunk FieldOutputs stay flat (geom.batch_shape = ()), get concatenated, and the final
        # FieldOutput per field applies batch_shape once via its __post_init__.
        chunks: dict[str, list[FieldOutput]] = {name: [] for name in self.fields}
        for start in range(0, n_points, batch_size):
            stop = min(start + batch_size, n_points)
            sub_points = points[start:stop]
            sub_voxel = voxel_indices[start:stop] if voxel_indices is not None else None
            geom = self.octree.query(sub_points, sub_voxel)
            for name, field in self.fields.items():
                extra = self._gather_aux_features(self._aux_specs[name], geom)
                chunks[name].append(field(geom, prior_only=prior_only, extra_feats=extra))

        out = {}
        for name, fos in chunks.items():
            out[name] = FieldOutput.concatenate(fos, batch_shape=batch_shape)
        return out

    # ------------------------------------------------------------------
    # Per-vertex parameter enumeration + optimizer-state migration
    # ------------------------------------------------------------------

    def iter_per_vertex_params(self) -> Iterable[nn.Parameter]:
        """Yield every `nn.Parameter` whose leading dim tracks `octree.capacity`.

        These are the parameters whose `.data` gets resized in-place on every pow-2 boundary crossing. Used by
        `_migrate_optim_state` to find which optimizer state tensors need their leading dim grown to match.

        Yields:
            Per-vertex `nn.Parameter`s in order: each field's `values`, `grads`, and (if private) own-bank
            `features`, followed by every shared bank's `features`.
        """
        for field in self.fields.values():
            if field.values is not None:
                yield field.values
            if field.grads is not None:
                yield field.grads
            if field.own_bank is not None:
                yield field.own_bank.features
        for bank in self.banks.values():
            yield bank.features

    def attach_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Wire the optimizer so its per-vertex state survives octree resizes.

        Call once, immediately after constructing the optimizer with this bank's parameters. Registers a resize
        observer on the octree that walks `optimizer.state` and grows any state tensor whose leading dim is smaller
        than the corresponding parameter's leading dim (Adam's `exp_avg` / `exp_avg_sq`, etc.).

        The observer is registered *last*, so it fires *after* the per-vertex parameters have already been resized in
        place by `FeatureBank.grow_to` and `FieldStorage._on_octree_resize` (octree fires observers in registration
        order). That way state migration sees the new parameter shape it needs to match.

        Args:
            optimizer: the torch optimizer whose state should be migrated alongside per-vertex parameter resizes.
        """
        # Snapshot the per-vertex parameter set at attach time. New fields added later won't be covered — but FieldBank
        # doesn't support post-construction field addition today, so this matches reality.
        per_vertex = list(self.iter_per_vertex_params())

        def _migrate(new_capacity: int) -> None:
            for p in per_vertex:
                state = optimizer.state.get(p)
                if not state:
                    continue
                for k, v in state.items():
                    # Skip scalars (e.g. Adam's per-param 'step' counter).
                    if not isinstance(v, torch.Tensor) or v.dim() == 0:
                        continue
                    # Only resize state tensors keyed on the per-vertex leading dim. If this state matches
                    # `p.shape[0]` already, it was either created fresh post-resize or matches some other axis
                    # layout — leave it alone.
                    if v.shape[0] >= p.shape[0]:
                        continue
                    new_v = torch.zeros(
                        (p.shape[0], *v.shape[1:]),
                        dtype=v.dtype,
                        device=v.device,
                    )
                    new_v[: v.shape[0]] = v
                    state[k] = new_v

        self.octree.register_resize_observer(_migrate)
