"""Shared implicit-feature bank for one or more FieldStorage instances.

A `FeatureBank` owns a `(V, F)` parameter tensor plus an optional shared trunk (`ImplicitNet`). Multiple `FieldStorage`
instances can point at the same bank via their `shared_bank` config setting; they all consume the bank's *entire*
feature vector (no per-field slicing).

Sizing: `features` is allocated at `init_capacity` (typically `octree.capacity`, the pow-2-rounded vertex high-water
mark) and grows in lockstep with the octree's `num_vertices` via a resize observer. The grow operation resizes
`self.features.data` in place — the `nn.Parameter` object identity is preserved, so any optimizer holding a reference
(e.g. Adam's param_groups) keeps working without rebind. Optimizer *state* (`exp_avg`, `exp_avg_sq`) is sized to the old
capacity though; `FieldBank.attach_optimizer` installs the migration callback that resizes that state in lockstep.
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from oren.implicit_net import ImplicitNet, ImplicitNetConfig
from oren.utils.config_abc import ConfigABC
from oren.utils.param_resize import grow_param_first_dim


@dataclass
class FeatureBankConfig(ConfigABC):
    """Shared FeatureBank declaration consumed by FieldBank.

    All FieldStorages naming this bank via `shared_bank` or `auxiliary_banks` must agree on `feature_dim`.
    `init_capacity` is intentionally not configured here — it's a runtime quantity that comes from the octree's
    announced `capacity` (pow-2 of `max(num_vertices, init_vertex_num)`).

    `init` selects the per-vertex feature initialization scheme. The default `"zero"` matches the historical
    behavior (well-defined baseline at every vertex for trilinear interpolation). Non-zero schemes let the
    downstream head's first layer see varied input from the start, at the cost of injecting noise the head must
    learn to ignore on still-untrained vertices. `init_scale` acts as a knob whose meaning depends on the scheme:

        - "zero":          features stay at 0; `init_scale` is ignored.
        - "normal":        features ~ N(0, init_scale). Default scale: 1e-2.
        - "uniform":       features ~ U(-init_scale, init_scale).
        - "xavier_normal": features ~ N(0, init_scale * sqrt(1 / feature_dim)). `init_scale` is the Glorot gain.
                            Useful when `feature_dim` varies: scales per-vertex feature magnitude to ~ init_scale
                            regardless of `F`.

    New rows added by `grow_to` on octree resize follow the same scheme so the "old" and "new" vertex regions are
    drawn from the same distribution.
    """

    name: str = None  # identifier referenced by FieldStorageConfig.shared_bank / auxiliary_banks
    feature_dim: int = 4
    shared_net_cfg: Optional[ImplicitNetConfig] = None
    init: Literal["zero", "normal", "uniform", "xavier_normal"] = "zero"
    init_scale: float = 1e-2
    # Allocate a `features_used` bool buffer and sparse-encode `features` in state_dict (drop untouched rows).
    # Mirrors `FieldStorageConfig.track_used_vertices` and applies independently. For private banks owned by a
    # FieldStorage, the field's `track_used_vertices` flag is copied here at private-bank construction time;
    # for *shared* banks declared on a FieldBank, set this in YAML directly.
    track_used_vertices: bool = False


class FeatureBank(nn.Module):
    """`(V, F)` implicit-feature parameter plus optional shared trunk.

    Args:
        cfg: declaration carrying name, feature_dim, init scheme, and (optional) shared trunk config. See
            `FeatureBankConfig`.
        init_capacity: initial vertex capacity (V). Typically passed as `octree.capacity` — pow-2-rounded above the
            octree's current vertex count. Grows monotonically via the octree's resize observer.
    """

    def __init__(self, cfg: FeatureBankConfig, init_capacity: int):
        super().__init__()
        assert cfg.name, "FeatureBankConfig.name must be a non-empty string"
        self.cfg = cfg
        self.name = cfg.name
        self.feature_dim = cfg.feature_dim
        # Start with zeros so the "zero" scheme is a no-op; non-zero schemes overwrite the freshly allocated rows.
        self.features = nn.Parameter(torch.zeros((init_capacity, cfg.feature_dim), dtype=torch.float32))
        if cfg.init != "zero":
            self._init_rows_inplace(self.features.data)
        self.shared_net: Optional[ImplicitNet] = (
            ImplicitNet(cfg.shared_net_cfg) if cfg.shared_net_cfg is not None else None
        )

        # Touched-vertex tracking for sparse checkpoint storage. The mask itself is non-persistent (it is implied by
        # the indices saved in `features_used_indices`); _save_to_state_dict / _load_from_state_dict below take care
        # of the encoding. Multiple FieldStorage gathers on the same bank OR-mark it.
        if cfg.track_used_vertices:
            self.register_buffer("features_used", torch.zeros((init_capacity,), dtype=torch.bool), persistent=False)
        else:
            self.features_used: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _init_rows_inplace(self, rows: torch.Tensor) -> None:
        """Apply `cfg.init` to `rows` in place. Used both for initial allocation and for new rows on grow.

        `rows` is a `(K, feature_dim)` view (a slice of `self.features.data`); the operation writes into its
        underlying storage. No-op for the "zero" scheme.
        """
        scheme = self.cfg.init
        scale = self.cfg.init_scale
        if scheme == "zero":
            return
        if scheme == "normal":
            rows.normal_(mean=0.0, std=scale)
        elif scheme == "uniform":
            rows.uniform_(-scale, scale)
        elif scheme == "xavier_normal":
            # Glorot-style for a (V, F) embedding table: scale by 1/sqrt(F) so the per-vertex feature magnitude is
            # roughly `init_scale` (gain) regardless of feature_dim. We deliberately don't bring V into the std --
            # PyTorch's xavier_normal_ would use std=sqrt(2/(V+F)), which shrinks as the octree grows.
            std = scale * math.sqrt(1.0 / self.feature_dim)
            rows.normal_(mean=0.0, std=std)
        else:  # pragma: no cover -- guarded by the Literal in FeatureBankConfig
            raise ValueError(f"unknown feature init scheme {scheme!r}")

    def grow_to(self, new_capacity: int) -> None:
        """Resize observer entry point — grow `features` to `new_capacity` rows.

        Optimizer *state* migration is handled by `FieldBank.attach_optimizer` (its migrator is registered after this
        callback on the same octree). Newly appended rows are initialized with the same scheme as the original
        rows so the bank stays distribution-homogeneous across pow-2 boundary crossings.

        Args:
            new_capacity: new leading-dim row count to grow `features` to; must be >= current size.
        """
        old_size = self.features.shape[0]
        grew = grow_param_first_dim(self.features, new_capacity, fill_value=0.0)
        if grew and self.cfg.init != "zero":
            self._init_rows_inplace(self.features.data[old_size:])
        if self.features_used is not None:
            grow_param_first_dim(self.features_used, new_capacity, fill_value=False)

    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        """Dense save by default. With tracking on, replace the `features` tensor with a sparse
        `(indices, rows_at_indices)` pair plus the original capacity, so untouched rows are not serialized
        and the load path can rebuild the dense buffer without depending on external resize-observer fires."""
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.features_used is None:
            return
        dense_key = prefix + "features"
        if dense_key not in destination:
            return
        dense = destination.pop(dense_key)
        indices = self.features_used.nonzero(as_tuple=False).squeeze(-1).to(torch.long)
        destination[prefix + "features_used_indices"] = indices.detach().cpu()
        # `dense` here is a `nn.Parameter` when keep_vars=True; `.detach()` returns a Tensor regardless.
        destination[prefix + "features_used_rows"] = dense.detach()[indices].cpu()
        # Save the original `features.shape[0]` so the load path can grow the local buffer to that capacity before
        # the in-place `copy_` runs. Without this we would rely on the octree's _load_from_state_dict firing resize
        # observers first; storing it here makes the FeatureBank load self-contained.
        destination[prefix + "features_used_capacity"] = torch.tensor(dense.shape[0], dtype=torch.int64)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Detect a sparse `(indices, rows_at_indices, capacity)` checkpoint and reconstruct the dense
        `features` tensor at the saved capacity before delegating to the standard load. The local buffer is
        grown in place if needed."""
        idx_key = prefix + "features_used_indices"
        row_key = prefix + "features_used_rows"
        cap_key = prefix + "features_used_capacity"
        if idx_key in state_dict and row_key in state_dict:
            indices = state_dict.pop(idx_key).to(torch.long)
            rows = state_dict.pop(row_key)
            if cap_key in state_dict:
                capacity = int(state_dict.pop(cap_key).item())
            else:
                # Older sparse checkpoints (pre-capacity) fall back to the local buffer's current size, which only
                # works when the octree resize observer already grew it. Keep the fallback so existing checkpoints
                # still load.
                capacity = self.features.shape[0]
            if capacity > self.features.shape[0]:
                grow_param_first_dim(self.features, capacity, fill_value=0.0)
            dense = torch.zeros((capacity, self.feature_dim), dtype=self.features.dtype, device=self.features.device)
            if self.cfg.init != "zero":
                # Untouched rows re-drawn from the init distribution. Specific values may diverge from save-time noise
                # but the distribution matches and these rows received no gradient.
                self._init_rows_inplace(dense)
            indices = indices.to(dense.device)
            dense.index_copy_(0, indices, rows.to(dense.device, dtype=dense.dtype))
            state_dict[prefix + "features"] = dense
            if self.features_used is not None:
                if capacity > self.features_used.shape[0]:
                    grow_param_first_dim(self.features_used, capacity, fill_value=False)
                self.features_used.zero_()
                self.features_used[indices.to(self.features_used.device)] = True
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
