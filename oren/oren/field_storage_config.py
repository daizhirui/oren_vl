"""Config dataclasses for FieldStorage and its auxiliary-bank wiring."""

from dataclasses import dataclass, field
from typing import Literal, Optional

from oren.field_fusion import FuserConfig, TrilinearFuserConfig
from oren.implicit_net import ImplicitNetConfig
from oren.utils.config_abc import ConfigABC


@dataclass
class AuxiliaryBankSpec(ConfigABC):
    """Reference to a shared FeatureBank whose per-point features should be concatenated to this field's own features
    before the decoding head.

    Constraint: `name` must reference a *shared* FeatureBank on the parent FieldBank (rejected at construction if it
    points at a private bank).

    Semantics: `detach=True` blocks the gradient flow from this field's losses into the aux bank's parameters — the
    field reads the aux features but does not shape them. Symmetric counterpart to `prior.detach()` in hybrid mode
    (blame attribution).
    """

    name: str = None  # references a FeatureBank on the parent FieldBank
    detach: bool = False  # cut gradient flow into the aux bank


@dataclass
class FieldStorageConfig(ConfigABC):
    """Per-field config consumed by FieldStorage.

    Three modes are supported:
      - "explicit": per-vertex `values (V, D)` trilinear-interpolated to a prior; no MLP head. `pred = prior`.
      - "implicit": per-vertex `features (V, F)` interpolated and fed to a decoding-head MLP; no explicit prior.
        `pred = implicit`.
      - "hybrid": both branches; `pred = prior.detach() + implicit`, where the implicit head consumes
        `cat([prior.detach(), feats], dim=-1)` (concat is done by FieldStorage; ImplicitNet stays single-input).
    """

    name: str = None  # "sdf", "occ", "vl", ... — output key prefix
    output_dim: int = 1  # D
    mode: Literal["explicit", "implicit", "hybrid"] = "hybrid"

    # ---- Explicit branch (mode in {explicit, hybrid}) ----
    explicit_prior_init: float = 0.0  # scalar init for the (V, D) buffer
    gradient_augmentation: bool = False  # Hermite-style GA; only valid when D == 1

    # ---- Pluggable prior gather (modes "explicit" / "hybrid") ----
    # Selects which Fuser drives the prior-gather on `values`. Default keeps the existing trilinear (or Hermite GA
    # when `gradient_augmentation=True`) behavior, so existing SDF / OCC fields require no config change.
    #
    # Compatibility constraints:
    #   - `gradient_augmentation=True` requires a `TrilinearFuserConfig` (it is the only kernel with an analytic
    #     Hermite extension; the prior call passes `grads` to the fuser).
    #   - In hybrid mode (prior + implicit residual), pair non-differentiable scatter-only prior fusers
    #     (e.g. `NearestVertexFuserConfig`) only with scatter-based initialization of `values`; gradient flow into
    #     `values` from the implicit residual loss would otherwise be confined to one vertex per query point.
    prior_fuser_cfg: FuserConfig = field(default_factory=TrilinearFuserConfig)

    # ---- Implicit branch (mode in {implicit, hybrid}) ----
    implicit_feature_dim: int = 4  # F per level
    implicit_feature_level: int = 1  # # of levels (leaf upward) to sample
    implicit_feature_aggregation: Literal["cat", "sum", "max", "mean"] = "cat"
    implicit_net_cfg: ImplicitNetConfig = field(default_factory=ImplicitNetConfig)
    # Pluggable per-level gather on the field's `bank.features`. Reused across all L levels of the implicit branch
    # (the fuser is geometry-only; the level snapshot it consumes carries the per-level vertex_indices and
    # voxel_offsets, so a single fuser instance covers every level). Default `TrilinearFuserConfig` reproduces the
    # original trilinear gather. Swap for `RbfFuserConfig` / `NearestVertexFuserConfig` / etc. to experiment with
    # smoother or sharper interpolation on the feature bank without touching the head MLP.
    implicit_fuser_cfg: FuserConfig = field(default_factory=TrilinearFuserConfig)

    # ---- Sharing ----
    # Default: each field owns its (V, F) bank. Opt in to sharing by naming a FeatureBank declared on the parent
    # FieldBank's `shared_banks`. All fields sharing a bank must agree on `implicit_feature_dim`.
    shared_bank: Optional[str] = None

    # ---- Auxiliary read-only inputs ----
    # Each entry names another *shared* bank whose per-point features are concatenated to this field's own features
    # before the decoding head.
    auxiliary_banks: list[AuxiliaryBankSpec] = field(default_factory=list)

    # ---- Sparse checkpoint storage ----
    # When True, FieldStorage maintains a `values_used` bool buffer and (for private banks) propagates a
    # `features_used` bool buffer on its FeatureBank. Each Fuser.scatter / Fuser.gather call OR-marks the
    # global vertex indices it touched. The state_dict serialization then drops untouched rows from `values`
    # and `bank.features`, storing only (indices, rows_at_indices). On load, the dense buffer is rebuilt at
    # the current `octree.capacity` -- untouched rows are reset to `explicit_prior_init` (for `values`) or
    # the bank's init scheme (for `features`), which is lossless for the default zero-init and lossy in
    # distribution only for non-zero init schemes (untouched rows received no gradient signal, so the
    # specific values are irrelevant). Trades a small bookkeeping cost on every fuser call for ~50%
    # checkpoint shrinkage on scatter modes and significant shrinkage on optimize modes when many vertices
    # are never gathered. Not currently supported with `shared_bank` (use bank-level tracking instead).
    track_used_vertices: bool = False
