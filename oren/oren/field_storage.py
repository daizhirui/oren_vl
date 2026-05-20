"""Per-field state and forward for one spatial field over a SemiSparseOctree."""

from typing import Optional

import torch
import torch.nn as nn

from oren.feature_bank import FeatureBank, FeatureBankConfig
from oren.field_fusion import TrilinearFuser, build_fuser
from oren.field_output import FieldOutput, OctreeGeometry
from oren.field_storage_config import FieldStorageConfig
from oren.implicit_net import ImplicitNet
from oren.semi_sparse_octree import SemiSparseOctree
from oren.utils.param_resize import grow_param_first_dim


class FieldStorage(nn.Module):
    """Per-vertex state + per-point forward for one spatial field.

    Args:
        cfg: shape and mode of this field.
        octree: geometry provider. Held as a back-reference; deliberately *not* registered as a submodule (the octree
            is shared across many FieldStorage instances and we don't want it duplicated in `state_dict()`).
        bank: when `cfg.shared_bank` is set, the parent `FieldBank` passes in the shared `FeatureBank`. When None, each
            field owns a private bank (allocated here if mode requires it). Phase 1's explicit mode doesn't touch
            banks.
    """

    def __init__(
        self,
        cfg: FieldStorageConfig,
        octree: SemiSparseOctree,
        bank: Optional[FeatureBank] = None,
        aux_extra_dim: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        # Aux-bank features are gathered upstream by the parent FieldBank and passed to `forward` via `extra_feats`.
        # We size the decoding head for the full assembled input here (FieldBank computes the aux contribution;
        # FieldStorage knows its own bank's contribution).
        self._aux_extra_dim = aux_extra_dim
        # Held as a plain attribute (not a submodule) so the octree's params/buffers don't appear under this field's
        # state_dict prefix.
        object.__setattr__(self, "octree", octree)
        self.octree: SemiSparseOctree  # for pyright; octree already stored above
        # `bank` may be shared (owned by the parent FieldBank) or private (allocated in the implicit branch below).
        # Shared banks are held as *plain attributes* — bypassing nn.Module's submodule registration — because
        # FieldBank already exposes them under `banks.<name>.features`; registering them again here would duplicate
        # those parameter keys under `<field>.bank.features` in state_dict. Private banks (when allocated below) are
        # assigned via normal `self.bank = ...` so they ARE registered as submodules of this field.
        if bank is None:
            self.bank: Optional[FeatureBank] = None
        else:
            object.__setattr__(self, "bank", bank)
            self.bank: Optional[FeatureBank]  # for pyright; bank already stored above

        # Initial per-vertex-parameter capacity comes from the octree, NOT `octree.cfg.init_voxel_num`.
        # `octree.capacity` is the pow-2-rounded vertex high-water mark currently announced to all resize observers;
        # sizing here matches that exactly so the catch-up fire of `register_resize_observer` below is a no-op at
        # construction time.
        init_capacity = octree.capacity

        # ---- Explicit branch parameters ----
        # Allocated for mode in {"explicit", "hybrid"}. Init values come from `cfg.explicit_prior_init`.
        if cfg.mode in ("explicit", "hybrid"):
            self.values = nn.Parameter(
                torch.full(
                    (init_capacity, cfg.output_dim),
                    cfg.explicit_prior_init,
                    dtype=torch.float32,
                ),
            )
            if cfg.gradient_augmentation:
                assert cfg.output_dim == 1, (
                    f"gradient_augmentation is only valid for scalar fields (output_dim=1); "
                    f"got output_dim={cfg.output_dim}"
                )
                self.grads = nn.Parameter(
                    torch.zeros((init_capacity, 3), dtype=torch.float32),
                )
            else:
                self.grads = None
        else:
            self.values = None
            self.grads = None

        # ---- Implicit branch: own private FeatureBank when not sharing ----
        # If the field's mode uses an implicit branch and no shared bank was passed in by the parent FieldBank,
        # allocate a private bank here. Private banks are registered as submodules of the field, so their parameters
        # land in the field's state_dict.
        if cfg.mode in ("implicit", "hybrid"):
            agg_ops = {"cat", "sum", "mean", "max"}
            assert cfg.implicit_feature_aggregation in agg_ops, (
                f"implicit_feature_aggregation must be one of {agg_ops}; " f"got {cfg.implicit_feature_aggregation!r}"
            )
            assert (
                cfg.implicit_feature_level >= 1
            ), f"implicit_feature_level must be >= 1; got {cfg.implicit_feature_level}"
            if bank is None:
                # Private banks never carry a trunk — the field's own `implicit_net` is the only head. The bank's name
                # reuses the field's name so any state_dict prefix collision is impossible (private banks live under
                # the field's submodule scope, not under FieldBank.banks).
                private_bank_cfg = FeatureBankConfig(
                    name=cfg.name,
                    feature_dim=cfg.implicit_feature_dim,
                    shared_net_cfg=None,
                )
                private_bank = FeatureBank(private_bank_cfg, init_capacity=init_capacity)
                # Register the private bank under `self.bank` so its parameters appear once in state_dict at
                # `<field>.bank.features`.
                self.bank = private_bank
                self._own_bank = True
                octree.register_resize_observer(private_bank.grow_to)
            else:
                self._own_bank = False
                # External shared bank — `self.bank` was set above via object.__setattr__ as a plain attribute (not a
                # submodule), so the shared bank's params don't get duplicated under this field's state_dict scope.
                # FieldBank also owns the resize observer registration; we don't add another one here.

            # Decoding head input dim is the size of the tensor FieldStorage hands the head. Components:
            #   - own bank, after multi-level aggregation:
            #         L*F (cat) or F (sum/mean/max)
            #   - aux banks contributed by the parent FieldBank (via `aux_extra_dim` here; FieldBank knows the totals
            #     because it owns the bank registry)
            #   - the prior, in hybrid mode (concatenated by FieldStorage itself at forward time)
            base_dim = self._post_aggregation_dim(
                cfg.implicit_feature_dim,
                cfg.implicit_feature_level,
                cfg.implicit_feature_aggregation,
            )
            head_in_dim = base_dim + aux_extra_dim + (cfg.output_dim if cfg.mode == "hybrid" else 0)

            cfg.implicit_net_cfg.input_dim = head_in_dim
            self.implicit_net = ImplicitNet(cfg.implicit_net_cfg)
        else:
            self._own_bank = False
            self.implicit_net = None

        # ---- Resize observer for explicit branch parameters ----
        # Grow per-vertex tensors in lockstep with octree.sso.num_vertices. The lambda captures `self` weakly via the
        # bound method; the field unregisters in `__del__` to avoid leaks on temporary FieldStorage instances (e.g.
        # constructed in tests).
        octree.register_resize_observer(self._on_octree_resize)

        # ---- Pluggable prior fuser (modes "explicit" / "hybrid") ----
        # Built AFTER `_on_octree_resize` is registered so values/grads grow before the fuser's own state buffers on
        # subsequent resizes. Stored as a submodule so trainable fuser variants (Phase 2 attention) have their
        # parameters discovered via FieldBank's parameter enumeration.
        if cfg.mode in ("explicit", "hybrid"):
            self.prior_fuser = build_fuser(cfg.prior_fuser_cfg, octree)
            if cfg.gradient_augmentation:
                assert isinstance(self.prior_fuser, TrilinearFuser), (
                    "gradient_augmentation=True requires prior_fuser_cfg=TrilinearFuserConfig (the only kernel with "
                    f"an analytic Hermite extension); got {type(self.prior_fuser).__name__}"
                )
        else:
            self.prior_fuser = None

        # ---- Pluggable implicit-branch fuser (modes "implicit" / "hybrid") ----
        # Reads `bank.features` at each implicit level. One fuser instance covers all L levels: the fuser's state
        # buffers are sized by `octree.capacity` (covers every level's vertex IDs) and the per-call `LevelGeometry`
        # supplies the level-specific `vertex_indices` / `voxel_offsets`. Built after the bank's `grow_to` and
        # `_on_octree_resize` are registered so the bank features and the fuser's state buffers grow in a sane
        # order on subsequent resizes.
        if cfg.mode in ("implicit", "hybrid"):
            self.implicit_fuser = build_fuser(cfg.implicit_fuser_cfg, octree)
        else:
            self.implicit_fuser = None

    @property
    def own_bank(self) -> Optional[FeatureBank]:
        """The bank this field owns, or None if borrowing a shared bank.

        FieldBank uses this to enumerate per-vertex parameters whose resize + optimizer-state migration this field is
        responsible for. Shared banks are enumerated by FieldBank directly via `self.banks`, so returning None here
        keeps them from being yielded twice.
        """
        return self.bank if self._own_bank else None

    def _on_octree_resize(self, new_capacity: int) -> None:
        """Octree resize callback — grow `values` and (optional) `grads`.

        Optimizer *state* migration is handled by `FieldBank.attach_optimizer` (registered after this callback on the
        same octree).
        """
        if self.values is not None:
            grow_param_first_dim(
                self.values,
                new_capacity,
                fill_value=self.cfg.explicit_prior_init,
            )
        if self.grads is not None:
            grow_param_first_dim(self.grads, new_capacity, fill_value=0.0)

    # ------------------------------------------------------------------
    # Branch implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _post_aggregation_dim(feature_dim: int, num_levels: int, aggregation: str) -> int:
        """Channel dim of `_implicit_feature` output before any head-side concat."""
        if aggregation == "cat":
            return feature_dim * num_levels
        # sum / mean / max all reduce across levels  dim stays at F.
        return feature_dim

    def _implicit_feature(self, geom: OctreeGeometry) -> torch.Tensor:
        """Gather the bank's per-vertex features at every level in [1, L] via `implicit_fuser`, aggregate across
        levels, and (optionally) push through the bank's shared trunk. Returns the per-point feature tensor
        consumed by the decoding head.

        Always goes through `geom.at_level(k)` — never `octree.query(...)` directly — so the per-call cache mediates
        duplicate level requests across fields on the same FieldBank.
        """
        bank = self.bank
        assert bank is not None, "_implicit_feature requires a bank (private or shared)"
        assert self.implicit_fuser is not None  # invariant when mode in {implicit, hybrid}
        L = self.cfg.implicit_feature_level
        agg = self.cfg.implicit_feature_aggregation

        per_level: list[torch.Tensor] = []
        for level in range(1, L + 1):
            level_geom = geom.at_level(level)
            per_level.append(self.implicit_fuser.gather(level_geom, bank.features))

        if L == 1:
            feats = per_level[0]
        elif agg == "cat":
            feats = torch.cat(per_level, dim=-1)
        elif agg == "sum":
            feats = torch.stack(per_level, dim=0).sum(dim=0)
        elif agg == "mean":
            feats = torch.stack(per_level, dim=0).mean(dim=0)
        elif agg == "max":
            feats = torch.stack(per_level, dim=0).max(dim=0).values
        else:  # pragma: no cover — guarded by ctor assertion
            raise ValueError(f"unknown aggregation {agg!r}")

        if bank.shared_net is not None:
            feats = bank.shared_net(feats)
        return feats

    def _explicit(self, geom: OctreeGeometry) -> torch.Tensor:
        """Delegate the prior gather to the field's configured Fuser.

        Returns `(N, D)` where `D = cfg.output_dim`. The default `TrilinearFuserConfig` reproduces the original
        trilinear gather; with `gradient_augmentation=True` the gather is the Hermite GA path inside
        `TrilinearFuser.gather` (passes `grads`). Other fusers (NearestVertex / InverseDistance / Rbf / OU) plug in
        through this single call site without touching the field's forward.
        """
        assert self.prior_fuser is not None  # invariant when mode in {explicit, hybrid}
        level_geom = geom.at_level(1)
        if isinstance(self.prior_fuser, TrilinearFuser):
            return self.prior_fuser.gather(
                level_geom,
                self.values,
                grads=self.grads if self.cfg.gradient_augmentation else None,
            )
        return self.prior_fuser.gather(level_geom, self.values)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        geom: OctreeGeometry,
        prior_only: bool = False,
        extra_feats: Optional[torch.Tensor] = None,
    ) -> FieldOutput:
        """Run this field's forward over the cached geometry.

        Args:
            geom: per-call octree geometry cache built by `octree.query(points)`.
            prior_only: if True, skip the implicit branch and return prior as the prediction.
            extra_feats: optional (n_points, aux_extra_dim) auxiliary-bank features pre-gathered by the parent
                FieldBank, concatenated to this field's own features before the decoding head.

        Returns:
            FieldOutput with `voxel_indices`, optional `prior`, optional `implicit`, and the final `pred`
            populated per this field's mode.
        """
        leaf = geom.at_level(1)
        mode = self.cfg.mode

        prior: Optional[torch.Tensor] = None
        if mode in ("explicit", "hybrid"):
            prior = self._explicit(geom)

        if prior_only or mode == "explicit":
            return FieldOutput(
                voxel_indices=leaf.voxel_indices,
                prior=prior,
                implicit=None,
                pred=prior,
                batch_shape=geom.batch_shape,
            )

        # mode in {implicit, hybrid} from here on.
        feats = self._implicit_feature(geom)
        if extra_feats is not None:
            feats = torch.cat([feats, extra_feats], dim=-1)

        if mode == "implicit":
            implicit = self.implicit_net(feats)
            pred = implicit
            return FieldOutput(
                voxel_indices=leaf.voxel_indices,
                prior=None,
                implicit=implicit,
                pred=pred,
                batch_shape=geom.batch_shape,
            )

        # hybrid: concat the (detached) prior to the head input.
        # The +D term in the head's `input_dim` accounts for this.
        assert prior is not None
        x = torch.cat([prior.detach(), feats], dim=-1)
        implicit = self.implicit_net(x)
        pred = prior.detach() + implicit
        return FieldOutput(
            voxel_indices=leaf.voxel_indices,
            prior=prior,
            implicit=implicit,
            pred=pred,
            batch_shape=geom.batch_shape,
        )
