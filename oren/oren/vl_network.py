"""VlNetwork -- thin adapter around a single-field FieldBank for vision-language features.

Two entry points beyond what `SdfNetwork` / `OccNetwork` expose:

    - :meth:`forward` (`points`): gather VL features at query points. Returns the field's :class:`FieldOutput`; for
        explicit-mode fields the `pred` is the trilinear (or kernel-weighted) gather of `values`.
    - :meth:`update` (`points`, `feats`): ingest per-frame features. Delegates to the field's prior fuser's
        `scatter` so the running-state buffers (e.g. nearest-vertex `counts`, kernel `weight_sum`) accumulate
        across frames.

Default config matches the demo: `output_dim=1024`, `mode="explicit"`, `gradient_augmentation=False`,
`prior_fuser_cfg=NearestVertexFuserConfig(mode="running_average")`. Switch `prior_fuser_cfg` for kernel-weighted
fusion (`TrilinearFuserConfig` / `RbfFuserConfig` / ...).
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from oren.field_bank import FieldBank, FieldStorage
from oren.field_fusion import NearestVertexFuserConfig
from oren.field_output import FieldOutput
from oren.field_storage_config import FieldStorageConfig
from oren.octree_config import OctreeConfig
from oren.semi_sparse_octree import SemiSparseOctree
from oren.utils.config_abc import ConfigABC
from oren.utils.registry import register_model


def _default_vl_field_cfg() -> FieldStorageConfig:
    """Default VL field config: 1024-dim CLIP-shaped explicit field with running-average nearest-vertex fusion.

    Generalises the demo's scatter-then-store flow: the explicit branch holds the running mean of per-pixel CLIP
    features assigned to each leaf-voxel nearest vertex. Swap `prior_fuser_cfg` to use a kernel-weighted fuser
    (e.g. trilinear, RBF) without touching the rest of the wiring.
    """
    return FieldStorageConfig(
        name="vl",
        output_dim=1024,
        mode="explicit",
        gradient_augmentation=False,
        explicit_prior_init=0.0,
        prior_fuser_cfg=NearestVertexFuserConfig(mode="running_average"),
    )


@dataclass
class VlNetworkConfig(ConfigABC):
    """Config for :class:`VlNetwork`."""

    octree_cfg: OctreeConfig = field(default_factory=OctreeConfig)
    field: FieldStorageConfig = field(default_factory=_default_vl_field_cfg)


@register_model
class VlNetwork(nn.Module):
    """One-octree, one-field adapter for vision-language fusion."""

    field_prefix: str = "vl"  # "vl_prior", "vl_residual", "vl_pred", "vl"

    def __init__(self, cfg: VlNetworkConfig):
        """Build the VL network: construct the octree and a single-field `FieldBank` from `cfg`.

        Args:
            cfg: network configuration carrying the octree config and the VL field config.
        """
        super().__init__()

        self.cfg = cfg
        self.octree: SemiSparseOctree = SemiSparseOctree(cfg.octree_cfg)
        self.field_bank: FieldBank = FieldBank(octree=self.octree, fields=[cfg.field], shared_banks=[])

    @property
    def vl_field(self) -> FieldStorage:
        """The single :class:`FieldStorage` instance this network wraps. Convenience accessor for callers that need
        direct access to `values` / `prior_fuser` (e.g. the offline demo's mid-training save path).
        """
        return self.field_bank.fields[self.cfg.field.name]

    def forward(
        self,
        points: torch.Tensor,
        voxel_indices: torch.Tensor = None,
        prior_only: bool = False,
    ) -> FieldOutput:
        """Run the single-field FieldBank and return its `FieldOutput` for the VL field.

        Args:
            points: `(..., 3)` query points in world coordinates.
            voxel_indices: optional `(...,)` precomputed leaf voxel indices forwarded to the field bank.
            prior_only: if True, skip the implicit branch; `pred` is then equal to `prior`.
        """
        return self.field_bank(points, voxel_indices=voxel_indices, prior_only=prior_only)[self.cfg.field.name]

    @torch.no_grad()
    def scatter_update(self, points: torch.Tensor, feats: torch.Tensor, level: int = 1) -> None:
        """Ingest per-frame features into the VL field via the fuser's scatter.

        Used by the ROS mapping node and the offline demo to incorporate new observations into the field. The fuser
        owns the running-state buffers (nearest-vertex `counts`, kernel `weight_sum`) that compose multiple
        calls into a cumulative weighted average. Direct-update mode only -- the learnable mode is exercised by the
        trainer via :meth:`forward` and an external optimizer step on `vl_field.values`.

        Args:
            points: `(N, 3)` world-space points whose features should be folded into the field.
            feats: `(N, C)` per-point feature vectors (`C == cfg.field.output_dim`).
            level: octree level at which to perform the scatter; defaults to leaf (1).
        """
        fuser = self.vl_field.prior_fuser
        assert fuser is not None, "VlNetwork.scatter_update requires a prior fuser; check cfg.field.prior_fuser_cfg"
        assert feats.shape[-1] == self.cfg.field.output_dim, (
            f"VlNetwork.scatter_update: feats trailing dim {feats.shape[-1]} mismatches field output_dim "
            f"{self.cfg.field.output_dim}"
        )

        level_geom = self.octree.query(points.view(-1, 3)).at_level(level)
        fuser.scatter(level_geom, self.vl_field.values, feats.view(-1, feats.shape[-1]))
