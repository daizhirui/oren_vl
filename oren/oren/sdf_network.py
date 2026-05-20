"""SdfNetwork -- thin adapter around a single-field FieldBank for SDF training.

The class binds the YAML's `model.cfg_identifier` to a configuration that selects (octree geometry, one FieldStorage
with name="sdf").
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from oren.field_bank import FieldBank, FieldStorage
from oren.field_output import FieldOutput
from oren.field_storage_config import FieldStorageConfig
from oren.octree_config import OctreeConfig
from oren.semi_sparse_octree import SemiSparseOctree
from oren.utils.config_abc import ConfigABC
from oren.utils.registry import register_model


@dataclass
class SdfNetworkConfig(ConfigABC):
    octree_cfg: OctreeConfig = field(default_factory=OctreeConfig)
    # FieldStorageConfig for the single SDF field. Default name="sdf" matches `SdfNetwork.field_prefix` and the
    # trainer/criterion output-key convention; other knobs (mode, dims, sharing) inherit FieldStorageConfig's defaults.
    field: FieldStorageConfig = field(default_factory=lambda: FieldStorageConfig(name="sdf"))


@register_model
class SdfNetwork(nn.Module):
    field_prefix: str = "sdf"  # "sdf_prior", "sdf_residual", "sdf_pred", "sdf"

    def __init__(self, cfg: SdfNetworkConfig):
        """Build the SdfNetwork adapter: construct the octree and a single-field `FieldBank` from `cfg`.

        Args:
            cfg: network configuration carrying the octree config and the SDF field config.
        """
        super().__init__()
        self.cfg = cfg
        self.octree: SemiSparseOctree = SemiSparseOctree(cfg.octree_cfg)
        self.field_bank: FieldBank = FieldBank(octree=self.octree, fields=[cfg.field], shared_banks=[])

    @property
    def sdf_field(self) -> FieldStorage:
        """The single :class:`FieldStorage` instance this network wraps. Mirrors :attr:`VlNetwork.vl_field` so trainer
        and ROS-side callers that need direct access to `values` / `prior_fuser` (e.g. the scatter-mode fast path)
        share one shape.
        """
        return self.field_bank.fields[self.cfg.field.name]

    def forward(
        self,
        points: torch.Tensor,
        voxel_indices: torch.Tensor = None,
        prior_only: bool = False,
    ) -> FieldOutput:
        """Run the single-field FieldBank and return its `FieldOutput` for the SDF field.

        Leading dims of `points` are preserved on the returned tensors (FieldBank flattens internally and FieldOutput
        reshapes back via its `__post_init__`); the trailing `D = cfg.field.output_dim` dim is kept (callers
        `squeeze(-1)` for the scalar SDF case). `prior` and `implicit` are `None` when the field's mode doesn't have
        that branch.

        Args:
            points: (..., 3) query points in world coordinates.
            voxel_indices: optional (...,) precomputed leaf voxel indices forwarded to the field bank.
            prior_only: if True, skip the implicit branch; `pred` is then equal to `prior`.
        """
        return self.field_bank(points, voxel_indices=voxel_indices, prior_only=prior_only)[self.cfg.field.name]

    @torch.no_grad()
    def scatter_update(self, points: torch.Tensor, sdf_values: torch.Tensor, level: int = 1) -> None:
        """Ingest per-frame SDF labels into the SDF field via the prior fuser's scatter.

        Mirrors :meth:`VlNetwork.scatter_update`: the fuser owns the running-state buffers
        (nearest-vertex `counts`, kernel `weight_sum`) so multiple calls compose into a cumulative weighted average.
        Direct-update mode only -- the learnable mode is exercised by the trainer via :meth:`forward` and an external
        optimizer step on `sdf_field.values`.

        Args:
            points: `(N, 3)` world-space points whose SDF labels should be folded into the field. Higher-rank inputs
                are accepted and flattened.
            sdf_values: per-point SDF labels. Accepted shapes: `(N,)` (scalar) or `(N, 1)` -- matched against
                `cfg.field.output_dim` (which must be 1 for SDF).
            level: octree level at which to perform the scatter; defaults to leaf (1).
        """
        fuser = self.sdf_field.prior_fuser
        assert fuser is not None, "SdfNetwork.scatter_update requires a prior fuser; check cfg.field.prior_fuser_cfg"
        assert self.cfg.field.output_dim == 1, (
            f"SdfNetwork.scatter_update expects a scalar SDF field (output_dim=1); got "
            f"output_dim={self.cfg.field.output_dim}"
        )

        # Flatten arbitrary leading dims to (N, 3) / (N, 1). The fuser's `scatter` expects feats as (N, D).
        pts_flat = points.reshape(-1, 3)
        sdf_flat = sdf_values.reshape(-1, 1)
        assert (
            sdf_flat.shape[0] == pts_flat.shape[0]
        ), f"SdfNetwork.scatter_update: sdf_values has {sdf_flat.shape[0]} entries but points has {pts_flat.shape[0]}"

        level_geom = self.octree.query(pts_flat).at_level(level)
        fuser.scatter(level_geom, self.sdf_field.values, sdf_flat)
