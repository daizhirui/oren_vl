"""OccNetwork — thin adapter around a single-field FieldBank for occupancy training.

Mirrors `SdfNetwork`. The class binds the YAML's `model.cfg_identifier` to a configuration that selects (octree
geometry, one FieldStorage with name="occ"). Internally the wrapper holds a `FieldBank` with exactly one field;
`forward(...)` returns the field's `FieldOutput` directly.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from oren.field_bank import FieldBank
from oren.field_output import FieldOutput
from oren.field_storage_config import FieldStorageConfig
from oren.octree_config import OctreeConfig
from oren.semi_sparse_octree import SemiSparseOctree
from oren.utils.config_abc import ConfigABC
from oren.utils.registry import register_model


@dataclass
class OccNetworkConfig(ConfigABC):
    octree_cfg: OctreeConfig = field(default_factory=OctreeConfig)
    # FieldStorageConfig for the single occupancy field. Default name="occ" matches `OccNetwork.field_prefix` and the
    # trainer/criterion output-key convention; other knobs (mode, dims, sharing) inherit FieldStorageConfig's defaults.
    field: FieldStorageConfig = field(default_factory=lambda: FieldStorageConfig(name="occ"))


@register_model
class OccNetwork(nn.Module):
    field_prefix: str = "occ"  # "occ_prior", "occ_residual", "occ_pred", "occ"

    def __init__(self, cfg: OccNetworkConfig):
        """Build the OccNetwork adapter: construct the octree and a single-field `FieldBank` from `cfg`.

        Args:
            cfg: network configuration carrying the octree config and the occupancy field config.
        """
        super().__init__()
        self.cfg = cfg
        self.octree: SemiSparseOctree = SemiSparseOctree(cfg.octree_cfg)
        self.field_bank: FieldBank = FieldBank(octree=self.octree, fields=[cfg.field], shared_banks=[])

    def forward(
        self,
        points: torch.Tensor,
        voxel_indices: torch.Tensor = None,
        prior_only: bool = False,
    ) -> FieldOutput:
        """Run the single-field FieldBank and return its `FieldOutput` for the occupancy field.

        Leading dims of `points` are preserved on the returned tensors (FieldBank flattens internally and FieldOutput
        reshapes back via its `__post_init__`); the trailing `D = cfg.field.output_dim` dim is kept (callers
        `squeeze(-1)` for the scalar occupancy case). `prior` and `implicit` are `None` when the field's mode doesn't
        have that branch.

        Args:
            points: (..., 3) query points in world coordinates.
            voxel_indices: optional (...,) precomputed leaf voxel indices forwarded to the field bank.
            prior_only: if True, skip the implicit branch; `pred` is then equal to `prior`.
        """
        return self.field_bank(points, voxel_indices=voxel_indices, prior_only=prior_only)[self.cfg.field.name]
