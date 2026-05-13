from dataclasses import dataclass, field

import torch
import torch.nn as nn

from oren.octree_config import OctreeConfig
from oren.residual_net import ResidualNet, ResidualNetConfig
from oren.semi_sparse_octree import SemiSparseOctree
from oren.utils.config_abc import ConfigABC
from oren.utils.registry import register_model


@dataclass
class OccNetworkConfig(ConfigABC):
    octree_cfg: OctreeConfig = field(default_factory=OctreeConfig)
    residual_net_cfg: ResidualNetConfig = field(default_factory=ResidualNetConfig)


@register_model
class OccNetwork(nn.Module):
    field_prefix: str = "occ"  # "occ_prior", "occ_residual", "occ_pred", "occ"

    def __init__(self, cfg: OccNetworkConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.octree_cfg.enable_occupancy or cfg.octree_cfg.enable_implicit, (
            "OccNetwork requires either enable_occupancy=True (per-vertex prior) "
            "or enable_implicit=True (implicit-feature MLP): both off is unsupervised"
        )
        self.octree: SemiSparseOctree = SemiSparseOctree(cfg.octree_cfg)
        cfg.residual_net_cfg.implicit_feature_dim = (
            cfg.octree_cfg.implicit_feature_dim * cfg.octree_cfg.implicit_num_levels
        )
        self.residual: ResidualNet = ResidualNet(cfg.residual_net_cfg)

    def forward(self, points: torch.Tensor, voxel_indices: torch.Tensor = None, prior_only: bool = False):
        """
        Computes occupancy logits for the given points.

        Three modes depending on octree_cfg:
          1. enable_occupancy=T, enable_implicit=T (default): occ_prior from octree, residual from MLP, occ = prior+residual.
          2. enable_occupancy=T, enable_implicit=F: occ_prior from octree, no residual, occ = prior.
          3. enable_occupancy=F, enable_implicit=T: occ_prior=zeros, residual = MLP(0, implicit_features), occ = residual.

        Returns:
            (..., ) voxel indices for the points
            (..., ) occupancy logit prior (zeros when enable_occupancy=False)
            (..., ) occupancy logit residual (None if prior_only or enable_implicit=False)
            (..., ) final occupancy logits (prior + residual)
        """
        shape = points.shape
        points = points.view(-1, 3)
        if voxel_indices is not None:
            voxel_indices = voxel_indices.view(-1)
        out = self.octree(points, voxel_indices)
        voxel_indices = out.voxel_indices
        occ_prior = out.occ_preds  # None when enable_occupancy=False
        implicit_features = out.implicit_features  # None when enable_implicit=False

        # Build a zero prior tensor when occupancy storage is disabled so downstream
        # losses see a consistent tensor shape and the residual MLP can still receive a
        # "prior=0" input on the implicit path.
        if occ_prior is None:
            occ_prior = torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)

        if prior_only or implicit_features is None:
            occ_pred = occ_prior.detach()
            occ_residual = torch.zeros_like(occ_prior)
        else:
            occ_residual = self.residual(occ_prior.unsqueeze(-1).detach(), implicit_features).squeeze(-1)
            occ_pred = occ_prior.detach() + occ_residual

        voxel_indices = voxel_indices.view(shape[:-1])
        occ_prior = occ_prior.view(shape[:-1])
        occ_pred = occ_pred.view(shape[:-1])
        occ_residual = occ_residual.view(shape[:-1])

        return voxel_indices, occ_prior, occ_residual, occ_pred
