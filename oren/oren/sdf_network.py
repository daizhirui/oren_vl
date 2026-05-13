from dataclasses import dataclass, field

import torch
import torch.nn as nn

from oren.octree_config import OctreeConfig
from oren.residual_net import ResidualNet, ResidualNetConfig
from oren.semi_sparse_octree import SemiSparseOctree
from oren.utils.config_abc import ConfigABC
from oren.utils.registry import register_model


@dataclass
class SdfNetworkConfig(ConfigABC):
    octree_cfg: OctreeConfig = field(default_factory=OctreeConfig)
    residual_net_cfg: ResidualNetConfig = field(default_factory=ResidualNetConfig)


@register_model
class SdfNetwork(nn.Module):
    field_prefix: str = "sdf"  # "sdf_prior", "sdf_residual", "sdf_pred", "sdf"

    def __init__(self, cfg: SdfNetworkConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.octree_cfg.enable_sdf, "SdfNetwork requires octree_cfg.enable_sdf=True"
        self.octree: SemiSparseOctree = SemiSparseOctree(cfg.octree_cfg)
        cfg.residual_net_cfg.implicit_feature_dim = (
            # +1 for the prior sdf value
            cfg.octree_cfg.implicit_feature_dim * cfg.octree_cfg.implicit_num_levels + 1
        )
        self.residual: ResidualNet = ResidualNet(cfg.residual_net_cfg)

    def forward(self, points: torch.Tensor, voxel_indices: torch.Tensor = None, prior_only: bool = False):
        """
        Computes the SDF values for the given points.
        Args:
            points: (..., 3) points in world coordinates
            voxel_indices: (...,) optional voxel indices for the points
            prior_only: if true, only return the SDF prior from the octree

        Returns:
            (..., ) voxel indices for the points
            (..., ) SDF prior from the octree
            (..., ) SDF residual from the residual network
            (..., ) final SDF values (prior + residual)
        """
        shape = points.shape
        points = points.view(-1, 3)
        if voxel_indices is not None:
            voxel_indices = voxel_indices.view(-1)
        out = self.octree(points, voxel_indices)
        voxel_indices = out.voxel_indices
        sdf_prior = out.sdf_preds
        implicit_features = out.implicit_features

        if prior_only or implicit_features is None:
            sdf_pred = sdf_prior.detach()
            sdf_residual = torch.zeros_like(sdf_prior)
        else:
            sdf_residual = self.residual(sdf_prior.unsqueeze(-1).detach(), implicit_features).squeeze(-1)
            sdf_pred = sdf_prior.detach() + sdf_residual

        voxel_indices = voxel_indices.view(shape[:-1])
        sdf_prior = sdf_prior.view(shape[:-1])
        sdf_pred = sdf_pred.view(shape[:-1])
        sdf_residual = sdf_residual.view(shape[:-1])

        return voxel_indices, sdf_prior, sdf_residual, sdf_pred
