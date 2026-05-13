from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from oren.utils.config_abc import ConfigABC


@dataclass
class ResidualNetConfig(ConfigABC):
    mlp_activation: str = "LeakyReLU"  # activation function for the MLP
    implicit_feature_dim: int = 4
    hidden_dims: list[int] = field(default_factory=lambda: [64, 64, 64])
    normalization: Optional[str] = None  # "LayerNorm", "BatchNorm1d", or None
    output_scale: float = 0.1  # scale the output
    output_dim: int = 1  # output dimension of the network, should be 1 for SDF residual


class ResidualNet(nn.Module):
    def __init__(self, cfg: ResidualNetConfig):
        """
        Args:
            cfg: configuration of the network
        """
        super().__init__()
        self.cfg = cfg

        activation = getattr(nn, cfg.mlp_activation)

        layers = []
        in_dim = cfg.implicit_feature_dim
        for hidden_dim in cfg.hidden_dims:
            if cfg.normalization == "LayerNorm":
                layers.append(nn.LayerNorm(in_dim))
            elif cfg.normalization == "BatchNorm1d":
                layers.append(nn.BatchNorm1d(in_dim))
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.cfg.output_dim))
        self.residual_net = nn.Sequential(*layers)

    def forward(self, prior: torch.Tensor, implicit_features: torch.Tensor):
        """
        Args:
            prior: (..., self.cfg.output_dim) prior from the octree
            implicit_features: (..., self.cfg.implicit_feature_dim) implicit features from the octree
        Returns:
            (..., self.cfg.output_dim) residual from the residual network
        """
        x = torch.cat([prior, implicit_features], dim=-1)
        return self.residual_net(x) * self.cfg.output_scale
