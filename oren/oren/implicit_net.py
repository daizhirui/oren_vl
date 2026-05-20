"""Single-input MLP head used by FieldStorage in `implicit` and `hybrid` modes.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from oren.utils.config_abc import ConfigABC


@dataclass
class ImplicitNetConfig(ConfigABC):
    mlp_activation: str = "LeakyReLU"
    # Input dim of the MLP. After concatenation done at the call site this is:
    #   base = L*F (when implicit aggregation == "cat")  or  F (sum/mean/max)
    #   base += sum(aux_bank.feature_dim for aux in auxiliary_banks)
    #   input_dim = D + base   if mode == "hybrid"   else base
    # FieldStorage overwrites this at construction time from the assembled head-input dim, so the YAML value is
    # informational only.
    input_dim: int = 4
    hidden_dims: list[int] = field(default_factory=lambda: [64, 64, 64])
    normalization: Optional[str] = None
    output_scale: float = 0.1
    output_dim: int = 1


class ImplicitNet(nn.Module):
    """Single-input MLP. `forward(x)` runs `x` through the MLP and scales the output.

    The head intentionally does NOT concatenate anything internally. When used in hybrid mode it expects to receive
    `cat([prior.detach(), feats], dim=-1)` pre-assembled by the caller; in implicit mode it receives the raw (possibly
    aux-augmented) `feats`.
    """

    def __init__(self, cfg: ImplicitNetConfig):
        """Build the MLP stack from `cfg` (hidden dims, activation, optional normalization, output dim).

        Args:
            cfg: MLP configuration carrying input dim, hidden dims, activation name, optional normalization, and
                output dim and scale.
        """
        super().__init__()
        self.cfg = cfg

        activation = getattr(nn, cfg.mlp_activation)

        layers = []
        in_dim = cfg.input_dim
        for hidden_dim in cfg.hidden_dims:
            if cfg.normalization == "LayerNorm":
                layers.append(nn.LayerNorm(in_dim))
            elif cfg.normalization == "BatchNorm1d":
                layers.append(nn.BatchNorm1d(in_dim))
            else:
                assert cfg.normalization is None, f"Unsupported normalization: {cfg.normalization}"
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.cfg.output_dim))
        # Attribute name preserved across the rename so pre-rename checkpoints (state_dict keys like
        # `residual.residual_net.0.weight`) keep loading via the SdfNetwork/OccNetwork wrappers, whose `self.residual`
        # attribute is likewise preserved during phase 0.
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP on a pre-assembled input tensor.

        Args:
            x: (..., input_dim) — already concatenated upstream.

        Returns:
            (..., output_dim) scaled by `cfg.output_scale`.
        """
        return self.net(x) * self.cfg.output_scale

