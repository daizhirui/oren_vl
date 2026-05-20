from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from oren.utils.registry import register_criterion
from oren.utils.config_abc import ConfigABC


@dataclass
class VlCriterionConfig(ConfigABC):
    # Weight for the VL feature loss term. 0 disables the term entirely.
    vl_loss_weight: float = 1.0
    vl_loss_prior_weight: float = 1.0

    # Loss type
    vl_loss_type: str = "l2"  # "l1", "l2", "cosine"


class CosineSimilarityLoss(nn.Module):

    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Cosine similarity in [-1, 1], we want to maximize it, so loss is 1 - cosine_similarity.
        cos_sim = F.cosine_similarity(input, target, dim=-1)
        if self.reduction == "mean":
            return 1.0 - cos_sim.mean()
        elif self.reduction == "sum":
            return cos_sim.numel() - cos_sim.sum()
        else:
            return 1.0 - cos_sim


@register_criterion
class VlCriterion(nn.Module):
    """Criterion for learning the VL features."""

    needs_grad: bool = False

    def __init__(self, cfg: VlCriterionConfig):
        """Construct the criterion for the VL features.

        Args:
            cfg: criterion config, carrying the loss weights and type.
        """
        super().__init__()
        self.cfg = cfg

        self.loss_fn = None
        if cfg.vl_loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction="mean")
        elif cfg.vl_loss_type == "l2":
            self.loss_fn = nn.MSELoss(reduction="mean")
        elif cfg.vl_loss_type == "cosine":
            self.loss_fn = CosineSimilarityLoss(reduction="mean")
        else:
            raise ValueError(f"Unsupported VL loss type: {cfg.vl_loss_type}")

    def forward(
        self,
        pred_vl: torch.Tensor,
        pred_prior: Optional[torch.Tensor],
        gt_vl: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the VL feature loss.

        Args:
            pred_vl: (B, F) predicted VL features.
            pred_prior: (B, F) predicted prior features.
            gt_vl: (B, F) ground truth VL features.

        Returns:
            Scalar loss and a dictionary of loss components.
        """
        loss_dict = {}

        loss = torch.tensor(0.0, device=gt_vl.device)
        if self.cfg.vl_loss_weight > 0.0:
            vl_loss = self.loss_fn(pred_vl, gt_vl).mean()
            loss += vl_loss * self.cfg.vl_loss_weight
            loss_dict["vl_loss"] = vl_loss.item()
        if self.cfg.vl_loss_prior_weight > 0.0 and pred_prior is not None:
            prior_loss = self.loss_fn(pred_prior, gt_vl).mean()
            loss += prior_loss * self.cfg.vl_loss_prior_weight
            loss_dict["vl_prior_loss"] = prior_loss.item()

        loss_dict["total_loss"] = loss.item()
        return loss, loss_dict
