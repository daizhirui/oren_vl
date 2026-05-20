from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from oren.utils.config_abc import ConfigABC
from oren.utils.registry import register_criterion


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp(min=1.0)


@dataclass
class OccCriterionConfig(ConfigABC):
    # Target value (in probability space) for the on-surface sample. 1.0 treats the surface as the inside boundary;
    # 0.5 treats it as the decision boundary.
    surface_target: float = 1.0

    # Surface (occupied, target=surface_target) - single sample at depth.
    surface_loss_weight: float = 1.0
    surface_loss_prior_weight: float = 1.0

    # Free space (target=0) - stratified samples + the negative-perturbation half of perturbed samples.
    free_loss_weight: float = 1.0
    free_loss_prior_weight: float = 1.0

    # Behind-surface samples (positive-perturbation half of perturbed samples). Ambiguous on thin objects, so the loss
    # form is configurable.
    occ_loss_weight: float = 1.0
    occ_loss_prior_weight: float = 1.0
    # "bce":   BCE(logit, 1) - trust the label.
    # "hinge": relu(margin - logit) - only push toward occupied; never punish a confident occupied prediction.
    # "off":   no loss on these samples.
    occ_loss_mode: Literal["bce", "hinge", "off"] = "hinge"
    # Margin in logit space for the hinge mode. logit >= margin => zero loss.
    # 1.0 -> p=0.73, 2.0 -> p=0.88.
    occ_loss_hinge_margin: float = 1.0

    # Field smoothness regularizer. For each sample point x the trainer also evaluates pred_occ at x + eps (random eps
    # in [-smoothness_eps, +smoothness_eps]^3) and we penalize (pred_occ(x) - pred_occ(x+eps))^2. 0 disables the term
    # entirely.
    smoothness_weight: float = 0.0
    smoothness_eps: float = 0.005


@register_criterion
class OccCriterion(nn.Module):
    """Binary occupancy criterion driven entirely by depth ray-casting (no SDF GT required).

    Sample layout per ray (B, N) where N = n_stratified + n_perturbed_neg + n_perturbed_pos + 1:
        [:n_stratified]                                                stratified free-space samples (target 0)
        [n_stratified : n_stratified+n_perturbed_neg]                  negative perturbations, before surface (target 0, merged into the free loss)
        [n_stratified+n_perturbed_neg : n_stratified+n_perturbed]      positive perturbations, behind surface (handled by occ_loss_mode)
        [-1]                                                           on-surface sample (target = surface_target)

    The split between negative and positive perturbations also matches positive_perturbation_mask (False for the first
    n_perturbed_neg, True for the next n_perturbed_pos).
    """

    needs_grad = False  # criterion does not need gradients w.r.t. the input positions

    def __init__(self, cfg: OccCriterionConfig) -> None:
        """Construct the OCC criterion.

        Args:
            cfg: criterion configuration (loss weights, surface target, smoothness, occ-loss mode).
        """
        super().__init__()
        self.cfg = cfg

        self.n_stratified: int = -1
        self.n_perturbed: int = -1
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        self.mask_surface: torch.Tensor | None = None
        self.mask_strat: torch.Tensor | None = None
        self.mask_neg_pert: torch.Tensor | None = None
        self.mask_pos_pert: torch.Tensor | None = None

    def forward(
        self,
        pred_occ: torch.Tensor,
        pred_prior: torch.Tensor,
        positive_perturbation_mask: torch.Tensor,
        n_stratified: int,
        n_perturbed: int,
        valid_mask: torch.Tensor | None = None,
        pred_occ_perturb: torch.Tensor | None = None,
    ):
        """
        Args:
            pred_occ: (B, N) final occupancy logits (prior + residual).
            pred_prior: (B, N) prior occupancy logits.
            positive_perturbation_mask: (B, n_perturbed) True for samples behind the surface.
            n_stratified: number of stratified free-space samples per ray (slice [: n_stratified]).
            n_perturbed: number of perturbed samples per ray (slice [n_stratified : n_stratified + n_perturbed]).
            valid_mask: (B, N) optional boolean mask. False entries are excluded from every reduction.
            pred_occ_perturb: (B, N) optional occupancy logits at x + eps used by the smoothness term.

        Returns:
            loss: scalar weighted-sum loss tensor.
            loss_dict: per-term breakdown (Python floats) plus a `total_loss` entry.
        """
        # Cache for the helpers (_free_loss, _occ_loss) that slice by these counts.
        self.n_stratified = n_stratified
        self.n_perturbed = n_perturbed

        B, N = pred_occ.shape
        assert (
            N == self.n_stratified + self.n_perturbed + 1
        ), f"Expected last dim {self.n_stratified + self.n_perturbed + 1}, got {N}"

        n_strat = self.n_stratified
        n_pert = self.n_perturbed
        device = pred_occ.device

        if valid_mask is None:
            valid_strat = torch.ones((B, n_strat), dtype=torch.bool, device=device)
            valid_pert = torch.ones((B, n_pert), dtype=torch.bool, device=device)
            valid_surf = torch.ones((B,), dtype=torch.bool, device=device)
        else:
            valid_strat = valid_mask[:, :n_strat]
            valid_pert = valid_mask[:, n_strat : n_strat + n_pert]
            valid_surf = valid_mask[:, -1]

        self.mask_surface = valid_surf
        self.mask_strat = valid_strat
        self.mask_neg_pert = valid_pert & (~positive_perturbation_mask)
        self.mask_pos_pert = valid_pert & positive_perturbation_mask

        loss = 0
        loss_dict = {}

        if self.cfg.surface_loss_weight > 0:
            l = self._surface_loss(pred_occ)
            loss += self.cfg.surface_loss_weight * l
            loss_dict["surface_loss"] = l.item()
        if self.cfg.surface_loss_prior_weight > 0:
            l = self._surface_loss(pred_prior)
            loss += self.cfg.surface_loss_prior_weight * l
            loss_dict["surface_loss_prior"] = l.item()

        if self.cfg.free_loss_weight > 0:
            l = self._free_loss(pred_occ)
            loss += self.cfg.free_loss_weight * l
            loss_dict["free_loss"] = l.item()
        if self.cfg.free_loss_prior_weight > 0:
            l = self._free_loss(pred_prior)
            loss += self.cfg.free_loss_prior_weight * l
            loss_dict["free_loss_prior"] = l.item()

        if self.cfg.occ_loss_mode != "off":
            if self.cfg.occ_loss_weight > 0:
                l = self._occ_loss(pred_occ)
                loss += self.cfg.occ_loss_weight * l
                loss_dict["occ_loss"] = l.item()
            if self.cfg.occ_loss_prior_weight > 0:
                l = self._occ_loss(pred_prior)
                loss += self.cfg.occ_loss_prior_weight * l
                loss_dict["occ_loss_prior"] = l.item()

        if self.cfg.smoothness_weight > 0 and pred_occ_perturb is not None:
            diff_sq = (pred_occ - pred_occ_perturb) ** 2
            l = _masked_mean(diff_sq, valid_mask)
            loss += self.cfg.smoothness_weight * l
            loss_dict["smoothness_loss"] = l.item()

        loss_dict["total_loss"] = loss.item()
        return loss, loss_dict

    def _surface_loss(self, logits: torch.Tensor) -> torch.Tensor:
        logit_surface = logits[:, -1]
        target = torch.full_like(logit_surface, self.cfg.surface_target)
        per_elem = self.bce(logit_surface, target)
        return _masked_mean(per_elem, self.mask_surface)

    def _free_loss(self, logits: torch.Tensor) -> torch.Tensor:
        # TODO: maybe separate stratified and perturbed samples instead of merging them? The latter are noisier but
        # also more important to get right.
        logit_strat = logits[:, : self.n_stratified]
        logit_pert = logits[:, self.n_stratified : self.n_stratified + self.n_perturbed]
        per_strat = self.bce(logit_strat, torch.zeros_like(logit_strat))
        per_neg = self.bce(logit_pert, torch.zeros_like(logit_pert))
        stacked = torch.cat([per_strat, per_neg], dim=1)
        mask = torch.cat([self.mask_strat, self.mask_neg_pert], dim=1)
        return _masked_mean(stacked, mask)

    def _occ_loss(self, logits: torch.Tensor) -> torch.Tensor:
        logit_pert = logits[:, self.n_stratified : self.n_stratified + self.n_perturbed]
        if self.cfg.occ_loss_mode == "bce":
            per_elem = self.bce(logit_pert, torch.ones_like(logit_pert))
        elif self.cfg.occ_loss_mode == "hinge":
            per_elem = F.relu(self.cfg.occ_loss_hinge_margin - logit_pert)
        else:
            return logits.new_zeros(())  # off mode, zero loss
        return _masked_mean(per_elem, self.mask_pos_pert)
