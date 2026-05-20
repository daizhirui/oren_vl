from dataclasses import dataclass

import torch
import torch.nn as nn

from oren.utils.config_abc import ConfigABC
from oren.utils.registry import register_criterion


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp(min=1.0)


@dataclass
class SdfCriterionConfig(ConfigABC):
    boundary_loss_weight: float = 1.0
    boundary_loss_prior_weight: float = 1.0
    boundary_loss_type: str = "L1"
    perturbation_loss_weight: float = 1.0
    perturbation_loss_prior_weight: float = 1.0
    perturbation_loss_type: str = "L1"
    perturbation_loss_exp_penalty: float = 1.0
    eikonal_loss_surface_weight: float = 1.0
    eikonal_loss_surface_prior_weight: float = 1.0
    eikonal_loss_perturbation_weight: float = 1.0
    eikonal_loss_perturbation_prior_weight: float = 1.0
    eikonal_loss_space_weight: float = 1.0
    eikonal_loss_space_prior_weight: float = 1.0
    eikonal_loss_type: str = "L1"
    projection_loss_weight: float = 1.0
    projection_loss_prior_weight: float = 1.0
    projection_loss_type: str = "L1"
    heat_loss_weight: float = 0.0
    heat_loss_lambda: float = 0.0
    sign_loss_free_weight: float = 0.0
    sign_loss_occ_weight: float = 0.0
    sign_loss_temperature: float = 100.0


@register_criterion
class SdfCriterion(nn.Module):
    """
    SDF criterion for training implicit surface representations.

    Sample layout per ray (B, N) where N = n_stratified + n_perturbed_neg + n_perturbed_pos + 1:
        [:n_stratified]                                                stratified free-space samples (GT SDF in gt_sdf_stratified)
        [n_stratified : n_stratified+n_perturbed_neg]                  negative perturbations, before surface (GT SDF > 0)
        [n_stratified+n_perturbed_neg : n_stratified+n_perturbed]      positive perturbations, behind surface (GT SDF < 0)
        [-1]                                                           on-surface sample (GT SDF = 0)

    The GT SDF values for both perturbation slots are concatenated into gt_sdf_perturb in the same order;
    positive_perturbation_mask is False for the first n_perturbed_neg and True for the next n_perturbed_pos.
    """

    needs_grad = True  # criterion needs gradients w.r.t. the input positions for the eikonal loss

    def __init__(self, cfg: SdfCriterionConfig) -> None:
        """Construct the SDF criterion and resolve the per-term distance loss kernels.

        Args:
            cfg: criterion configuration carrying per-term weights and L1/L2 selectors.
        """
        super().__init__()
        self.cfg = cfg

        self.n_stratified: int = -1
        self.n_perturbed: int = -1

        if self.cfg.boundary_loss_type == "L1":
            self.boundary_loss_fn = nn.L1Loss(reduction="none")
        elif self.cfg.boundary_loss_type == "L2":
            self.boundary_loss_fn = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown boundary loss type: {self.cfg.boundary_loss_type}")

        if self.cfg.perturbation_loss_type == "L1":
            self.perturbation_loss_fn = nn.L1Loss(reduction="none")
        elif self.cfg.perturbation_loss_type == "L2":
            self.perturbation_loss_fn = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown perturbation loss type: {self.cfg.perturbation_loss_type}")

        if self.cfg.eikonal_loss_type == "L1":
            self.eikonal_loss_fn = nn.L1Loss(reduction="none")
        elif self.cfg.eikonal_loss_type == "L2":
            self.eikonal_loss_fn = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown eikonal loss type: {self.cfg.eikonal_loss_type}")

        if self.cfg.projection_loss_type == "L1":
            self.projection_loss_fn = nn.L1Loss(reduction="none")
        elif self.cfg.projection_loss_type == "L2":
            self.projection_loss_fn = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown projection loss type: {self.cfg.projection_loss_type}")

        self.mask_surface = None
        self.mask_perturb = None
        self.mask_strat = None

    def forward(
        self,
        pred_sdf: torch.Tensor,
        pred_prior: torch.Tensor,
        pred_grad: torch.Tensor,
        pred_prior_grad: torch.Tensor,
        gt_sdf_perturb: torch.Tensor,
        gt_sdf_stratified: torch.Tensor,
        positive_perturbation_mask: torch.Tensor,
        perturb_sigma_pos: float,
        perturb_sigma_neg: float,
        n_stratified: int,
        n_perturbed: int,
        valid_mask: torch.Tensor | None = None,
    ):
        """
        Compute the total loss as a weighted sum of individual loss components based on the configuration.
        Args:
            pred_sdf: (B, N) Predicted final SDF values for all samples (stratified + perturbed + surface).
            pred_prior: (B, N) Predicted prior SDF values for all samples (stratified + perturbed + surface).
            pred_grad: (B, 3) Predicted gradients for all samples.
            pred_prior_grad: (B, 3) Predicted prior gradients for all samples.
            gt_sdf_perturb: (B, n_perturbed) Ground truth SDF values for perturbed samples.
            gt_sdf_stratified: (B, n_stratified) Ground truth SDF values for stratified samples.
            positive_perturbation_mask: (B, n_perturbed) Boolean mask indicating which perturbed samples are in free
                space (positive SDF).
            perturb_sigma_pos: float, lower bound on |pred SDF| for positive perturbations (samples behind the
                surface). Should match cfg.sample_rays.sigma_s_pos.
            perturb_sigma_neg: float, lower bound on |pred SDF| for negative perturbations (samples in front of the
                surface). Should match cfg.sample_rays.sigma_s_neg.
            n_stratified: number of stratified free-space samples per ray (slice [: n_stratified]).
            n_perturbed: number of perturbed samples per ray (slice [n_stratified : n_stratified + n_perturbed]).
            valid_mask: (B, N) Optional boolean mask. False entries are excluded from every loss reduction.

        Returns:
            loss: scalar weighted-sum loss tensor.
            loss_dict: per-term breakdown (Python floats) plus a `total_loss` entry.
        """
        # Cache for the helpers (get_eikonal_loss_*, get_heat_loss, etc.) that index by these counts.
        self.n_stratified = n_stratified
        self.n_perturbed = n_perturbed

        if valid_mask is not None:
            self.mask_surface = valid_mask[:, -1]
            self.mask_perturb = valid_mask[:, self.n_stratified : self.n_stratified + self.n_perturbed]
            self.mask_strat = valid_mask[:, : self.n_stratified]
        else:
            self.mask_surface = None
            self.mask_perturb = None
            self.mask_strat = None

        loss = 0
        loss_dict = {}
        if self.cfg.boundary_loss_weight > 0:
            boundary_loss = self.get_boundary_loss(pred_sdf)
            loss += self.cfg.boundary_loss_weight * boundary_loss
            loss_dict["boundary_loss"] = boundary_loss.item()
        if self.cfg.boundary_loss_prior_weight > 0:
            boundary_loss_prior = self.get_boundary_loss(pred_prior)
            loss += self.cfg.boundary_loss_prior_weight * boundary_loss_prior
            loss_dict["boundary_loss_prior"] = boundary_loss_prior.item()

        if self.cfg.perturbation_loss_weight > 0:
            perturbation_loss = self.get_perturbation_loss(
                pred_sdf[:, self.n_stratified : self.n_stratified + self.n_perturbed],
                positive_perturbation_mask,
                gt_sdf_perturb,
                perturb_sigma_pos,
                perturb_sigma_neg,
            )
            loss += self.cfg.perturbation_loss_weight * perturbation_loss
            loss_dict["perturbation_loss"] = perturbation_loss.item()
        if self.cfg.perturbation_loss_prior_weight > 0:
            perturbation_loss_prior = self.get_perturbation_loss(
                pred_prior[:, self.n_stratified : self.n_stratified + self.n_perturbed],
                positive_perturbation_mask,
                gt_sdf_perturb,
                perturb_sigma_pos,
                perturb_sigma_neg,
            )
            loss += self.cfg.perturbation_loss_prior_weight * perturbation_loss_prior
            loss_dict["perturbation_loss_prior"] = perturbation_loss_prior.item()

        grad_norm = torch.norm(pred_grad, dim=-1)
        grad_norm_prior = torch.norm(pred_prior_grad, dim=-1)

        if self.cfg.eikonal_loss_surface_weight > 0:
            eikonal_loss_surface = self.get_eikonal_loss_surface(grad_norm)
            loss += self.cfg.eikonal_loss_surface_weight * eikonal_loss_surface
            loss_dict["eikonal_loss_surface"] = eikonal_loss_surface.item()
        if self.cfg.eikonal_loss_surface_prior_weight > 0:
            eikonal_loss_surface_prior = self.get_eikonal_loss_surface(grad_norm_prior)
            loss += self.cfg.eikonal_loss_surface_prior_weight * eikonal_loss_surface_prior
            loss_dict["eikonal_loss_surface_prior"] = eikonal_loss_surface_prior.item()
        if self.cfg.eikonal_loss_perturbation_weight > 0:
            eikonal_loss_perturbation = self.get_eikonal_loss_perturbation(grad_norm)
            loss += self.cfg.eikonal_loss_perturbation_weight * eikonal_loss_perturbation
            loss_dict["eikonal_loss_perturbation"] = eikonal_loss_perturbation.item()
        if self.cfg.eikonal_loss_perturbation_prior_weight > 0:
            eikonal_loss_perturbation_prior = self.get_eikonal_loss_perturbation(grad_norm_prior)
            loss += self.cfg.eikonal_loss_perturbation_prior_weight * eikonal_loss_perturbation_prior
            loss_dict["eikonal_loss_perturbation_prior"] = eikonal_loss_perturbation_prior.item()
        if self.cfg.eikonal_loss_space_weight > 0:
            eikonal_loss_space = self.get_eikonal_loss_space(grad_norm)
            loss += self.cfg.eikonal_loss_space_weight * eikonal_loss_space
            loss_dict["eikonal_loss_space"] = eikonal_loss_space.item()
        if self.cfg.eikonal_loss_space_prior_weight > 0:
            eikonal_loss_space_prior = self.get_eikonal_loss_space(grad_norm_prior)
            loss += self.cfg.eikonal_loss_space_prior_weight * eikonal_loss_space_prior
            loss_dict["eikonal_loss_space_prior"] = eikonal_loss_space_prior.item()

        if self.cfg.projection_loss_weight > 0:
            projection_loss = self.get_projection_loss(
                pred_sdf[:, : self.n_stratified],
                gt_sdf_stratified,
            )
            loss += self.cfg.projection_loss_weight * projection_loss
            loss_dict["projection_loss"] = projection_loss.item()

        if self.cfg.projection_loss_prior_weight > 0:
            projection_loss_prior = self.get_projection_loss(
                pred_prior[:, : self.n_stratified],
                gt_sdf_stratified,
            )
            loss += self.cfg.projection_loss_prior_weight * projection_loss_prior
            loss_dict["projection_loss_prior"] = projection_loss_prior.item()

        if self.cfg.heat_loss_weight > 0:
            if grad_norm is None:
                grad_norm = torch.norm(pred_grad, dim=-1)
            heat_loss = self.get_heat_loss(pred_sdf, grad_norm)
            loss += self.cfg.heat_loss_weight * heat_loss
            loss_dict["heat_loss"] = heat_loss.item()

        loss_dict["total_loss"] = loss.item()
        return loss, loss_dict

    def get_boundary_loss(self, pred_sdf: torch.Tensor):
        """Mean distance-loss between predicted on-surface SDF and zero.

        Args:
            pred_sdf: (B, N) predicted SDF values for all samples; only the surface slot `[:, -1]` is used.

        Returns:
            Scalar masked-mean boundary loss.
        """
        pred_sdf_surface = pred_sdf[:, -1]
        per_elem = self.boundary_loss_fn(pred_sdf_surface, torch.zeros_like(pred_sdf_surface))
        return _masked_mean(per_elem, self.mask_surface)

    def get_perturbation_loss(
        self,
        pred_sdf_perturb: torch.Tensor,
        positive_perturbation_mask: torch.Tensor,
        gt_sdf_perturb: torch.Tensor,
        perturb_sigma_pos: float,
        perturb_sigma_neg: float,
    ):
        """Hinge-style perturbation loss over both signs of perturbed samples.

        After flipping signs for positive perturbations, the loss is zero when `lower_bound <= pred <= upper_bound`,
        grows linearly above the upper bound (the GT magnitude), and grows exponentially below the lower bound
        (the sampling-floor `sigma`).

        Args:
            pred_sdf_perturb: (B, n_perturbed) predicted SDF values for perturbed samples.
            positive_perturbation_mask: (B, n_perturbed) True where the sample is behind the surface.
            gt_sdf_perturb: (B, n_perturbed) ground-truth SDF for perturbed samples (same sign convention as pred).
            perturb_sigma_pos: lower bound on `|pred|` for positive perturbations (samples behind the surface).
            perturb_sigma_neg: lower bound on `|pred|` for negative perturbations (samples in front of the
                surface).

        Returns:
            Scalar masked-mean perturbation loss over `self.mask_perturb`.
        """
        # Clone to avoid modifying input tensors
        pred_sdf_perturb = pred_sdf_perturb.clone()
        gt_sdf_perturb = gt_sdf_perturb.clone()

        # Flip sign for positive perturbations (negative SDF values) to unify loss calculation
        pred_sdf_perturb[positive_perturbation_mask] = -pred_sdf_perturb[positive_perturbation_mask]
        gt_sdf_perturb[positive_perturbation_mask] = -gt_sdf_perturb[positive_perturbation_mask]

        perturb_loss_upperbound = gt_sdf_perturb
        # Lower bound depends on which side of the surface the sample is on, matching the sampling range floor
        # (|offset| >= sigma_s_pos or sigma_s_neg).
        perturb_loss_lowerbound = torch.where(
            positive_perturbation_mask,
            torch.full_like(pred_sdf_perturb, perturb_sigma_pos),
            torch.full_like(pred_sdf_perturb, perturb_sigma_neg),
        )

        above_upper_loss = torch.clamp(pred_sdf_perturb - perturb_loss_upperbound, min=0)
        below_lower = torch.clamp(perturb_loss_lowerbound - pred_sdf_perturb, min=0)

        # Use exponential penalty for below lower bound to emphasize constraint
        below_lower_loss = torch.exp(self.cfg.perturbation_loss_exp_penalty * below_lower) - 1
        perturb_loss = above_upper_loss + below_lower_loss

        return _masked_mean(perturb_loss, self.mask_perturb)

    def get_eikonal_loss_surface(self, grad_norm: torch.Tensor):
        """Eikonal regularizer at the on-surface slot: push `|grad|` toward 1.

        Args:
            grad_norm: (B, N) per-sample gradient L2 norms; only the surface slot `[:, -1]` is used.

        Returns:
            Scalar masked-mean eikonal loss over `self.mask_surface`.
        """
        grad_norm_surface = grad_norm[:, -1]  # surface
        per_elem = self.eikonal_loss_fn(grad_norm_surface, torch.ones_like(grad_norm_surface))
        return _masked_mean(per_elem, self.mask_surface)

    def get_eikonal_loss_perturbation(self, grad_norm: torch.Tensor):
        """Eikonal regularizer at perturbed samples: push `|grad|` toward 1.

        Args:
            grad_norm: (B, N) per-sample gradient L2 norms; the perturbation slice is used.

        Returns:
            Scalar masked-mean eikonal loss over `self.mask_perturb`.
        """
        grad_norm_perturbation = grad_norm[:, self.n_stratified : self.n_stratified + self.n_perturbed]
        per_elem = self.eikonal_loss_fn(grad_norm_perturbation, torch.ones_like(grad_norm_perturbation))
        return _masked_mean(per_elem, self.mask_perturb)

    def get_eikonal_loss_space(self, grad_norm: torch.Tensor):
        """Eikonal regularizer at stratified free-space samples: push `|grad|` toward 1.

        Args:
            grad_norm: (B, N) per-sample gradient L2 norms; the stratified slice is used.

        Returns:
            Scalar masked-mean eikonal loss over `self.mask_strat`.
        """
        grad_norm_space = grad_norm[:, : self.n_stratified]  # free space
        per_elem = self.eikonal_loss_fn(grad_norm_space, torch.ones_like(grad_norm_space))
        return _masked_mean(per_elem, self.mask_strat)

    def get_projection_loss(self, pred_sdf: torch.Tensor, gt_sdf_stratified: torch.Tensor):
        """Distance loss between predicted and ground-truth SDF at stratified samples.

        Args:
            pred_sdf: (B, n_stratified) predicted SDF at stratified samples.
            gt_sdf_stratified: (B, n_stratified) ground-truth SDF at the same samples.

        Returns:
            Scalar masked-mean projection loss over `self.mask_strat`.
        """
        per_elem = self.projection_loss_fn(pred_sdf, gt_sdf_stratified)
        return _masked_mean(per_elem, self.mask_strat)

    def get_sign_loss(
        self,
        positive_sdf_mask: torch.Tensor,
        negative_sdf_mask: torch.Tensor,
        pred_sdf: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """Soft sign losses: push free-space predictions positive and occupied predictions negative.

        Uses `tanh(temperature * pred)` so the loss saturates once the predicted sign is confidently correct.

        Args:
            positive_sdf_mask: boolean mask selecting samples whose GT SDF should be positive (free space).
            negative_sdf_mask: boolean mask selecting samples whose GT SDF should be negative (occupied).
            pred_sdf: predicted SDF values; the two masks index into this tensor.
            mask: optional boolean validity mask AND-ed with both sign masks before reduction.

        Returns:
            sign_loss_free: scalar mean sign loss for free-space samples (zero if none).
            sign_loss_occ: scalar mean sign loss for occupied samples (zero if none).
        """
        if mask is not None:
            positive_sdf_mask = positive_sdf_mask & mask
            negative_sdf_mask = negative_sdf_mask & mask
        free_pred = pred_sdf[positive_sdf_mask.squeeze()]
        occ_pred = pred_sdf[negative_sdf_mask.squeeze()]
        if free_pred.numel() == 0:
            sign_loss_free = pred_sdf.new_zeros(())
        else:
            sign_loss_free = (torch.tanh(self.cfg.sign_loss_temperature * free_pred) - 1).abs().mean()
        if occ_pred.numel() == 0:
            sign_loss_occ = pred_sdf.new_zeros(())
        else:
            sign_loss_occ = (torch.tanh(self.cfg.sign_loss_temperature * occ_pred) + 1).abs().mean()
        return sign_loss_free, sign_loss_occ

    def get_heat_loss(self, pred_sdf: torch.Tensor, grad_norm: torch.Tensor):
        """Heat-equation regularizer evaluated at stratified free-space samples.

        Penalizes `0.5 * exp(-lambda * |sdf|)^2 * (|grad|^2 + 1)`, encouraging smaller gradients where the
        prediction is near the surface.

        Args:
            pred_sdf: (B, N) predicted SDF for all samples; only the stratified slice is used.
            grad_norm: (B, N) per-sample gradient L2 norms; only the stratified slice is used.

        Returns:
            Scalar masked-mean heat-equation loss over `self.mask_strat`.
        """
        pred_sdf = pred_sdf[:, : self.n_stratified]  # only consider free space samples
        grad_norm = grad_norm[:, : self.n_stratified]
        heat = torch.exp(-self.cfg.heat_loss_lambda * pred_sdf.abs())
        per_elem = 0.5 * heat**2 * (grad_norm**2 + 1)
        return _masked_mean(per_elem, self.mask_strat)
