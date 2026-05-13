from dataclasses import dataclass
from functools import reduce
from typing import Optional

import torch

from .config_abc import ConfigABC
from .nearest_neighbor import nearest_neighbor


def generate_sample_mask(shape, num_samples: int):
    n = reduce(lambda x, y: x * y, shape)
    if num_samples >= n:
        return torch.ones(shape, dtype=torch.bool)

    indices = torch.randperm(n)[:num_samples]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[indices] = True
    return mask.view(shape)


@dataclass
class SampleRaysConfig(ConfigABC):
    n_stratified: int = 20  # number of stratified samples
    n_perturbed_pos: int = 4  # number of perturbed samples behind the surface (positive_perturbation_mask=True)
    n_perturbed_neg: int = 4  # number of perturbed samples in front of the surface (positive_perturbation_mask=False)
    depth_min: float = 0.07  # minimum (sensor) depth value
    depth_max: float = 10.0  # maximum (sensor) depth value
    surface_margin: float = 0.10  # additional range beyond surface
    sigma_s_pos: float = 0.05  # σ for positive perturbations (samples behind the surface, occupied side)
    sigma_s_neg: float = 0.05  # σ for negative perturbations (samples in front of the surface, free side)

    @property
    def n_perturbed(self) -> int:
        return self.n_perturbed_pos + self.n_perturbed_neg

    def __post_init__(self):
        assert self.n_stratified >= 0, "n_stratified must be non-negative"
        assert self.n_perturbed_pos >= 0, "n_perturbed_pos must be non-negative"
        assert self.n_perturbed_neg >= 0, "n_perturbed_neg must be non-negative"
        assert self.n_perturbed_pos + self.n_perturbed_neg > 0, "at least one of n_perturbed_pos/neg must be positive"
        assert self.depth_min > 0, "depth_min must be positive"
        assert self.depth_max > self.depth_min, "depth_max must be greater than depth_min"
        assert self.surface_margin >= 0, "surface_margin must be non-negative"
        assert (
            self.depth_min + self.surface_margin < self.depth_max
        ), "depth_min + surface_margin must be less than depth_max"
        if self.n_perturbed_pos > 0:
            assert self.sigma_s_pos > 0, "sigma_s_pos must be positive when n_perturbed_pos > 0"
        if self.n_perturbed_neg > 0:
            assert self.sigma_s_neg > 0, "sigma_s_neg must be positive when n_perturbed_neg > 0"
        return super().__post_init__()


@dataclass
class SampleResults:
    sampled_xyz: torch.Tensor
    positive_sdf_mask: torch.Tensor
    negative_sdf_mask: torch.Tensor
    valid_indices: torch.Tensor
    stratified_sdf: Optional[torch.Tensor]  # None when generate_sdf_samples(compute_sdf_labels=False)
    perturbation_sdf: Optional[torch.Tensor]  # None when generate_sdf_samples(compute_sdf_labels=False)
    n_stratified: int
    n_perturbed: int
    positive_perturbation_mask: torch.Tensor


@torch.no_grad()
def generate_sdf_samples(
    rays_d_all: torch.Tensor,
    rays_o_all: torch.Tensor,
    depth_samples_all: torch.Tensor,
    cfg: SampleRaysConfig,
    extra_surface_pcd: torch.Tensor = None,
    device=None,
    compute_sdf_labels: bool = True,
) -> SampleResults:
    """
    Sample points along rays using surface-guided sampling strategy (GPU parallelized).
    Only processes valid rays (positive, finite depth values) and returns compact results.

    Args:
        rays_d_all: Ray directions (num_rays, 3)
        rays_o_all: Ray origins (num_rays, 3)
        depth_samples_all: Surface depth values D[u,v] (num_rays,) or (num_rays, 1)
        cfg: Configuration for sampling
        extra_surface_pcd: Additional surface points for computing SDF (num_extra_points, 3)
        device: Device for computation

    Returns:
        sampled_xyz: 3D coordinates of sampled points (num_valid_rays, N+M+1, 3)
        sampled_depth: Depth values for sampled points (num_valid_rays, N+M+1)
        negative_sdf_mask: Mask indicating positive perturbations (num_valid_rays, N+M+1)
        surface_mask: Mask indicating surface samples (num_valid_rays, N+M+1)
        perturbation_mask: Mask indicating perturbation samples (num_valid_rays, N+M+1)
        ray_sample_mask: Mask indicating free space samples (num_valid_rays, N+M+1)
        valid_indices: Indices of valid rays in original input (num_valid_rays, )
    """
    if device is None:
        device = rays_d_all.device

    n_stratified = cfg.n_stratified
    n_positive = cfg.n_perturbed_pos  # samples after surface (positive_perturbation_mask=True)
    n_negative = cfg.n_perturbed_neg  # samples before surface (positive_perturbation_mask=False)
    n_perturbed = n_positive + n_negative
    depth_min = cfg.depth_min
    depth_max = cfg.depth_max
    surface_margin = cfg.surface_margin
    sigma_s_pos = cfg.sigma_s_pos
    sigma_s_neg = cfg.sigma_s_neg

    # total_samples = n_stratified + n_perturbed + 1

    # Create valid mask to filter out invalid depth values (0, negative, or NaN)
    # Valid rays must have depth in (depth_min + surface_margin, depth_max) to
    # ensure we can sample both free space and near-surface points.
    valid_mask = (
        (depth_samples_all > depth_min + surface_margin)
        & (depth_samples_all < depth_max)
        & torch.isfinite(depth_samples_all)
    )
    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]  # (num_valid_rays,)
    num_valid_rays = valid_indices.shape[0]

    # Extract only valid rays data
    rays_d_valid = rays_d_all[valid_indices]  # (num_valid_rays, 3)
    rays_o_valid = rays_o_all[valid_indices]  # (num_valid_rays, 3)
    depth_samples_valid = depth_samples_all[valid_indices].flatten()  # (num_valid_rays,)

    #############################################################
    # 1. Stratified sampling (vectorized) - only for valid rays #
    #############################################################
    # Compared with uniform sampling from [depth_min, d_max],
    # stratified sampling ensures coverage of free space.

    if n_stratified > 0:
        d_max = depth_samples_valid - surface_margin  # (num_valid_rays,)
        d_range = d_max - depth_min  # (num_valid_rays,)

        if n_stratified == 1:
            bin_size = d_range.unsqueeze(1)  # (num_valid_rays, 1)
            bin_starts = torch.full((num_valid_rays, 1), depth_min, dtype=torch.float32, device=device)
        else:
            bin_size = d_range.unsqueeze(1) / n_stratified  # (num_valid_rays, 1)
            bin_indices = torch.arange(n_stratified, device=device, dtype=torch.float32).unsqueeze(0)
            bin_starts = depth_min + bin_indices * bin_size  # (num_valid_rays, n_stratified)

        # Uniform random samples within each bin
        uniform_samples = torch.rand(num_valid_rays, n_stratified, device="cpu").to(device)
        stratified_depths = bin_starts + uniform_samples * bin_size  # (num_valid_rays, n_stratified)
    else:
        stratified_depths = torch.empty((num_valid_rays, 0), dtype=torch.float32, device=device)

    # Stratified samples sit in free space (depth < surface depth), so their sdf signs are fixed.
    stratified_positive_mask = torch.zeros(num_valid_rays, n_stratified, dtype=torch.bool, device=device)
    stratified_negative_mask = torch.ones(num_valid_rays, n_stratified, dtype=torch.bool, device=device)

    ############################
    # 2. perturbation by depth #
    ############################
    # n_negative samples in [-3*sigma_s_neg, -sigma_s_neg], n_positive samples in [sigma_s_pos, 3*sigma_s_pos].
    # The deadband [-sigma_s_*, sigma_s_*] is avoided to keep perturbations from straddling the surface.
    # pos/neg counts and σ are configurable independently via SampleRaysConfig.

    # Negative perturbations: [-3*sigma_s_neg, -sigma_s_neg]
    negative_offsets = torch.rand(num_valid_rays, n_negative, device="cpu").to(device)  # [0, 1]
    negative_offsets = -3 * sigma_s_neg + negative_offsets * (2 * sigma_s_neg)  # [-3*sigma_s_neg, -sigma_s_neg]

    # Positive perturbations: [sigma_s_pos, 3*sigma_s_pos]
    positive_offsets = torch.rand(num_valid_rays, n_positive, device="cpu").to(device)  # [0, 1]
    positive_offsets = sigma_s_pos + positive_offsets * (2 * sigma_s_pos)  # [sigma_s_pos, 3*sigma_s_pos]

    # Note that:
    # positive offsets cause negative sdf values, while negative offsets cause positive sdf values.

    # Combine offsets and compute depths
    perturbation_offsets = torch.cat([negative_offsets, positive_offsets], dim=1)  # (num_valid_rays, n_perturbed)
    perturbed_depths = depth_samples_valid.unsqueeze(1) + perturbation_offsets  # (num_valid_rays, n_perturbed)

    # Create mask: first n_negative are False (negative), last n_positive are True (positive)
    positive_perturbation_mask = torch.cat(
        [
            torch.zeros(num_valid_rays, n_negative, dtype=torch.bool, device=device),
            torch.ones(num_valid_rays, n_positive, dtype=torch.bool, device=device),
        ],
        dim=1,
    )  # (num_valid_rays, n_perturbed)

    ######################
    # 3. Surface samples #
    ######################
    surface_samples = depth_samples_valid.view(-1, 1)  # (num_valid_rays, 1)

    #######################
    # Combine all samples #
    #######################

    # (num_valid_rays, n_stratified + n_perturbed + 1)
    all_depths = torch.cat([stratified_depths, perturbed_depths, surface_samples], dim=1)

    # Original negative_sdf_mask for backward compatibility
    negative_sdf_mask = torch.cat(
        [
            stratified_positive_mask,
            positive_perturbation_mask,
            torch.zeros(num_valid_rays, 1, dtype=torch.bool, device=device),
        ],
        dim=1,
    )  # (num_valid_rays, n_stratified + n_perturbed + 1)
    positive_sdf_mask = torch.cat(
        [
            stratified_negative_mask,
            ~positive_perturbation_mask,
            torch.zeros(num_valid_rays, 1, dtype=torch.bool, device=device),
        ],
        dim=1,
    )  # (num_valid_rays, n_stratified + n_perturbed + 1)

    # Calculate 3D coordinates (vectorized)
    # (num_valid_rays, 1, 3) + (num_valid_rays, total_samples, 1) * (num_valid_rays, 1, 3)
    sampled_xyz = rays_o_valid.unsqueeze(1) + all_depths.unsqueeze(2) * rays_d_valid.unsqueeze(1)

    if compute_sdf_labels:
        sdf = nearest_neighbor(
            src=sampled_xyz[:, :-1].contiguous().view(-1, 3),
            dst=(
                sampled_xyz[:, -1].contiguous().view(-1, 3)
                if extra_surface_pcd is None
                else torch.cat([extra_surface_pcd.to(device), sampled_xyz[:, -1].contiguous().view(-1, 3)], dim=0)
            ),
        )[0].view(num_valid_rays, -1)
        stratified_sdf = sdf[:, :n_stratified].view(num_valid_rays, n_stratified)
        perturbation_sdf = sdf[:, n_stratified : n_stratified + n_perturbed].view(num_valid_rays, n_perturbed)
        perturbation_sdf = torch.where(positive_perturbation_mask, -perturbation_sdf, perturbation_sdf)
    else:
        stratified_sdf = None
        perturbation_sdf = None

    return SampleResults(
        sampled_xyz=sampled_xyz,
        positive_sdf_mask=positive_sdf_mask,
        negative_sdf_mask=negative_sdf_mask,
        valid_indices=valid_indices,
        stratified_sdf=stratified_sdf,
        perturbation_sdf=perturbation_sdf,
        n_stratified=n_stratified,
        n_perturbed=n_perturbed,
        positive_perturbation_mask=positive_perturbation_mask,
    )
