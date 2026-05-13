"""
Common methods for semi-sparse octree implementations. This class is agnostic to the underlying octree implementation.
The derived classes should implement the octree operations of inserting points and finding voxel indices. When the tree
is modified, the buffers should be updated.

The sdf_priors and grad_priors are learnable parameters that store the SDF and gradient priors for each vertex in the
octree. The buffers store the octree structure:
- voxels: (N, 4) [x, y, z, voxel_size]
- voxel_centers: (N, 3) in meter
- vertex_indices: (N, 8) index of vertices, -1 if not exists
- structure: (N, 8) [children(8)]
"""

from abc import ABC, abstractmethod

from oren import torch
from oren.ga_trilinear import ga_trilinear, trilinear_interpolation, normalize_to_voxel_unit_cube
from oren.octree_config import OctreeConfig


class SemiSparseOctreeBase(torch.nn.Module, ABC):

    class ModelOutput:

        def __init__(
            self,
            voxel_indices: torch.Tensor,
            sdf_preds: torch.Tensor | None = None,
            occ_preds: torch.Tensor | None = None,
            implicit_features: torch.Tensor | None = None,
        ):
            self.voxel_indices = voxel_indices
            self.sdf_preds = sdf_preds
            self.occ_preds = occ_preds
            self.implicit_features = implicit_features

        @staticmethod
        def empty(n: int, device: torch.device, cfg: OctreeConfig):
            output = SemiSparseOctreeBase.ModelOutput(torch.zeros((n,), dtype=torch.long, device=device))
            output.sdf_preds = torch.zeros((n,), dtype=torch.float32, device=device) if cfg.enable_sdf else None
            output.occ_preds = torch.zeros((n,), dtype=torch.float32, device=device) if cfg.enable_occupancy else None
            output.implicit_features = (
                torch.zeros(
                    (n, cfg.implicit_feature_dim * cfg.implicit_num_levels),
                    dtype=torch.float32,
                    device=device,
                )
                if cfg.enable_implicit
                else None
            )
            return output

        def update(self, other: "SemiSparseOctreeBase.ModelOutput", start: int = 0, end: int = None):
            if end is None:
                end = self.voxel_indices.shape[0]
            assert (
                end - start == other.voxel_indices.shape[0]
            ), "The batch size of the other ModelOutput must match the update range."
            self.voxel_indices[start:end] = other.voxel_indices
            if self.sdf_preds is not None and other.sdf_preds is not None:
                self.sdf_preds[start:end] = other.sdf_preds
            if self.occ_preds is not None and other.occ_preds is not None:
                self.occ_preds[start:end] = other.occ_preds
            if self.implicit_features is not None and other.implicit_features is not None:
                self.implicit_features[start:end] = other.implicit_features

    def __init__(self, cfg: OctreeConfig):
        super(SemiSparseOctreeBase, self).__init__()
        self.cfg = cfg

        assert (
            self.cfg.enable_sdf or self.cfg.enable_occupancy or self.cfg.enable_implicit
        ), "At least one of enable_sdf, enable_occupancy or enable_implicit must be True"

        self.sdf_priors: torch.nn.Parameter | None = None
        if self.cfg.enable_sdf:
            # Initialize learnable parameters for SDF and gradient priors of each vertex
            self.sdf_priors = torch.nn.Parameter(
                torch.zeros((self.cfg.init_voxel_num,), dtype=torch.float32),
                requires_grad=True,
            )

        self.occupancy_priors: torch.nn.Parameter | None = None
        if self.cfg.enable_occupancy:
            self.occupancy_priors = torch.nn.Parameter(
                torch.full((self.cfg.init_voxel_num,), self.cfg.init_occ_prior, dtype=torch.float32),
                requires_grad=True,
            )

        self.grad_priors: torch.nn.Parameter | None = None
        if self.cfg.gradient_augmentation:
            self.grad_priors = torch.nn.Parameter(
                torch.zeros((self.cfg.init_voxel_num, 3), dtype=torch.float32),
                requires_grad=True,
            )

        self.implicit_features: torch.nn.Parameter | None = None
        if self.cfg.enable_implicit:
            assert (
                self.cfg.implicit_feature_dim > 0
            ), "implicit_feature_dim must be greater than 0 when enable_implicit is True"
            assert (
                self.cfg.implicit_num_levels > 0
            ), "implicit_num_levels must be greater than 0 when enable_implicit is True"
            self.implicit_features = torch.nn.Parameter(
                torch.zeros((self.cfg.init_voxel_num, self.cfg.implicit_feature_dim), dtype=torch.float32),
                requires_grad=True,
            )

        self.ever_inserted = False

        n = self.cfg.init_voxel_num
        self.register_buffer("voxels", torch.zeros((n, 4), dtype=torch.float32))
        self.register_buffer("voxel_centers", torch.zeros((n, 3), dtype=torch.float32))
        self.register_buffer("vertex_indices", torch.zeros((n, 8), dtype=torch.int32))
        self.register_buffer("structure", torch.zeros((n, 8), dtype=torch.int32))

        self.voxels: torch.Tensor  # (N, 4) [x, y, z, voxel_size]
        self.voxel_centers: torch.Tensor  # (N, 3) in meter
        self.vertex_indices: torch.Tensor  # (N, 8) index of vertices, -1 if not exists
        self.structure: torch.Tensor  # (N, 8) [children(8)]

    @abstractmethod
    def points_to_voxels(self, points: torch.Tensor) -> torch.Tensor:
        """
        Converts points to voxel coordinates.
        Args:
            points: (..., 3) point cloud in world coordinates
        Returns:
            voxels: (..., 3) voxel coordinates
        """
        pass

    @abstractmethod
    def insert_voxels(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Inserts voxels into the octree, updates the buffers and returns the voxel indices.
        Args:
            voxels: (n_voxels, 3) voxel coordinates
        Returns:
            voxel_indices: (n_voxels,) index of the voxel for each voxel
        """
        pass

    @torch.no_grad()
    def insert_points(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inserts points into the octree.
        Args:
            points: (n_points, 3) point cloud in world coordinates
        Returns:
            voxels_unique: (n_unique, 3) unique voxel coordinates inserted
            voxel_indices: (n_unique,) index of the voxel for each voxel, on CPU.
        """
        voxels = self.points_to_voxels(points)  # (n_points, 3) voxel coordinates
        voxels_raw, counts = torch.unique(voxels, dim=0, return_inverse=False, return_counts=True)
        voxels_valid = voxels_raw[counts > self.cfg.insertion_threshold]  # (n_valid, 3) of grid coordinates
        voxels_unique = torch.unique(voxels_valid, dim=0)  # (n_unique, 3) of grid coordinates
        if voxels_unique.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device="cpu"), torch.empty((0,), dtype=torch.long, device="cpu")
        if self.cfg.skip_insertion_if_exists and self.ever_inserted:
            device = self.voxels.device
            voxel_indices = self.find_voxel_indices(voxels_unique.to(device), True, level=1)  # (n_unique,)
            voxel_sizes = self.get_voxel_discrete_size(voxel_indices)  # (n_unique,)
            mask = voxel_sizes != 1  # only insert voxels that do not exist at size 1
            voxel_indices[mask] = self.insert_voxels(voxels_unique[mask]).to(device)
        else:
            voxel_indices = self.insert_voxels(voxels_unique)
        return voxels_unique, voxel_indices.cpu()

    @torch.no_grad()
    def get_voxel_discrete_size(self, voxel_indices: torch.Tensor) -> torch.Tensor:
        """
        Get the voxel sizes for the given voxel indices.
        Args:
            voxel_indices: (...) index of the voxels

        Returns:
            (..., ) voxel discrete sizes
        """
        assert self.voxels is not None, "Octree is empty. Please insert points first."
        assert voxel_indices.dtype == torch.long, "voxel_indices must be of type torch.long"

        voxel_sizes = self.voxels[voxel_indices.view(-1), -1]  # (..., ) discrete sizes
        voxel_sizes = voxel_sizes.view(voxel_indices.shape)
        voxel_sizes[voxel_indices < 0] = -1  # set invalid voxel sizes to -1
        return voxel_sizes

    @abstractmethod
    def find_voxel_indices(self, points: torch.Tensor, are_voxels: bool, level: int = 1) -> torch.Tensor:
        """
        Finds the voxel indices for the given points.
        Args:
            points: (n_points, 3) point cloud in world coordinates
            are_voxels: bool, if True, points are treated as voxel coordinates
        Returns:
            voxel_indices: (n_points,) index of the voxel for each point, -1 if not exists
        """
        pass

    def forward(self, points: torch.Tensor, voxel_indices: torch.Tensor = None, batch_size: int = -1) -> ModelOutput:
        """
        Forward pass of the octree.
        Args:
            points: (n_points, 3) point cloud in world coordinates
            voxel_indices: (n_points,) index of the voxel for each point, -1 if not exists
            batch_size: int, number of points to process in a batch. If -1, process all points at once.
        Returns:
            ModelOutput containing the sdf predictions and voxel indices.
        """
        if voxel_indices is None:
            voxel_indices = self.find_voxel_indices(points, False)

        if batch_size > 0:
            n_points = points.shape[0]
            result = self.ModelOutput.empty(n_points, points.device, self.cfg)

            for start in range(0, n_points, batch_size):
                end = min(start + batch_size, n_points)
                batch_result = self.forward(points[start:end], voxel_indices[start:end])
                result.update(batch_result, start, end)

            return result

        # Implement the forward pass logic here
        assert voxel_indices.dtype == torch.long

        # find the voxel centers for each point
        voxel_centers = self.voxel_centers[voxel_indices]  # (n_points, 3)
        # find the vertex indices for each point
        vertex_indices = self.vertex_indices[voxel_indices]  # (n_points, 8)
        # find the voxel sizes for each point
        voxel_sizes = self.voxels[voxel_indices, -1:]  # (n_points, 1)

        result = self.ModelOutput(voxel_indices)
        # (n_points, 3), normalized to [0, 1]
        voxel_offsets = normalize_to_voxel_unit_cube(points, voxel_centers, voxel_sizes, self.cfg.resolution)
        vertex_grad_priors = None

        if self.cfg.enable_sdf:

            # get the sdf priors and gradient priors for each vertex
            vertex_sdf_priors = self.sdf_priors[vertex_indices]  # (n_points, 8)
            if self.cfg.gradient_augmentation and vertex_grad_priors is None:
                vertex_grad_priors = self.grad_priors[vertex_indices]  # (n_points, 8, 3)

            result.sdf_preds, voxel_offsets = ga_trilinear(
                points=points,
                voxel_centers=voxel_centers,
                voxel_sizes=voxel_sizes,
                resolution=self.cfg.resolution,
                vertex_values=vertex_sdf_priors,
                vertex_grad=vertex_grad_priors,
                gradient_augmentation=self.cfg.gradient_augmentation,
                little_endian=self.little_endian_vertex_order,
                voxel_offsets=voxel_offsets,
            )

        if self.cfg.enable_occupancy:
            vertex_occupancy_priors = self.occupancy_priors[vertex_indices]  # (n_points, 8)
            if self.cfg.gradient_augmentation and vertex_grad_priors is None:
                vertex_grad_priors = self.grad_priors[vertex_indices]  # (n_points, 8, 3)

            result.occ_preds, voxel_offsets = ga_trilinear(
                points=points,
                voxel_centers=voxel_centers,
                voxel_sizes=voxel_sizes,
                resolution=self.cfg.resolution,
                vertex_values=vertex_occupancy_priors,
                vertex_grad=vertex_grad_priors,
                gradient_augmentation=self.cfg.gradient_augmentation,
                little_endian=self.little_endian_vertex_order,
                voxel_offsets=voxel_offsets,
            )

        # If implicit features are used, perform trilinear interpolation for each level of implicit features
        if self.cfg.enable_implicit:
            # (n_points, 8, implicit_feature_dim)
            per_point_vertex_implicit_features_level_1 = self.implicit_features[vertex_indices]
            implicit_features = trilinear_interpolation(
                points=voxel_offsets,
                per_point_vertex_values=per_point_vertex_implicit_features_level_1,
                little_endian=self.little_endian_vertex_order,
            )
            if self.cfg.implicit_num_levels > 1:
                implicit_features = [implicit_features]
                for level in range(2, self.cfg.implicit_num_levels + 1):
                    # level=1: leaf level
                    implicit_voxel_indices = self.find_voxel_indices(points, False, level)
                    implicit_voxel_centers = self.voxel_centers[implicit_voxel_indices]  # (n_points, 3)
                    implicit_vertex_indices = self.vertex_indices[implicit_voxel_indices]  # (n_points, 8)
                    implicit_voxel_sizes = self.voxels[implicit_voxel_indices, -1:]  # (n_points, 1)
                    # (n_points, 3), normalized to [0, 1]
                    voxel_offsets_level_n = normalize_to_voxel_unit_cube(
                        points,
                        implicit_voxel_centers,
                        implicit_voxel_sizes,
                        self.cfg.resolution,
                    )
                    # (n_points, 8, implicit_feature_dim)
                    per_point_vertex_implicit_features_level_n = self.implicit_features[implicit_vertex_indices]
                    implicit_features_level_n = trilinear_interpolation(
                        points=voxel_offsets_level_n,
                        per_point_vertex_values=per_point_vertex_implicit_features_level_n,
                        little_endian=self.little_endian_vertex_order,
                    )
                    implicit_features.append(implicit_features_level_n)
                implicit_features = torch.cat(implicit_features, dim=-1)

            result.implicit_features = implicit_features

        return result

    @torch.no_grad()
    def grid_vertex_filter(
        self,
        grid_points: torch.Tensor,
        min_voxel_size: int = 1,
        max_voxel_size: int = 2,
        dilation_iters: int = 1,
        batch_size: int = 204800,
        device: str | None = None,
    ) -> torch.Tensor:
        """
        Filter out grid vertices that are in voxels that are too big.
        Args:
            grid_points: (nx, ny, nz, 3) grid points in world coordinates
            min_voxel_size: minimum voxel size to keep
            max_voxel_size: maximum voxel size to keep
            dilation_iters: number of dilation iterations to fill small holes
            batch_size: number of points to process in a batch
            device: device to use, if None, use the device of grid_points

        Returns:
            (nx, ny, nz) boolean mask, True if the vertex is valid (in a voxel that is not too big)
        """
        assert grid_points.ndim == 4 and grid_points.shape[-1] == 3

        if batch_size <= 0:
            bs = grid_points.shape[0] * grid_points.shape[1] * grid_points.shape[2]
        else:
            bs = batch_size

        grid_shape = grid_points.shape
        grid_points = grid_points.view(-1, 3)

        model_device = self.structure.device
        if device is None:  # device for the output mask
            device = grid_points.device

        valid_mask = []
        for start in range(0, grid_points.shape[0], bs):
            end = min(start + bs, grid_points.shape[0])
            indices = self.find_voxel_indices(grid_points[start:end].to(model_device), False).view(-1)
            sizes = self.get_voxel_discrete_size(indices)
            valid_mask.append(((sizes >= min_voxel_size) & (sizes <= max_voxel_size)).to(device))

        if len(valid_mask) == 1:
            valid_mask = valid_mask[0]
        else:
            valid_mask = torch.cat(valid_mask, dim=0)
        valid_mask = valid_mask.view(grid_shape[:-1])  # (nx, ny, nz)

        # run a dilation to fill small holes: if any vertex is valid, we should keep the cube
        # such that we need to mark all 8 vertices as valid.
        # use convolution with all-ones kernel
        kernel = torch.ones((3, 3, 3), dtype=torch.float32, device=valid_mask.device).view(1, 1, 3, 3, 3)
        for _ in range(dilation_iters):
            valid_mask = (  # (nx, ny, nz)
                torch.nn.functional.conv3d(
                    input=valid_mask.view(1, 1, *valid_mask.shape).to(torch.float32),
                    weight=kernel,
                    padding=1,
                ).view(*valid_mask.shape)
                >= 1
            ).to(torch.bool)

        return valid_mask

    @property
    @abstractmethod
    def little_endian_vertex_order(self) -> bool:
        """
        Returns:
            bool: True if the vertex order is little-endian, False if big-endian.
        """
        pass
