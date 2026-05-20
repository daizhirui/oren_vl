import torch

from oren.utils.depth_utils import depth_to_camera_points


class Frame:
    """Common scaffolding shared by depth, LiDAR, and VL frames.

    Subclasses must populate `self.points` (trailing dim 3) and `self.valid_mask` (broadcastable
    against `self.points` on all but the last dim) after calling `super().__init__(fid, ref_pose)`.
    The default `get_points` / `apply_bound` / `sample_points` implementations then work for both
    `(N, 3)` + `(N,)` and `(H, W, 3)` + `(H, W)` layouts via boolean fancy indexing and 1D flat
    indices into `valid_mask.flatten()`.
    """

    def __init__(
        self,
        fid: int,
        ref_pose: torch.Tensor,
        min_range: float | None = 0.0,
        max_range: float | None = None,
        is_depth: bool = False,
    ) -> None:
        """Store the frame id, normalize `ref_pose`, and record the range-filter config.

        Args:
            fid: frame index used as the timestamp / id.
            ref_pose: `(4, 4)` sensor-to-world pose, or a flat 16-element tensor/array.
            min_range: lower bound (exclusive) on the per-point range metric; defaults to `0.0` so degenerate
                zero-valued entries are masked out. Pass `None` to disable the lower bound.
            max_range: optional upper bound (exclusive) on the per-point range metric; `None` (default) means no
                upper cap.
            is_depth: if True the range metric is `points[..., 2]` (camera-z, used by depth maps); if False it is
                `||points||_2` (L2 distance from the sensor origin, used by LiDAR / VL point clouds).
        """
        self.stamp = fid
        if ref_pose.ndim != 2:
            ref_pose = ref_pose.reshape(4, 4)
        if not isinstance(ref_pose, torch.Tensor):  # from gt data
            self.ref_pose = torch.tensor(ref_pose, requires_grad=False, dtype=torch.float32)
        else:  # from tracked data
            self.ref_pose = ref_pose.clone().requires_grad_(False)
        # Lazy cache of self._to_world(self.points); materialized on first access to `points_world` and kept in sync
        # by `project_to_bound` (the only post-init mutator of `self.points`). Call `_invalidate_world_cache()` if
        # `self.points` or `self.ref_pose` is reassigned/moved externally.
        self._points_world: torch.Tensor | None = None
        # Range-filter config, consumed by `_compute_range_mask()` after the subclass populates `self.points`.
        self._min_range = min_range
        self._max_range = max_range
        self._is_depth = is_depth

    def __eq__(self, value) -> bool:
        if not isinstance(value, Frame):
            return False
        return self.stamp == value.stamp

    def __ne__(self, value) -> bool:
        if not isinstance(value, Frame):
            return True
        return self.stamp != value.stamp

    def get_frame_index(self) -> int:
        return self.stamp

    def get_ref_pose(self) -> torch.Tensor:
        return self.ref_pose

    def get_ref_translation(self) -> torch.Tensor:
        return self.ref_pose[:3, 3]

    def get_ref_rotation(self) -> torch.Tensor:
        return self.ref_pose[:3, :3]

    def get_valid_mask(self) -> torch.Tensor:
        return self.valid_mask

    def get_rays_direction(self) -> torch.Tensor:
        raise NotImplementedError

    def get_depth(self) -> torch.Tensor:
        raise NotImplementedError

    def _to_world(self, points: torch.Tensor) -> torch.Tensor:
        """Transform `points` from the sensor frame to the world frame using `self.ref_pose`.

        Broadcasts cleanly over any leading dims as long as the trailing dim is 3.
        """
        pose = self.ref_pose.to(points.device)
        return points @ pose[:3, :3].T + pose[:3, 3]

    @property
    def points_world(self) -> torch.Tensor:
        """World-frame copy of `self.points`, shape-matching it and computed once on first access.

        `apply_bound` / `get_points` / `sample_points` / `project_to_bound` read from this instead of re-running
        `_to_world` per call. `project_to_bound` patches the cache in place for entries it mutates.
        """
        if self._points_world is None:
            self._points_world = self._to_world(self.points)
        return self._points_world

    def _invalidate_world_cache(self) -> None:
        """Drop the cached `points_world` so the next access recomputes from `self.points` / `self.ref_pose`."""
        self._points_world = None

    def _compute_range_mask(self) -> torch.Tensor:
        """Build the initial `valid_mask` from `self.points` using the configured range metric.

        Metric is `points[..., 2]` (z-coord) when `is_depth=True` else `||points||_2`. Returns an all-True mask
        shaped like `self.points.shape[:-1]` if neither `min_range` nor `max_range` was given. Subclasses call this
        once `self.points` is set and assign the result to `self.valid_mask`.

        Returns:
            `valid_mask` of shape `self.points.shape[:-1]` and dtype bool, where True means the point passed the range
            filter and any pre-existing `mask`.
        """
        shape = self.points.shape[:-1]
        device = self.points.device
        if self._min_range is None and self._max_range is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        metric = self.points[..., 2] if self._is_depth else torch.norm(self.points, dim=-1)
        mask = torch.ones(shape, dtype=torch.bool, device=device)
        if self._min_range is not None:
            assert self._min_range >= 0, f"min_range {self._min_range} must be non-negative"
            mask = mask & (metric > self._min_range)
        if self._max_range is not None:
            assert self._max_range > 0, f"max_range {self._max_range} must be positive"
            mask = mask & (metric < self._max_range)
        return mask

    def get_points(self, to_world_frame: bool, device: str) -> torch.Tensor:
        """Return the valid points, optionally transformed to world coordinates.

        Args:
            to_world_frame: if True, transform points to world coordinates via `self.ref_pose`.
            device: device to move tensors to before returning.

        Returns:
            `(N, 3)` valid points in sensor or world coordinates.
        """
        src = self.points_world if to_world_frame else self.points
        return src[self.valid_mask].reshape(-1, 3).to(device)

    def apply_bound(self, bound_min: torch.Tensor, bound_max: torch.Tensor) -> None:
        """Narrow `valid_mask` to points whose world-frame coordinates lie inside the AABB.

        Args:
            bound_min: `(3,)` lower corner of the axis-aligned bounding box in world coordinates.
            bound_max: `(3,)` upper corner of the axis-aligned bounding box in world coordinates.
        """
        bound_min = bound_min.to(self.points.device)
        bound_max = bound_max.to(self.points.device)
        points_world = self.points_world
        in_bounds = ((points_world >= bound_min) & (points_world <= bound_max)).all(dim=-1)
        self.valid_mask = self.valid_mask & in_bounds

    @torch.no_grad()
    def project_to_bound(self, bound_min: torch.Tensor, bound_max: torch.Tensor) -> None:
        """Project out-of-bound valid points back onto the AABB along the ray from the sensor origin.

        Any valid point whose world-frame coordinate falls outside `[bound_min, bound_max]` is moved onto the box
        boundary along the ray from `ref_pose`'s translation; valid points already inside the box are left untouched.
        Works for any `points` / `valid_mask` layout `Frame` supports (e.g. `(N, 3)` + `(N,)` or `(H, W, 3)` + `(H, W)`)
        via boolean fancy indexing.

        Args:
            bound_min: `(3,)` lower corner of the axis-aligned bounding box in world coordinates.
            bound_max: `(3,)` upper corner of the axis-aligned bounding box in world coordinates.
        """
        device = self.points.device
        bound_min = bound_min.to(device)
        bound_max = bound_max.to(device)

        mask = self.valid_mask
        if not mask.any():
            return

        R_s2w = self.ref_pose[:3, :3]
        t_s2w = self.ref_pose[:3, 3]
        points_world_valid = self.points_world[mask]

        # Out-of-bounds mask
        oob_mask = (points_world_valid < bound_min).any(-1) | (points_world_valid > bound_max).any(-1)
        if not oob_mask.any():
            return

        pts_to_proj = points_world_valid[oob_mask]
        origin = t_s2w.view(1, 3)

        # Ray-AABB intersection using the slab method.
        ray_dir = pts_to_proj - origin
        safe_dir = torch.where(ray_dir.abs() < 1e-8, torch.sign(ray_dir) * 1e-8, ray_dir)
        inv_dir = 1.0 / safe_dir
        t_near = (bound_min - origin) * inv_dir
        t_far = (bound_max - origin) * inv_dir
        t_exit = torch.min(torch.max(t_near, t_far), dim=-1)[0]

        # Project to boundary, keeping just inside.
        projected_world = origin + (t_exit.unsqueeze(-1) * 0.9999) * ray_dir

        # Back to sensor coordinates.
        projected_sensor = (projected_world - t_s2w) @ R_s2w

        # Write projected points back, and patch the world-frame cache in lock-step so subsequent reads stay valid.
        full_mask = mask.clone()
        full_mask[mask] = oob_mask
        self.points[full_mask] = projected_sensor
        self._points_world[full_mask] = projected_world

    def _sample_indices(self, num_samples: int, ratio: float) -> torch.Tensor:
        """Pick `num_samples` (or `int(valid_mask.numel() * ratio)`) random flat indices into the valid set.

        Returns a 1D `(M,)` tensor of indices into `valid_mask.flatten()`. Callers index payload tensors whose leading
        dims match `valid_mask` via `payload.reshape(-1, payload.shape[-1])[indices]` -- the reshape is a no-op for
        `(N, ...)` payloads and folds the spatial dims for `(H, W, ...)`.
        """
        if num_samples <= 0:
            num_samples = int(self.valid_mask.numel() * ratio)
        indices = torch.nonzero(self.valid_mask.flatten(), as_tuple=True)[0]
        if len(indices) <= num_samples:
            return indices
        perm = torch.randperm(len(indices), device=indices.device)[:num_samples]
        return indices[perm]

    def sample_points(
        self,
        num_samples: int = -1,
        ratio: float = 0.25,
        to_world_frame: bool = True,
        device: str | None = None,
    ) -> torch.Tensor:
        """Randomly sample valid points from this frame.

        Args:
            num_samples: number of samples to sample; if `<= 0`, fall back to
                `int(valid_mask.numel() * ratio)`.
            ratio: fraction of valid-mask cells to sample when `num_samples <= 0`.
            to_world_frame: if True, transform sampled points to world coordinates.
            device: optional device to move points to before the optional world-frame transform.

        Returns:
            `(n_sampled, 3)` sampled points in sensor or world coordinates.
        """
        sampled_indices = self._sample_indices(num_samples, ratio)
        src = self.points_world if to_world_frame else self.points
        # Flatten the spatial dims so 1D masks (lidar/vl) and 2D masks (depth) share the same indexing path.
        points = src.reshape(-1, 3)[sampled_indices]
        if device is not None:
            points = points.to(device)
        return points

    def sample_rays(
        self,
        num_samples: int = -1,
        ratio: float = 0.25,
        to_world_frame: bool = True,
        device: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample rays originating from this frame.

        Indexing mirrors :meth:`sample_points`: `_sample_indices` produces 1D indices into the flattened
        `valid_mask`, and `rays_d` / depth are flattened on their spatial dims before indexing, so the
        same path works for 1D (LiDAR) and 2D (depth) layouts.

        Args:
            num_samples: number of rays to sample; if `<= 0`, fall back to
                `int(valid_mask.numel() * ratio)`.
            ratio: fraction of valid-mask cells to sample when `num_samples <= 0`.
            to_world_frame: if True, return ray origins / directions in world coordinates via
                `self.ref_pose`; otherwise origins are zero (sensor frame) and directions are the
                stored camera-frame rays.
            device: optional device to move the returned tensors to.

        Returns:
            Tuple `(rays_o, rays_d, depth)` aligned row-for-row.

            - `rays_o`: `(n_sampled, 3)` ray origins (sensor translation broadcast when
                `to_world_frame=True`, else zeros).
            - `rays_d`: `(n_sampled, 3)` ray directions; unit-z in camera frame for `DepthFrame`,
                unit-norm in camera frame for `LiDARFrame`.
            - `depth`: `(n_sampled,)` per-ray depth metric (camera-z for depth, `||p||_2` for LiDAR).
        """
        sampled_indices = self._sample_indices(num_samples, ratio)
        rays_d = self.get_rays_direction().reshape(-1, 3)[sampled_indices]
        depth = self.get_depth().reshape(-1)[sampled_indices]
        if to_world_frame:
            pose = self.ref_pose.to(rays_d.device)
            rays_d = rays_d @ pose[:3, :3].T
            rays_o = pose[:3, 3].view(1, 3).expand_as(rays_d)
        else:
            rays_o = torch.zeros_like(rays_d)
        if device is not None:
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            depth = depth.to(device)
        return rays_o, rays_d, depth


class DepthFrame(Frame):
    def __init__(
        self,
        fid: int,
        depth: torch.Tensor,
        intrinsic: torch.Tensor,
        ref_pose: torch.Tensor,
        min_range: float | None = 0.0,
        max_range: float | None = None,
        device: str | None = None,
    ) -> None:
        """
        Args:
            fid: int, frame idx
            depth: (H, W) in meter; used only during construction to backproject into `self.points` and not stored
                (read it back via `get_depth()`, which returns `self.points[..., 2]`).
            intrinsic: (3, 3) intrinsic matrix used to build `self.rays_d`; not stored on the frame.
            ref_pose: (4, 4) reference pose in world coordinates
            min_range: lower bound on per-pixel depth (camera z) in meters; defaults to 0.0 so zero-depth (missing)
                pixels are dropped. Pass `None` to keep them.
            max_range: upper bound on per-pixel depth in meters; `None` (default) means no upper cap.
            device: str, device to put the tensors on
        """
        super().__init__(fid, ref_pose, min_range=min_range, max_range=max_range, is_depth=True)
        if not isinstance(depth, torch.Tensor):
            depth = torch.FloatTensor(depth)  # / 2

        # Reuse the shared depth_to_camera_points helper for the (H, W, 3) camera-frame backprojection.
        # rays_d is the same backprojection at unit depth (z == 1), so self.points[..., 2] equals the input depth
        # and `get_depth()` can derive depth from `self.points` without a separate copy.
        self.points = depth_to_camera_points(depth, intrinsic)  # (H, W, 3) in camera coordinates
        self.rays_d = depth_to_camera_points(torch.ones_like(depth), intrinsic)  # (H, W, 3) unit-depth rays
        self.valid_mask: torch.Tensor = self._compute_range_mask()

        if device is not None:
            self.ref_pose = self.ref_pose.to(device)
            self.rays_d = self.rays_d.to(device)
            self.points = self.points.to(device)
            self.valid_mask = self.valid_mask.to(device)

    def get_rays_direction(self):
        return self.rays_d

    def get_depth(self):
        # rays_d.z == 1, so points[..., 2] is the per-pixel depth.
        return self.points[..., 2]


class LiDARFrame(Frame):
    def __init__(
        self,
        fid: int,
        pointcloud: torch.Tensor,
        ref_pose: torch.Tensor,
        min_range: float | None = 0.0,
        max_range: float | None = None,
        device: str | None = None,
    ) -> None:
        """Build a LiDAR frame from a point cloud and the sensor pose.

        Args:
            fid: frame index used as the timestamp / id.
            pointcloud: (N, 3) point cloud in the sensor (LiDAR) frame.
            ref_pose: (4, 4) sensor-to-world pose; reshaped from (16,) if flat.
            min_range: lower bound on `||p||_2` in meters; defaults to `0.0` to drop zero/origin returns. Pass
                `None` to keep them.
            max_range: optional upper bound on `||p||_2` in meters; `None` (default) means no upper cap.
            device: optional device to move all tensors to before storage.
        """
        super().__init__(fid, ref_pose, min_range=min_range, max_range=max_range, is_depth=False)
        self.points = pointcloud
        self.rays_d: torch.Tensor = self._get_rays()  # (N, 3) in sensor frame coordinates

        self.valid_mask: torch.Tensor = self._compute_range_mask()

        if device is not None:
            self.points = self.points.to(device)
            self.ref_pose = self.ref_pose.to(device)
            self.rays_d = self.rays_d.to(device)
            self.valid_mask = self.valid_mask.to(device)

    def get_depth(self):
        return torch.norm(self.points, dim=-1)  # (N,)

    @torch.no_grad()
    def _get_rays(self):
        rays_d = torch.nn.functional.normalize(self.points, p=2, dim=-1)
        return rays_d

    def get_rays_direction(self):
        return self.rays_d


class VlFrame(Frame):
    """Frame for a vision-language dataset: per-point coordinates, per-point VL features, and pose.

    Mirrors :class:`LiDARFrame`'s contract -- points are stored in the sensor (camera) frame and converted to world
    coordinates on demand via `ref_pose`, so a `VlFrame` slots into the same `frame.get_points(to_world_frame=True,
    device=...)` call sites used by `SdfTrainer` / `OccTrainer`. The extra payload is `vl_features`, a `(N, C)` per-
    point feature tensor aligned with `points` (e.g. CLIP features sampled at each valid pixel of the source depth
    map and feature grid).

    Construct from already-backprojected points + per-point features. The VL pipeline's natural data flow is:
    `depth + intrinsic + features (C, H, W)` -> backproject + reshape -> `(N, 3) camera-frame points` +
    `(N, C) per-point features` -> `VlFrame(...)`. See `oren.utils.depth_utils.depth_to_world_points` for
    the backprojection (call it with an identity pose to keep points in the camera frame).
    """

    def __init__(
        self,
        fid: int,
        points: torch.Tensor,
        vl_features: torch.Tensor,
        ref_pose: torch.Tensor,
        min_range: float | None = 0.0,
        max_range: float | None = None,
        device: str | None = None,
    ) -> None:
        """Build a VL frame from precomputed camera-frame points + per-point features + pose.

        Args:
            fid: frame index used as the timestamp / id.
            points: `(N, 3)` point cloud in the sensor (camera) frame; align with `vl_features` row-for-row.
            vl_features: `(N, C)` per-point feature vectors. `C` is the field's feature dim.
            ref_pose: `(4, 4)` sensor-to-world pose. Flat 16-element tensors are reshaped.
            min_range: lower bound on `||p||_2` in meters; defaults to `0.0` to drop zero/origin entries from
                the backprojection. Pass `None` to keep them.
            max_range: optional upper bound on `||p||_2` in meters; `None` (default) means no upper cap.
            device: optional device to move all tensors to before storage.
        """
        if not isinstance(points, torch.Tensor):
            points = torch.as_tensor(points, dtype=torch.float32)
        if not isinstance(vl_features, torch.Tensor):
            vl_features = torch.as_tensor(vl_features, dtype=torch.float32)
        assert (
            points.ndim == 2 and points.shape[-1] == 3
        ), f"VlFrame.points must be (N, 3); got shape {tuple(points.shape)}"
        assert vl_features.ndim == 2 and vl_features.shape[0] == points.shape[0], (
            f"VlFrame.vl_features must be (N, C) with N matching points; got {tuple(vl_features.shape)} "
            f"vs points {tuple(points.shape)}"
        )

        super().__init__(fid, ref_pose, min_range=min_range, max_range=max_range, is_depth=False)
        self.points = points
        self.vl_features = vl_features

        # `apply_bound` narrows this further without touching `points`.
        self.valid_mask: torch.Tensor = self._compute_range_mask()

        if device is not None:
            self.points = self.points.to(device)
            self.vl_features = self.vl_features.to(device)
            self.ref_pose = self.ref_pose.to(device)
            self.valid_mask = self.valid_mask.to(device)

    def get_vl_features(self, valid_only: bool = True, device: str | None = None) -> torch.Tensor:
        """Return per-point VL features, optionally filtered by `valid_mask` and moved to `device`.

        Args:
            valid_only: if True, drop rows masked out by `valid_mask` (e.g. via `apply_bound`).
            device: optional device to move the returned tensor to.

        Returns:
            `(N, C)` per-point feature tensor whose rows align with `get_points(...)` for the same `valid_only`.
        """
        feats = self.vl_features[self.valid_mask] if valid_only else self.vl_features
        if device is not None:
            feats = feats.to(device)
        return feats

    def sample_points_and_features(
        self,
        num_samples: int = -1,
        ratio: float = 0.25,
        to_world_frame: bool = True,
        device: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same sampling as :meth:`sample_points`, but also returns the per-sample VL features.

        Returns:
            Tuple `(points (n_sampled, 3), features (n_sampled, C))` aligned row-for-row. Features are *not* affected
            by `to_world_frame` (features are pose-invariant in world space the same way they are in camera space).
        """
        sampled_indices = self._sample_indices(num_samples, ratio)
        src = self.points_world if to_world_frame else self.points
        # `points` and `vl_features` share the same flat layout (N == valid_mask.numel()), so the same 1D indices work.
        points = src.reshape(-1, 3)[sampled_indices]
        feats = self.vl_features[sampled_indices]
        if device is not None:
            points = points.to(device)
            feats = feats.to(device)
        return points, feats
