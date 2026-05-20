import torch


def depth_to_camera_points(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Backproject a depth map to camera-frame points.

    Pixel centers are at `(u + 0.5, v + 0.5)`, matching the resampling convention used by
    `generate_vl_features`. Callers that need to drop zero-depth (or otherwise invalid) pixels
    compute their own mask -- typically `depth > 0` -- and apply it to the returned grid.

    Args:
        depth: (H, W) meters.
        K: (3, 3) intrinsics at the depth scale.

    Returns:
        camera_points: `(H, W, 3)`.
    """
    H, W = depth.shape
    device = depth.device
    dtype = depth.dtype
    u = torch.arange(W, device=device, dtype=dtype) + 0.5
    v = torch.arange(H, device=device, dtype=dtype) + 0.5
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (uu - cx) * depth / fx
    y_cam = (vv - cy) * depth / fy
    z_cam = depth
    return torch.stack([x_cam, y_cam, z_cam], dim=-1)


def depth_to_world_points(
    depth: torch.Tensor,
    pose: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """Backproject a depth map to world-space points.

    Thin wrapper around :func:`depth_to_camera_points` that applies `pose` after the camera-frame
    backprojection. Kept for callers that already operate on world-frame points and don't
    construct a frame object.

    Args:
        depth: (H, W) meters.
        pose: (4, 4) cam->world (T_wc).
        K: (3, 3) intrinsics at the depth scale.

    Returns:
        world_points: `(H, W, 3)`.
    """
    pts_cam = depth_to_camera_points(depth, K)
    return pts_cam @ pose[:3, :3].T + pose[:3, 3]
