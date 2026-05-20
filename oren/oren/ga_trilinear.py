import torch


def get_vertex_offsets():
    """Build the canonical (1, 8, 3) tensor of voxel-vertex offsets in `{-1, 1}^3` (big-endian ordering).

    Returns:
        offsets: (1, 8, 3) tensor of voxel-vertex offsets in `{-1, 1}^3`.
    """
    cut = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    xx, yy, zz = torch.meshgrid(cut, cut, cut, indexing="ij")  # big-endian
    offsets = torch.stack([xx, yy, zz], dim=-1).reshape(1, 8, 3)  # (1,8,3)
    return offsets


vertex_offsets1 = get_vertex_offsets()  # (1,8,3), {-1, 1}^3
vertex_offsets2 = vertex_offsets1 * 0.5 + 0.5  # (1,8,3), {0, 1}^3
vertex_offsets3: torch.Tensor = 1 - vertex_offsets2  # flipped version of offsets2
is_little_endian = False


def get_vertices(voxel_centers: torch.Tensor, voxel_sizes: torch.Tensor, resolution: float):
    """
    Get the 8 vertices of each voxel.
    Args:
        voxel_centers: (n_points, 3) metric center of the voxels
        voxel_sizes: (n_points, 1) grid size of the voxels
        resolution: float, the resolution of the voxel grid

    Returns:
        vertices: (n_points, 8, 3) metric coordinates of the 8 voxel corners for each input voxel.
    """
    global vertex_offsets1
    vertex_offsets1 = vertex_offsets1.to(voxel_centers.device)

    half_sizes = (voxel_sizes * 0.5).view(-1, 1, 1) * vertex_offsets1  # (n_points, 8, 3)
    vertices = voxel_centers.view(-1, 1, 3) + half_sizes * resolution  # (n_points, 8, 3)
    return vertices


def trilinear_interpolation(points: torch.Tensor, per_point_vertex_values: torch.Tensor, little_endian: bool):
    """
    Perform trilinear interpolation.
    Args:
        points: (n_points, 3) point coordinates relative to the voxel, in [0, 1]^3
        per_point_vertex_values: (n_points, 8, ...) values at the 8 vertices of the voxel containing each point
        little_endian: bool, whether the vertex ordering is little-endian. e.g. 1->(1,0,0). If False, big-endian is
            used.
    Returns:
        interpolated: (n_points, ...) interpolated values at the points
    """
    points = points.unsqueeze(1)  # (n_points, 1, 3)
    # (p * q + (1 - p) * (1 - q)).prod(dim=-1), where q in {0, 1}^3
    # = (1 - q - p + 2 * p * q).prod(dim=-1)
    # = (vertex_offsets3 - points + points * vertex_offsets2 * 2).prod(dim=-1)
    # = (vertex_offsets3 + points * (vertex_offsets2 * 2 - 1)).prod(dim=-1)
    # = (vertex_offsets3 + points * vertex_offsets1).prod(dim=-1)
    global vertex_offsets1, vertex_offsets3, is_little_endian
    vertex_offsets1 = vertex_offsets1.to(points.device)
    vertex_offsets3 = vertex_offsets3.to(points.device)
    if little_endian != is_little_endian:
        vertex_offsets1 = vertex_offsets1[:, :, [2, 1, 0]].contiguous()
        vertex_offsets3 = vertex_offsets3[:, :, [2, 1, 0]].contiguous()
        is_little_endian = little_endian
    weights = (vertex_offsets3 + points * vertex_offsets1).prod(dim=-1)  # (n_points, 8)
    feature_dims = per_point_vertex_values.shape[2:]
    per_point_vertex_values = per_point_vertex_values.view(points.shape[0], 8, -1)  # (n_points, 8, feature_dims)
    interpolated = torch.einsum("ni,nik->nk", weights, per_point_vertex_values)  # (n_points, feature_dims)
    interpolated = interpolated.view(points.shape[0], *feature_dims)  # (n_points, ...)
    return interpolated


def normalize_to_voxel_unit_cube(
    points: torch.Tensor,
    voxel_centers: torch.Tensor,
    voxel_sizes: torch.Tensor,
    resolution: float,
):
    """
    Normalize point coordinates to the unit cube of the voxel for trilinear interpolation.
    Args:
        points: (n_points, 3) point cloud in world coordinates
        voxel_centers: (n_points, 3) center of the voxel containing each point
        voxel_sizes: (n_points, 1) grid size of the voxel containing each point
        resolution: float, the resolution of the voxel grid
    Returns:
        p: (n_points, 3) coordinates of the points relative to the voxel, in [0, 1]^3
    """
    # voxel_sizes==0 means the voxel does not exist (caller indexed an out-of-bounds voxel_indices=-1 entry that
    # wrapped to a zero-initialized buffer row). Clamp to avoid div-by-zero -> inf -> NaN in finite-difference
    # gradients; callers mask out these positions in the loss anyway.
    # Comment out the following line because the C++ implementation try to ensure that voxel_indices=-1 entries have
    # size=1, so that the output will be finite and the gradients will not be NaN.
    # One known possible way to get zero voxel size is stale voxel_indices that pick a removed voxel, which is set to
    # size=0 in the C++ implementation.
    # If we get NaN, we may want to check if there are stale voxel_indices.
    # safe_sizes = voxel_sizes.clamp(min=1)

    p = (points - voxel_centers) / (voxel_sizes * resolution) + 0.5  # (n_points, 3)
    return p


def ga_trilinear(
    points: torch.Tensor,
    voxel_centers: torch.Tensor,
    voxel_sizes: torch.Tensor,
    resolution: float,
    vertex_values: torch.Tensor,
    vertex_grad: torch.Tensor | None = None,
    gradient_augmentation: bool = True,
    little_endian: bool = False,
    voxel_offsets: torch.Tensor | None = None,
):
    """
    Perform gradient-augmented trilinear interpolation.
    Args:
        points: (n_points, 3) point cloud in world coordinates
        voxel_centers: (n_points, 3) center of the voxel containing each point
        voxel_sizes: (n_points, 1) grid size of the voxel containing each point
        resolution: float, the resolution of the voxel grid
        vertex_values: (n_points, 8) values at the 8 vertices of the voxel containing each point
        vertex_grad: (n_points, 8, 3) gradient vectors at the 8 vertices of the voxel containing each point
        gradient_augmentation: bool, whether to use gradient-augmented trilinear interpolation
        little_endian: bool, whether the vertex ordering is little-endian. e.g. 1->(1,0,0).
            If False, big-endian is used.
        voxel_offsets: (n_points, 3) coordinates of the points relative to the voxel, in [0, 1]^3.
            If None, they will be computed.

    Returns:
        results: (n_points,) interpolated scalar values at the query points.
        voxel_offsets: (n_points, 3) the in-voxel offsets actually used (echoed back so the caller can reuse them).
    """

    if gradient_augmentation:
        with torch.no_grad():
            vertices = get_vertices(voxel_centers, voxel_sizes, resolution)  # (n_points, 8, 3)
        diffs = points.unsqueeze(1) - vertices  # (n_points, 8, 3)
        projection = torch.einsum("nik,nik->ni", vertex_grad, diffs)  # (n_points, 8)
        per_point_vertex_values = vertex_values + projection  # (n_points, 8)
    else:
        per_point_vertex_values = vertex_values  # (n_points, 8)

    if voxel_offsets is None:
        voxel_offsets = normalize_to_voxel_unit_cube(points, voxel_centers, voxel_sizes, resolution)  # (n_points, 3)
    results = trilinear_interpolation(voxel_offsets, per_point_vertex_values, little_endian=little_endian)
    return results, voxel_offsets
