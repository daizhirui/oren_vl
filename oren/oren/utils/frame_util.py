from typing import Any, Callable, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from oren.frame import Frame


def multiple_max_set_coverage(
    kf_seen_voxel_num: list,
    kf_voxel_indices: list,
    kf_unoptimized_voxels: Optional[torch.Tensor],
    kf_all_voxels: Optional[torch.Tensor],
    num_selections: int,
    num_voxels: int,
    device: str,
):
    """
    Overwrite all voxels contained in the keyframe multiple times

    Note that there is a related implementation outside of this function. In the insert_keyframe class method in
    mapping, The function is to add the new voxels contained in each newly added key frame to the set of unoptimized
    voxels. Cover all the voxels contained in the key frame multiple times.
    Args:
        kf_seen_voxel_num (list): This is a list, each element is the number of the corresponding voxels contained in
                                    the keyframe in key_frames.
        kf_voxel_indices (list): indices of voxels contained in each keyframe.
        kf_unoptimized_voxels (tensor, N + 1): mask of all unoptimized voxels, N=max number of voxels.
        kf_all_voxels (tensor, N + 1): mask of all voxels to be optimized.
        num_selections (int): Number of keyframes to be selected.
        num_voxels (int): Number of total voxels in the octree.
        device (str): device to run the computation.
    Returns:
        selected_frame_indices (list): indices of selected keyframes.
        kf_unoptimized_voxels (tensor, N + 1): mask of unoptimized voxels after selection.
        kf_all_voxels (tensor, N + 1): mask of all voxels to be optimized.
    """

    cnt = 0
    selected_frame_indices = []

    padded_tensor = pad_sequence(kf_voxel_indices, batch_first=True, padding_value=-1)
    padded_tensor = padded_tensor.long().to(device)  # (B, M)
    if kf_unoptimized_voxels is None:
        kf_unoptimized_voxels = torch.zeros(num_voxels + 1, dtype=torch.bool).to(device)  # unoptimized voxels
        kf_all_voxels = torch.zeros(num_voxels + 1, dtype=torch.bool).to(device)  # All voxels to be optimized
        # kf_optimized_voxels = torch.zeros(num_voxels + 1, dtype=torch.bool).to(device)

        kf_seen_voxel_num = torch.tensor(kf_seen_voxel_num)  # (B), on CPU
        value, index = torch.max(kf_seen_voxel_num, dim=0)
        selected_frame_indices.append(index.item())

        kf_unoptimized_voxels.index_fill_(0, padded_tensor.view(-1), True)
        kf_unoptimized_voxels[-1] = False

        voxel_indices = kf_voxel_indices[index].long()
        kf_unoptimized_voxels.index_fill_(0, voxel_indices.view(-1).to(device), False)

        cnt += 1

    kf_all_voxels.index_fill_(0, padded_tensor.view(-1), True)
    kf_all_voxels[-1] = False

    while cnt < num_selections:
        result_num = torch.sum(kf_unoptimized_voxels[padded_tensor].long(), dim=-1)  # (B)
        value, index = torch.max(result_num, dim=0)
        selected_frame_indices.append(index.item())

        voxel_indices = kf_voxel_indices[index].long().view(-1).to(device)
        kf_unoptimized_voxels.index_fill_(0, voxel_indices, False)

        cnt += 1

        if not kf_unoptimized_voxels.any():  # If all are optimized

            # Unoptimized voxels = all voxels that need to be optimized - voxels seen by the latest selected key frame.
            kf_unoptimized_voxels[...] = kf_all_voxels
            kf_unoptimized_voxels.index_fill_(0, voxel_indices, False)

    return selected_frame_indices, kf_unoptimized_voxels, kf_all_voxels


def _default_collate_fn(batch) -> Any:
    assert batch, "Batch is empty"
    if isinstance(batch[0], torch.Tensor):
        return torch.cat(batch, dim=0)
    elif isinstance(batch[0], (list, tuple)):
        transposed = list(zip(*batch))
        # Preserve container type: tuple -> tuple, list -> list, so callers that unpack a fixed-arity
        # tuple from the per-frame fn keep the same shape on the collated side.
        return type(batch[0])(_default_collate_fn(samples) for samples in transposed)
    elif isinstance(batch[0], dict):
        collated_dict = {}
        for key in batch[0].keys():
            collated_dict[key] = _default_collate_fn([d[key] for d in batch])
        return collated_dict
    else:
        raise TypeError(f"Unsupported data type in batch: {type(batch[0])}")


def sample_from_frames(
    frames: list[Frame],
    sample_frame_fn: Callable[[Frame, int], Any],
    sample_frame_fn_kwargs: dict | list[dict] | None = None,
    collate_fn: Callable | None = None,
) -> Any:
    """Sample from the frames using the provided sampling function
    and collate the results using the provided collate function.

    Args:
        frames: list of frames to sample from.
        sample_frame_fn: function that takes a frame and number of samples, and returns sampled data from the frame.
        sample_frame_fn_kwargs: keyword arguments to pass to the sample_frame_fn.
        collate_fn: function that takes a list of sampled data from each frame and collates them into tensor(s).
    Returns:
        Collated tensor(s) containing the sampled data from all frames.
    """
    if not frames:
        return None
    if collate_fn is None:
        collate_fn = _default_collate_fn
    if sample_frame_fn_kwargs is None:
        sample_frame_fn_kwargs = [{}] * len(frames)
    elif isinstance(sample_frame_fn_kwargs, dict):
        sample_frame_fn_kwargs = [sample_frame_fn_kwargs] * len(frames)
    assert len(sample_frame_fn_kwargs) == len(frames), "Length of sample_frame_fn_kwargs must match length of frames"
    sampled_data = [sample_frame_fn(frame, **kwargs) for frame, kwargs in zip(frames, sample_frame_fn_kwargs)]
    return collate_fn(sampled_data)
