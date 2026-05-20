"""In-place leading-dim resize for per-vertex `nn.Parameter` tensors.

Every FieldStorage / FeatureBank has one or more `(V, *)` parameters whose `V` axis tracks the octree's
`num_vertices`. When the octree's vertex count crosses a power-of-two boundary, the resize observer fires and these
parameters need to grow in-place.

The non-obvious bit is the *how*: only `Parameter.set_(new_tensor)` works.
- `param.data = new` updates the `Tensor`'s `sizes_` but leaves autograd's `AccumulateGrad.input_metadata` cached at
  the forward-time shape, which surfaces on the next backward as `Function ...Backward returned an invalid gradient at
  index 0 - got [new] but expected shape compatible with [old]`.
- `param.data.set_(new)` is a silent no-op for `nn.Parameter`s (the Parameter wrapper's own size metadata isn't
  refreshed).
- `Parameter.set_(new)` swaps the underlying storage atomically, refreshes autograd's input_metadata, AND preserves
  the `nn.Parameter` object identity (so optimizer `param_groups` references stay valid).
"""

import torch
import torch.nn as nn


@torch.no_grad()
def grow_param_first_dim(
    param: nn.Parameter,
    new_size: int,
    fill_value: float = 0.0,
) -> bool:
    """Grow `param` along its leading axis to `new_size` rows, in place.

    No-op when `new_size <= param.shape[0]` -- required to make `register_resize_observer`'s catch-up fire idempotent.
    Existing rows are copied verbatim; trailing dims of the new storage match `param.shape[1:]`.

    Args:
        param: The `nn.Parameter` to resize; its underlying storage is swapped via `Parameter.set_`.
        new_size: Desired length along axis 0.
        fill_value: Scalar used to initialise any newly added rows.

    Returns:
        True if the parameter was actually resized; False if `new_size <= param.shape[0]`.
    """
    if new_size <= param.shape[0]:
        return False
    old = param.detach()
    new = torch.full(
        (new_size, *old.shape[1:]),
        fill_value,
        dtype=old.dtype,
        device=old.device,
    )
    new[: old.shape[0]] = old
    param.set_(new)
    return True
