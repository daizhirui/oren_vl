Custom CUDA Kernels for Fuser Gather / Scatter
==============================================

Phase 1 of the VL fusion plan (see `DESIGN-VL-Fusion.md`) landed the Python Fuser API in
`oren/field_fusion.py`. The eager implementation is correct and differentiable, but the gather hot
path materializes 5-6 intermediate tensors of shape `(N, 8, *)` per call, which costs both kernel
launches and global-memory traffic. This document spec's a drop-in custom CUDA replacement for the
gather (and optionally scatter) bodies on `_CornerWeightFuser` and `TrilinearFuser`.

The Python API stays the same; only the body of `gather` (and `_CornerWeightFuser.scatter` if we
choose to fuse it) routes through an `autograd.Function` whose forward/backward call into CUDA.


## What we are fusing

### Non-GA gather (`_CornerWeightFuser.gather`)

Eager body, lines reproduced for reference:

```python
weights = self._compute_weights(level_geom)               # (N, 8)
valid   = self._corner_valid_mask(level_geom, values.shape[0])
weights = weights * valid.to(weights.dtype)
norm    = weights.sum(dim=-1, keepdim=True).clamp(min=tiny)
weights = weights / norm
gathered = values[level_geom.vertex_indices.long()]       # (N, 8, D)
return torch.einsum("ni,nik->nk", weights.to(values.dtype), gathered)  # (N, D)
```

Five intermediates of shape `(N, 8)` or `(N, 8, D)` get written to and read back from global
memory, plus 6+ kernel launches. The whole thing reduces to one thread per query point, with the
8 corners and their weights computed in registers.

Per-point math:

    result[n, k] = sum_{i in 0..7} w_i * mask_i * values[vidx[n,i], k] / sum_i w_i * mask_i

where `w_i` is computed by the subclass's per-corner kernel:

    Trilinear        w_i = prod_d (offsets01_flip_i[d] + voxel_offsets[n,d] * offsets11_i[d])
    InverseDistance  w_i = 1 / (||p_n - corner_i|| + epsilon)
    Rbf              w_i = exp(-||p_n - corner_i||^2 / (2 * bandwidth^2))
    OrnsteinU.       w_i = exp(-||p_n - corner_i|| / bandwidth)

`corner_i = voxel_center + 0.5 * voxel_size * resolution * offsets11_i` is also computed in
registers.

### GA gather (`TrilinearFuser.gather` with `grads is not None`)

Eager body:

```python
vidx          = level_geom.vertex_indices.long()
vertex_values = values[vidx].squeeze(-1)                  # (N, 8)
vertex_grad   = grads[vidx]                               # (N, 8, 3)
corners       = self._corner_positions(level_geom)        # (N, 8, 3)
diffs         = level_geom.points.unsqueeze(1) - corners  # (N, 8, 3)
projection    = torch.einsum("nik,nik->ni", vertex_grad, diffs)
augmented     = vertex_values + projection                # (N, 8)
weights       = self._trilinear_weights(level_geom.voxel_offsets)
return (weights * augmented).sum(dim=1, keepdim=True)     # (N, 1)
```

Per-point math:

    result[n] = sum_{i in 0..7} w_i * (vertex_values[i] + dot(vertex_grad[i], p_n - corner_i))

with the same `w_i` as the trilinear non-GA path. No validity mask is applied here (matches the
existing `ga_trilinear` semantics in the legacy code); -1 vertex indices wrap to the last row and
the caller is expected to mask its loss on points with any missing corner.

### Scatter (`_CornerWeightFuser.scatter`)

Lower priority for fusion since it runs once per frame, not per training step. If we eventually
fuse it: one thread per (point, corner) pair, atomic-add into `weight_sum` and the weighted-feature
accumulator. The streaming-mean update math afterward (the `numer / denom` block) is per-touched-
vertex and can stay in PyTorch. Decision: leave scatter eager for now; revisit if profiling shows
it as a bottleneck.


## Kernel design

### Forward kernel (one per fuser subclass via template)

Signature roughly:

```cpp
template <typename WeightFn>
__global__ void fuser_gather_forward(
    const int64_t* __restrict__ vertex_indices,   // (N, 8)
    const float*   __restrict__ values,           // (V, D)
    const float*   __restrict__ points,           // (N, 3)
    const float*   __restrict__ voxel_centers,    // (N, 3)
    const float*   __restrict__ voxel_sizes,      // (N, 1)
    const float*   __restrict__ voxel_offsets,    // (N, 3) in [0,1]^3
    const float3*  __restrict__ offsets11,        // (8,)   constant
    const float3*  __restrict__ offsets01_flip,   // (8,)   constant; only Trilinear uses it
    float          resolution,
    typename WeightFn::Params weight_params,      // bandwidth / epsilon / etc.
    int            N,
    int            V,
    int            D,
    float*         __restrict__ out,              // (N, D)
    bool           apply_valid_mask               // True for non-GA, False for GA path
);
```

One thread per query point `n`. Inner loop:

```cpp
float w_sum = 0.0f;
float accum[D_MAX] = {0};   // D is small (1 for SDF/OCC, up to 512 for VL)
for (int i = 0; i < 8; ++i) {
    int64_t vidx = vertex_indices[n * 8 + i];
    bool valid = !apply_valid_mask || (vidx >= 0 && vidx < V);
    int64_t vidx_safe = valid ? vidx : 0;        // wrap-around is fine even without clamp;
                                                 // mask zeroes contribution either way
    float3 corner = voxel_centers[n] + 0.5f * voxel_sizes[n] * resolution * offsets11[i];
    float  w_i    = WeightFn::compute(
                        points[n], corner, voxel_offsets[n],
                        offsets11[i], offsets01_flip[i], weight_params);
    if (!valid) w_i = 0.0f;
    w_sum += w_i;
    for (int k = 0; k < D; ++k) accum[k] += w_i * values[vidx_safe * D + k];
}
float inv = (w_sum > tiny) ? (1.0f / w_sum) : 0.0f;
for (int k = 0; k < D; ++k) out[n * D + k] = accum[k] * inv;
```

For the GA path (no normalization, weights from trilinear, augmented values), the same template
specializes on a different `WeightFn` and replaces the `values[vidx_safe * D + k]` read with
`(values[vidx_safe] + dot(grads[vidx_safe], points[n] - corner))`. Or simpler: write a separate
`fuser_ga_gather_forward` kernel; the bodies share little enough that a template adds clutter.

`WeightFn` policies (one struct per subclass):

```cpp
struct TrilinearWeights {
    struct Params {};
    __device__ static float compute(float3 p, float3 c, float3 vo,
                                    float3 off11, float3 off01_flip, Params) {
        return (off01_flip.x + vo.x * off11.x)
             * (off01_flip.y + vo.y * off11.y)
             * (off01_flip.z + vo.z * off11.z);
    }
};

struct RbfWeights {
    struct Params { float inv_two_h2; };
    __device__ static float compute(float3 p, float3 c, float3 vo,
                                    float3 off11, float3 off01_flip, Params pr) {
        float3 d = p - c;
        return __expf(-(d.x*d.x + d.y*d.y + d.z*d.z) * pr.inv_two_h2);
    }
};
// InverseDistanceWeights, OrnsteinUhlenbeckWeights similar.
```

NVCC inlines `WeightFn::compute` per instantiation, so there is no runtime dispatch cost.

### Backward kernel

`gather` is differentiable in `values` (and in `grads` for the GA path). Weights are not
differentiable - they depend only on geometry tensors (`voxel_offsets`, `voxel_centers`,
`voxel_sizes`, `points`) which carry no autograd. So backward only needs to scatter `grad_out`
through the gather pattern.

Non-GA backward:

    dL/dvalues[vidx[n,i], k] += w_i_norm * grad_out[n, k]      (atomic add)

GA backward:

    dL/dvalues[vidx[n,i]]   += w_i * grad_out[n]                                   (atomic add)
    dL/dgrads[vidx[n,i], j] += w_i * (points[n,j] - corner_i[j]) * grad_out[n]     (atomic add)

Two kernel-design choices:

1. **Atomic adds (recommended).** One thread per (point, corner) pair, or one thread per point
   that loops 8 times. Float32 atomicAdd is fast on modern GPUs; contention is bounded by overlap
   among 8 corners per point, which is low in practice (only points in the same voxel collide on
   the same vertex). Non-deterministic ordering across runs.

2. **Sort-then-segment-reduce.** Deterministic but needs an extra sort pass via CUB / thrust.
   Worth it only if bitwise reproducibility is a hard requirement.

Pick atomic adds unless you specifically need determinism. PyTorch's stock `index_add` is
non-deterministic too, so we are not regressing relative to the eager baseline.

**Weights in backward**: do not save them on `ctx`. Recompute per thread in the backward kernel
from the same geometry inputs that forward used. Trilinear weights are 3 muls + 3 adds + 1
3-product; RBF/OU/InverseDistance involve one transcendental per corner via `__expf` /
`__frsqrt_rn`. All cheap relative to the global-memory traffic for the gradients.

Saved tensors (`ctx.save_for_backward`): just the geometry inputs forward already needs
(`vertex_indices`, `points`, `voxel_centers`, `voxel_sizes`, `voxel_offsets`) plus `values` shape
metadata. No `(N, 8)` activations.

### Float16 / bfloat16

Recommend: always accumulate in float32, regardless of `values` dtype. The atomicAdd story
matters here:

- Float32 atomicAdd: native on all modern GPUs.
- Bfloat16 atomicAdd: native on Hopper+ only; emulated (slower) on Ampere.
- Float16 atomicAdd: native on Ampere+, but precision is lousy for accumulating thousands of
  contributions per vertex.

Cast `grad_out` to float32 at kernel entry, accumulate into float32 atomic buffers, cast back at
exit. The extra cast is one extra memory pass; negligible compared to the atomicAdd traffic.
Cleaner numerics, fewer device-capability headaches.


## Wiring with `autograd.Function`

```python
class _CornerWeightGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, level_geom_tensors, values, weight_params, apply_valid_mask):
        out = fuser_kernels.gather_forward(*level_geom_tensors, values, weight_params,
                                           apply_valid_mask)
        ctx.save_for_backward(*level_geom_tensors, values)
        ctx.weight_params = weight_params
        ctx.apply_valid_mask = apply_valid_mask
        return out

    @staticmethod
    def backward(ctx, grad_out):
        *level_geom_tensors, values = ctx.saved_tensors
        grad_values = fuser_kernels.gather_backward(
            *level_geom_tensors, values.shape, ctx.weight_params,
            ctx.apply_valid_mask, grad_out.contiguous())
        return None, grad_values, None, None
```

Hooking it into the existing class (no Python API change):

```python
class _CornerWeightFuser(Fuser):
    def gather(self, level_geom, values):
        if not _kernels_available():
            return self._gather_eager(level_geom, values)   # fallback / debug
        return _CornerWeightGather.apply(
            (level_geom.vertex_indices, level_geom.points, level_geom.voxel_centers,
             level_geom.voxel_sizes, level_geom.voxel_offsets, self._offsets11, self._offsets01_flip),
            values,
            self._weight_params(),
            True,   # apply_valid_mask for non-GA
        )
```

Each subclass exposes its `_weight_params()` to populate the `WeightFn::Params` struct passed
into the kernel. `TrilinearFuser._weight_params()` is empty; `RbfFuser._weight_params()` returns
`{"inv_two_h2": 1.0 / (2 * self.bandwidth ** 2)}` and so on.

The GA path follows the same shape but routes through a separate `_GATrilinearGather` Function
that also captures `grads` and returns gradients for both `values` and `grads`.


## Validation plan

1. **Per-fuser correctness**: for each of the five subclasses, run a single-point scatter-then-
   gather round-trip in eager and in the kernel; expect bit-equal up to float rounding.
2. **`torch.autograd.gradcheck`** on small inputs (N ~ 16, V ~ 64, D = 1, 3, 512) against the
   eager implementation. This is the single most important regression test for the backward.
3. **Determinism control**: if atomics are used, verify the eager and kernel forward agree
   bitwise but accept slight backward differences (atomic ordering). Sort-based backward only if
   step 2 reveals unacceptable noise.
4. **Mixed precision**: gradcheck under float32, float16, bfloat16 if any of those are used in
   training. Confirm the float32 accumulation strategy keeps gradients accurate.
5. **Resize interaction**: after `octree.insert_points` grows `values` (and the fuser's
   `weight_sum`), kernel calls on the new larger buffers must keep working. Eager already
   handles this; just ensure the kernel reads `values.shape[0]` from the tensor at every call,
   not from a cached value.


## Benchmark plan

Three configurations on a representative training step:

| label | what | engineering cost |
| --- | --- | --- |
| baseline | current eager | 0 |
| compiled | `torch.compile(fuser.gather, dynamic=True)` | ~5 min, plus 0.5-2 s first-call compile |
| kernel | custom CUDA + autograd.Function | days |

Profile each with `nsys profile` + `torch.profiler`. Key numbers to record per step:

- Wall-clock per training step.
- Kernel time (CUDA only) for `gather` calls.
- Memory peak.
- Backward pass time for the gather call site.

Vary `N` (query points per call) across {1k, 10k, 100k, 1M} to see where launch-overhead-bound
crosses over to memory-bound. Custom CUDA should dominate at small `N`; `torch.compile` likely
matches it at large `N`.

If the kernel beats compile by less than ~2x at the working batch size, the maintenance cost
probably outweighs the win. Reconsider whether the kernel is worth shipping.


## Priority

Implement in this order:

1. **Non-GA `_CornerWeightFuser.gather`** with `TrilinearWeights`. Most-called path; backward is
   the cleaner of the two. Validates the autograd.Function plumbing.
2. **GA `TrilinearFuser.gather` (grads != None)**. Same kernel shell, extended math, two output
   gradient tensors.
3. **Other corner-weight subclasses** (`InverseDistance`, `Rbf`, `OrnsteinUhlenbeck`). Mostly
   template instantiations of step 1 once the policy plumbing is in place.
4. **Scatter** (`_CornerWeightFuser.scatter`). Only if profiling justifies it.
5. **NearestVertexFuser**: leave eager. Its gather is a single index op already; nothing to fuse.


## Open questions to revisit

- **Where does the kernel live?** Likely a new `oren/csrc/field_fusion/` directory with a
  `setup.py`-style extension, similar to how `erl_geometry` is built. Could also live as a
  pybind11 extension shipped with the `oren` package; depends on whether we want optional GPU
  build.
- **Build-time gating**: support an eager-only mode for users without CUDA / without the
  extension built. Already sketched above with `_kernels_available()` fallback.
- **D > 1 register pressure**: the per-thread accumulator `accum[k]` blows out registers for
  large `D` (VL features at 512). For those, fall back to per-corner streaming writes or chunked
  accumulation. The trade-off is per-D-range tuning. SDF/OCC at D=1 is the easy and most
  important case.
- **TF32**: enable `__cublas_set_math_mode(CUBLAS_TF32_TENSOR_OP_MATH)`-style behavior in the
  kernel? Probably not - the matmul-like reduction is small enough that TF32 brings no benefit.
