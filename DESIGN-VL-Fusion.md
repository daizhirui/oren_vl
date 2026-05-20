Vision-language Feature Fusion for 3D Scene Understanding
=========================================================

In this design, we propose a vision-language feature fusion approach for 3D scene understanding.
The main idea is to leverage the complementary information from both visual and language modalities
to enhance the performance of 3D scene understanding tasks.

The proposed approach consists of the following components:

1. **Visual-language feature extraction**: We use a pre-trained vision-language model (e.g., CLIP)
    to extract features from RGB-D images. We can also use other types of models (e.g. Utonia) to
    extract features from other types of data (e.g. point clouds). The extracted features are then
    fused to into a unified representation as a map. Here we develop based on our octree-based
    field storage framework.
2. **Feature fusion**: we need to design a fusion mechanism to update the map representation with the
    extracted features. There are several possible fusion mechanisms:
    1. weighted averaging:
       - find the nearest vertex in the map for each feature point and update the vertex feature by
         running mean, exponential moving average, or simply overwriting. This path is non-learnable
         (the nearest-vertex assignment is a hard discrete pick, so each vertex is only touched by
         the points assigned to it; no gradient smoothing across neighbors).
       - find the voxel containing each feature point and update the eight corners' features by
         weights computed from trilinear interpolation weights, inverse distance, or kernel function
         (e.g. RBF). This path supports both a direct-update mode (splat the weighted contributions
         into the eight corners) and a learnable mode: because the gather operation is differentiable
         in the per-vertex values, we can treat the extracted features' positions as query points,
         gather features back, and optimize the field's per-vertex values by reconstruction loss
         against the input features.
       - To query feature for a point, we can use the same mechanism to obtain the nearest vertex's
         feature or weighted average of the eight corners' features.
    2. attention-based fusion:
       - use the extracted features' positions as queries and the map vertices as keys and values to
         perform attention and update the map vertices' features. This needs to train transformer layers.
       - To query feature for a point, we can use the same attention mechanism to obtain the feature.
         Use the query point's position as query and the map vertices as keys and values to perform
         attention and obtain the feature.

## Implementation of Feature Fusion

Feature storage is ready for fusion. We need to implement the fusion mechanism support in a new
module, `oren/field_fusion.py`, that operates *on top of* `FieldStorage` without modifying it.
`FieldStorage` keeps its current responsibility (per-vertex state + per-point forward); fusers
are the policy layer that decides how new observations are written into that state.

### API

A `Fuser` abstract base class with two methods, `scatter` and `gather`. It subclasses `nn.Module`
so trainable variants (attention) fit the same interface as the no-learning ones.

```python
class Fuser(nn.Module):
    """Policy for writing point-wise features into a per-vertex parameter and reading them back.

    Subclasses implement one fusion mechanism each. Scatter and gather must use the same
    mechanism so a feature written in is recovered (within numerical error) by gathering at the
    same point.

    A Fuser is bound to one `SemiSparseOctree` at construction (so its state buffers can
    register a resize observer and grow in lockstep with `octree.sso.num_vertices`). The
    *target* tensor — whatever `(capacity, D)` `nn.Parameter` to read/write — is passed in at
    each call, not configured up front. This lets the same Fuser instance operate on
    `FieldStorage.values`, `FeatureBank.features`, or any other compatible per-vertex parameter
    on the same octree.
    """

    def __init__(self, octree: SemiSparseOctree, ...): ...

    def scatter(
        self,
        values: nn.Parameter,   # (capacity, D), modified in place
        points: torch.Tensor,   # (N, 3) world-space points
        feats: torch.Tensor,    # (N, D)
        level: int = 1,         # which octree level the vertex_indices come from
    ) -> None:
        """Update `values` in place from (points, feats) at the given level."""

    def gather(
        self,
        values: nn.Parameter,
        points: torch.Tensor,
        level: int = 1,
    ) -> torch.Tensor:          # (N, D)
        """Read features at points using the same mechanism as scatter."""
```

Both methods go through `self.octree.query(points)` to obtain an `OctreeGeometry`, then use
`geom.at_level(level)` for the leaf-voxel info. This shares the per-call geometry cache when
multiple fusers (or fuser invocations at different levels) act on the same forward pass.

#### Target parameters

The caller passes the actual `nn.Parameter` to read/write. Two natural targets exist on a field:

- `field.values`: shape `(capacity, field.cfg.output_dim)`, always level 1.
  - Requires `field.cfg.mode in {"explicit", "hybrid"}` (otherwise `field.values is None`).
  - Feature-dim contract: `feats.shape[-1] == field.cfg.output_dim`.
  - Gather math identical to `FieldStorage._explicit` on the trilinear path.
- `field.bank.features`: shape `(capacity, field.bank.feature_dim)`, level chosen by the caller
  in `[1, field.cfg.implicit_feature_level]`.
  - Requires `field.cfg.mode in {"implicit", "hybrid"}` and `field.bank is not None`.
  - Feature-dim contract: `feats.shape[-1] == field.bank.feature_dim` (set `bank.feature_dim=512`
    to ingest CLIP features directly).

Any other `(capacity, D)` parameter sized to the same octree works the same way — the Fuser only
needs the tensor and a vertex-index source. Each subclass implements its mechanism once; the
caller picks the target. State buffers (counts, weight_sum) size to `octree.capacity` from the
fuser's bound octree.

### Phase 1 subclasses (no learning)

1. `NearestVertexFuser(mode: Literal["overwrite", "running_average", "ema"], ema_alpha: float)`
    - **scatter**: for each point, locate the nearest vertex of its leaf voxel (the existing
      `vlocal` bit-pattern in `demo_semi_sparse_octree_vl.scatter_phase`) and update that vertex
      according to `mode`.
    - **gather**: read the nearest vertex's value directly (no interpolation).
    - Generalizes the current `scatter_phase` in the VL demo, which becomes a thin wrapper.
2. `TrilinearFuser`
    - **scatter**: splat each point's feature to all 8 corners of its leaf voxel, weighted by the
      trilinear interpolation coefficients. Uses `index_add_` to accumulate weighted contributions
      and a parallel weight accumulator for normalization. Scatter only writes to the `values`
      target; it never touches a `grads` tensor (a single scattered feature does not define a
      unique Hermite gradient update, and the GA path is purely a gather-side concern).
    - **gather**: trilinear interpolation of the 8 corner values. The gather signature accepts an
      optional `grads: nn.Parameter` argument of shape `(capacity, 3)`; when provided, switches to
      gradient-augmented (Hermite) trilinear interpolation via the existing `ga_trilinear` helper.
      With this, `TrilinearFuser.gather` becomes the canonical trilinear-gather implementation:
      `FieldStorage._explicit` can be refactored into a thin call into it (passing
      `self.values` and, when `cfg.gradient_augmentation`, `self.grads`), eliminating the
      duplicate interpolation code path that currently lives on the field.
3. `InverseDistanceFuser(epsilon: float = 1e-6)`
    - **scatter**: same 8-corner splat as `TrilinearFuser` but with inverse-distance weights
      `w_i = 1 / (||p - corner_i|| + epsilon)` computed in metric units. `epsilon` guards against
      division-by-zero when a point coincides with a corner.
    - **gather**: weighted sum of the 8 corner values with the same kernel weights, normalized by
      the weight sum.
4. `RbfFuser(bandwidth: float)`
    - **scatter**: same 8-corner splat as `TrilinearFuser` but with Gaussian RBF weights
      `w_i = exp(-||p - corner_i||^2 / (2 * bandwidth^2))` computed in metric units. `bandwidth`
      is a length scale in the same units as `octree.cfg.resolution`.
    - **gather**: weighted sum of the 8 corner values with the same kernel weights, normalized by
      the weight sum.
5. `OrnsteinUhlenbeckFuser(bandwidth: float)`
    - **scatter**: same 8-corner splat as `TrilinearFuser` but with Ornstein-Uhlenbeck (exponential
      / Matern-1/2) kernel weights `w_i = exp(-||p - corner_i|| / bandwidth)` — linear distance in
      the exponent, not squared. `bandwidth` is a length scale in the same units as
      `octree.cfg.resolution`. Compared to `RbfFuser`, the OU kernel has heavier tails (slower
      decay with distance).
    - **gather**: weighted sum of the 8 corner values with the same kernel weights, normalized by
      the weight sum.
    - **Learnable-mode caveat**: the kernel weight `w_i(p) = exp(-||p - corner_i|| / bandwidth)`
      is C^0 but not C^1 at each voxel vertex — the spatial gradient `nabla_p w_i` is undefined
      at `p = corner_i` (the distance function `||p - corner_i||` itself is non-differentiable
      there) and flips direction abruptly across the vertex. In learnable mode this gives the
      gathered output a non-smooth crease at every voxel vertex and produces unstable, direction-
      dependent gradients on `field.values` for query points sitting on or very near a vertex —
      a common case for points-on-surface data. For learnable use prefer `TrilinearFuser` or
      `RbfFuser`, both of which produce smoother reconstructions at vertices. Direct-update mode
      is unaffected (no gradient is taken through the kernel) and is the recommended use of
      `OrnsteinUhlenbeckFuser`.

#### State ownership

Fusers that need per-vertex auxiliary state own it themselves:

- `NearestVertexFuser` in `running_average` mode owns a `counts: (capacity,) long` buffer.
- `NearestVertexFuser` in `ema` mode is stateless beyond `ema_alpha` (the running update folds
  into `field.values` directly).
- `TrilinearFuser`, `InverseDistanceFuser`, `RbfFuser`, and `OrnsteinUhlenbeckFuser` own a
  `weight_sum: (capacity,) float` buffer for the online-normalization variant; in single-shot
  batched scatter they can normalize from a local tensor and skip the buffer.

Each stateful fuser registers a resize observer on its bound `octree` at construction so its
buffers grow in lockstep with `octree.sso.num_vertices`, mirroring how
`FieldStorage._on_octree_resize` already handles `values` and `grads` (and how
`FeatureBank.grow_to` handles `features`). This keeps `FieldStorage` and `FeatureBank` free of
accumulator state and lets multiple fusers compose against the same octree — including fusers
operating on `field.values` and `field.bank.features` of the same field — without stepping on
each other.

Optimizer/state migration is *not* needed for Phase 1 fusers - their buffers are non-parameter
state and don't participate in `FieldBank.attach_optimizer`. (Phase 2's `AttentionFuser` does
have parameters and will need to plug into the trainer's optimizer.)

#### Learnable mode for weighted averaging

`TrilinearFuser`, `InverseDistanceFuser`, and `RbfFuser` admit a *learnable* mode that uses
`gather` only — `scatter` is never called. (`OrnsteinUhlenbeckFuser` technically supports the
same training loop, but its kernel is non-smooth at voxel vertices — see its bullet above —
so the reconstructed field acquires creases at every vertex and the per-value gradients are
unstable for query points near vertices. Use it for direct-update mode only.) The target
parameter the caller passes (`field.values`, `field.bank.features`, ...) is left at its
initialization and is optimized end-to-end against the input features:

1. For each batch of `(points, feats)` from the dataset, compute `pred = fuser.gather(target, points, level=...)`.
2. Compute a reconstruction loss `loss = criterion(pred, feats)` (cosine distance for CLIP-like
   features, MSE for generic ones).
3. `loss.backward()` populates `target.grad`, and an optimizer step updates the per-vertex
   parameters. The 8-corner weighted gather distributes each point's gradient across its
   surrounding vertices smoothly, which produces a coherent field — unlike a nearest-vertex
   gather, where each point would only touch one vertex and neighboring vertices would not
   co-adapt.

Target choice changes the training contract:

- Target `field.values`: loss is computed in the field's `output_dim` space. Suitable when raw
  input features are also `output_dim`-shaped (e.g. directly fitting CLIP features into a 512-dim
  explicit field).
- Target `field.bank.features`: loss is computed in `bank.feature_dim` space, comparing gathered
  bank features against input features whose channels match the bank. The field's `implicit_net`
  is *not* in the loss path here — it only kicks in when the field is queried for its final
  `output_dim` prediction. For bank-target training to be meaningful, `bank.feature_dim` must be
  set to the input feature dim.

The Fuser API does not change: `gather` is differentiable in any `nn.Parameter` passed as the
target, so no new method is needed. What changes is the *caller*:

- Direct-update mode: a preprocessing loop calls `fuser.scatter(target, points, feats, level=...)`
  per frame; no optimizer, no gradients.
- Learnable mode: a training loop calls `fuser.gather(target, points, level=...)` per batch,
  computes loss, and steps an optimizer attached to `target` (plus any other field parameters
  via `FieldBank`). Scatter is never invoked.

`NearestVertexFuser` is excluded from learnable mode by design: its discrete vertex assignment
breaks gradient sharing across neighbors, so reconstruction training does not yield a useful
smooth field. The direct-update modes (`overwrite`, `running_average`, `ema`) are the only
supported usage for that fuser.

#### Pluggable prior gather on FieldStorage

The `FieldStorage._explicit` -> `TrilinearFuser.gather` consolidation generalizes: once the prior
gather is a Fuser call, FieldStorage can hold *any* Fuser as its prior-gather mechanism. The
prior path collapses to:

```python
def _prior(self, points):
    return self.prior_fuser.gather(self.values, points,
                                   grads=self.grads if self.cfg.gradient_augmentation else None,
                                   level=1)
```

This unifies several previously distinct concerns:

- The current "trilinear vs gradient-augmented trilinear" branch in `FieldStorage._explicit`
  becomes a Fuser choice — plain `TrilinearFuser` vs `TrilinearFuser` with a `grads` parameter.
- A field configured with `NearestVertexFuser` as its prior reads from the nearest leaf vertex
  (no interpolation) — useful when the prior is meant to be piecewise-constant, or when scatter
  is also done via `NearestVertexFuser` and round-trip exactness matters.
- Smoother kernels (`RbfFuser`, `InverseDistanceFuser`, `OrnsteinUhlenbeckFuser`) become
  available as prior gathers without any change to `FieldStorage` itself.

Compatibility constraints:

- `gradient_augmentation=True` requires the prior fuser to be a `TrilinearFuser` variant that
  consumes the `grads` tensor (it's the only kernel with an analytic Hermite extension).
- In hybrid mode (prior + implicit residual), the prior gather should be differentiable in
  `self.values` if the trainer optimizes the prior end-to-end; `NearestVertexFuser` breaks
  gradient sharing across neighbors (same caveat as the learnable-mode discussion above) and
  should be paired only with non-learning scatter initialization of `self.values`.

Implementation: extend `FieldStorageConfig` with a `prior_fuser_cfg` field defaulting to a plain
`TrilinearFuserConfig`, so existing SDF / OCC / VL fields keep their current behavior with no
config change. `FieldStorage.__init__` constructs the fuser (bound to its octree), registers it
as a submodule so trainable variants (Phase 2 attention) have their parameters discovered, and
`_prior` delegates to it.

### Phase 2: attention-based fusion

`AttentionFuser` is deferred to Phase 2 because it requires:

- Trainable transformer layers (`nn.Parameter`s registered as submodules of the fuser).
- Integration with a trainer's optimizer - the fuser must be discoverable by the trainer the way
  `FieldBank` discovers field parameters today, or added explicitly to a parameter group.
- A training objective. For VL features the natural choice is a reconstruction loss between
  gathered features and held-out per-pixel CLIP features on novel views; this needs a dataset
  split that the current `VLFeaturesDataset` doesn't expose.

The interface stays identical (`scatter` / `gather`), so once the training-loop questions are
answered the new subclass drops in alongside the Phase 1 ones.

## Integration points

- `VisionLanguageNetwork` (new, sibling of `SdfNetwork` / `OccNetwork` in `oren_vl/`): thin
  adapter that holds an octree, a `FieldBank` with a single `"vl"` field configured at
  `output_dim = C` (e.g. 512 for CLIP), and a Fuser bound to that octree (selected by
  `VisionLanguageNetworkConfig.fuser_cfg`). Two entry points beyond what SdfNetwork/OccNetwork
  expose:
    - `forward(points)`: gather VL features at query points. For non-learning configurations this
      simply runs the field bank (the field's `prior_fuser` does the gather). For the learnable
      mode this is the same call but inside a training loop that backprops through it.
    - `update(points, feats)` (or `scatter(points, feats)`): ingest per-frame features via
      `self.fuser.scatter(self.vl_field.values, points, feats)`. Used by the ROS mapping node
      and the offline demo to incorporate new observations into the field.
  The fuser instance held by `VisionLanguageNetwork` and the `prior_fuser` held by the field can
  be the same instance (passed through the config) so scatter and gather use exactly the same
  mechanism — round-tripping is exact for nearest-vertex and well-defined for the 8-corner
  variants.
- `oren_vl/demo_semi_sparse_octree_vl.py` (existing demo): not modified by this work. Used as a
  reference for how to build `NearestVertexFuser.scatter` (the bit-twiddling that locates the
  nearest leaf vertex for each point and the running-average accumulator pattern) and for the
  insert/scatter-phase split that `VisionLanguageNetwork.update` will mirror internally.
- `FieldStorage` and `FeatureBank` are untouched in their state layout. Fusers receive the target
  `nn.Parameter` (`field.values`, `field.bank.features`, ...) directly from the caller, hold a
  back-ref to the octree for geometry queries, and (for trilinear / inverse-distance / RBF / OU
  gather) reuse the existing `trilinear_interpolation` and `ga_trilinear` helpers.
- `FieldStorage._explicit` becomes a thin delegate to `TrilinearFuser.gather`: the field
  constructs (or borrows) one `TrilinearFuser` bound to its octree and forwards
  `self.values` (and `self.grads` when `cfg.gradient_augmentation`) at call time. This removes
  the duplicate trilinear/GA interpolation logic currently inlined in `_explicit` and keeps a
  single implementation of the math.
- `FieldBank` (the container) is untouched in Phase 1. A future Phase 2 may add a `FieldBank.fusers`
  mapping if the attention variant needs centralized lifecycle management.
- `TrilinearFuser.gather` covers the GA path via the optional `grads` argument (see Phase 1
  bullet), so `FieldStorage._explicit` consolidation is in scope. `TrilinearFuser.scatter` does
  *not* touch `grads` — a single scattered feature has no canonical Hermite gradient update, and
  GA stays purely a gather-side concern. For GA fields, `grads` is updated only by upstream
  optimizers (the existing trainers), never by a fuser.
- Running-average counts do *not* reset on octree growth. Existing per-vertex counts are
  preserved verbatim; only the newly appended slots are initialized to 0, so they pass the same
  touched/untouched gate as initial vertices. This matches the demo's existing scatter logic.
