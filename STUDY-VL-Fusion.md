Vision-language Field Variants: Study Plan
==========================================

This study compares fusion / training variants for the vision-language (VL) field built on top of
`FieldStorage` + `Fuser`. The baseline is `configs_local/replica-room0-vl.yaml`, which inherits
`configs/trainer-replica-vl.yaml` and currently runs:

- `cfg.mode = scatter`
- `field.mode = explicit`, `output_dim = 1024` (CLIP EVA02-L-14)
- `field.prior_fuser_cfg = TrilinearFuserConfig`
- Dataset: `Replica-SDF-aug3/room0/vl_features/eva02-l-14`

All variants below override only deltas against that base, keeping each leaf YAML small and the
study legible.

**Profiling is required for every study config.** The per-timer wall-clock breakdown that
`TrainerBase.train()` dumps to `<log_dir>/misc/profiling_stats.yaml` is only populated when the
trainer's GPU timers are enabled, which is gated by `profiling: true` (default in
`TrainerConfig` is `False`). Every Phase 1 and Phase 2 YAML must therefore set:

```yaml
profiling: true
profiling_verbose: false   # keep noise out of stdout; the timer dump goes to disk
final_save_profiling_stats: true   # default; included here for clarity
```

`final_save_profiling_stats` defaults to True in the trainer config, but the `profiling: true` flag
does not, so a config that omits it will end up with zero-filled per-timer rows in
`profiling_stats.yaml`. The headline `total_wall_time_s` and `train_loop_wall_time_s` numbers are
recorded unconditionally, but the per-component breakdown the study tables compare against will
be unusable. Treat `profiling: true` as part of the study contract, not an optional knob.

## Axes of variation

The variants are the Cartesian product of three independent axes; not every cell is interesting,
so the concrete list is curated below.

### Axis 1: training mode (`TrainerConfig.mode`)

- `scatter` -- direct-update fusion only. `train_with_frame` calls `model.scatter_update` per
  frame; no optimizer, no criterion, no inner loop. The fuser owns the per-vertex running state
  (`counts` or `weight_sum`).
- `optimize` -- reconstruction loss on the field via `VlCriterion`. `_train_one_iteration`
  samples `(points, features)` from key frames and steps Adam on the field parameters.

All `optimize-*` variants in this study run in **offline mode** (`online: false`), which is the
base config's default. The trainer streams the dataset once per `offline_epochs`, batches frames
into chunks of `offline_batch_frames`, and calls `train_with_frames` per chunk -- so each
optimizer step is informed by a fixed multi-frame window rather than the most-recent single
frame. Offline gives the most stable optimization signal for a static dataset like the Replica
VL feature dump; the `online` path is left to ROS-side runs. Knobs to tune per-variant:
`offline_epochs`, `offline_batch_frames`, `offline_shuffle`, `num_iterations_per_frame`,
`num_rays_total`, `batch_size`, `lr`.

### Axis 2: field storage regime (`FieldStorageConfig.mode`)

- `explicit` -- per-vertex `values (V, D)` only; `pred = prior_fuser.gather(values, points)`. D
  must equal the CLIP feature dim (1024 here).
- `implicit` -- per-vertex `features (V, F)` on a `FeatureBank` decoded by an MLP head; F is
  free, and the head maps `F -> D`. This is the only path that decouples the storage feature dim
  from the CLIP dim, so it's the natural lever for memory.
- `hybrid` -- `pred = prior.detach() + implicit_residual`. Both terms live in D-space; typically
  pairs a scatter-initialised prior with a learned residual.

### Axis 3: fuser choice (kernel for scatter / gather)

For `explicit` / `hybrid`, the relevant knob is `field.prior_fuser_cfg`. For `implicit`, it is
`field.implicit_fuser_cfg` (the per-level gather on `bank.features`).

- `NearestVertexFuser` -- scatter-only (`overwrite` / `running_average` / `ema`). Discrete vertex
  assignment breaks gradient sharing, so this fuser is excluded from learnable mode by design.
- `TrilinearFuser` -- partition-of-unity 8-corner splat. The recommended default for learnable
  gather; canonical implementation of the prior gather in `FieldStorage`.
- `InverseDistanceFuser` -- `w_i = 1 / (||p - c_i|| + epsilon)`. Sharper than trilinear near
  corners; epsilon-guarded against singularities.
- `RbfFuser` -- `w_i = exp(-||p - c_i||^2 / (2 * bw^2))`. Smooth (C-infinity) at vertices; good
  for learnable mode where the gather sits on or near a vertex.
- `OrnsteinUhlenbeckFuser` -- `w_i = exp(-||p - c_i|| / bw)`. $C^0$ but not $C^1$ at voxel vertices,
  so the learnable gather acquires creases at every vertex. Use for scatter only.

## Concrete variants

Naming convention: `{training-mode}-{field-mode}[-{fuser-or-F}]`. All variants inherit from
`../configs/trainer-replica-vl.yaml` and override only the listed fields. The dataset path
matches the existing `replica-room0-vl.yaml` so each variant is a drop-in swap.

### Scatter variants (no optimizer)

| Variant                     | Fuser                                | What it tests                                                |
|-----------------------------|--------------------------------------|--------------------------------------------------------------|
| `scatter-nearest-runavg`    | NearestVertex(`running_average`)     | Original demo baseline; hard vertex assignment, no smoothing |
| `scatter-nearest-overwrite` | NearestVertex(`overwrite`)           | Last-write-wins; sensitivity to frame order                  |
| `scatter-nearest-ema`       | NearestVertex(`ema`, alpha=0.1)      | Recency-biased nearest-vertex fusion                         |
| `scatter-trilinear`         | Trilinear                            | Current `replica-room0-vl.yaml`; smooth 8-corner splat       |
| `scatter-invdist`           | InverseDistance(epsilon=1e-6)        | Pole-at-corner kernel for comparison                         |
| `scatter-rbf`               | RBF(bandwidth=0.5 * resolution)      | Smoother neighborhood splat                                  |
| `scatter-ou`                | OrnsteinUhlenbeck(bw=0.5*resolution) | Heavier-tail splat; scatter-only by design                   |

These all keep `field.mode = explicit`, `output_dim = 1024`. Only the `prior_fuser_cfg` changes.

### Optimize-explicit variants (per-vertex CLIP)

Storage dim equals the CLIP dim (1024). The optimizer learns the 1024-d per-vertex values
end-to-end against the reconstruction loss; the gather kernel determines how each point's
gradient is distributed across its 8 corners.

| Variant                       | `prior_fuser_cfg`                 | Notes                                       |
|-------------------------------|-----------------------------------|---------------------------------------------|
| `optimize-explicit-trilinear` | Trilinear                         | Canonical learnable explicit; smooth gather |
| `optimize-explicit-rbf`       | RBF(bandwidth=0.5 * resolution)   | Smoother gradient distribution at vertices  |
| `optimize-explicit-invdist`   | InverseDistance(epsilon=1e-6)     | Compare against trilinear gradient flow     |

`NearestVertex` is *not* listed here -- it has no learnable mode (gradient sharing across
neighbors is broken by the discrete vertex assignment). `OrnsteinUhlenbeck` is excluded too
because of the non-smooth crease at every vertex; see `DESIGN-VL-Fusion.md` for the rationale.

### Optimize-implicit variants (small per-vertex latent + MLP)

Storage dim `bank.feature_dim = F` is decoupled from the CLIP dim. The MLP head decodes `F -> 1024`.
The sweep sizes F to characterise the memory / fidelity tradeoff.

| Variant                            | F   | Aggregation / levels                                | Notes                                       |
|------------------------------------|-----|-----------------------------------------------------|---------------------------------------------|
| `optimize-implicit-F16`            | 16  | single-level (`implicit_feature_level=1`)           | Aggressive compression; lower bound check   |
| `optimize-implicit-F32`            | 32  | single-level                                        | Default working point                       |
| `optimize-implicit-F64`            | 64  | single-level                                        | Mid-range capacity                          |
| `optimize-implicit-F128`           | 128 | single-level                                        | High capacity; ~8x smaller than explicit    |
| `optimize-implicit-F32-multilevel` | 32  | `implicit_feature_level=3`, aggregation=`cat`       | Multi-resolution latent; head sees 3F = 96  |

All use `implicit_fuser_cfg = TrilinearFuserConfig` (smooth gather is important for the head).
`implicit_net_cfg.hidden_dims` may need to grow for the larger F or for multi-level (e.g.
`[256, 256]`).

### Optimize-hybrid variants (jointly optimized prior + residual)

Hybrid runs in `cfg.mode = optimize` directly -- no scatter warmup needed. `FieldStorage` returns
both `pred = prior.detach() + residual` and `prior` itself; `VlCriterion` then computes two loss
terms (`vl_criterion.py:64-93`):

- `vl_loss_weight * loss(pred, gt_vl)` -- trains the residual end-to-end; gradient does not flow
  back into `values` here because of the `.detach()` on the prior inside `pred`.
- `vl_loss_prior_weight * loss(pred_prior, gt_vl)` -- trains the prior `values` directly through
  the prior fuser's differentiable gather. This is what lets the hybrid run in a single optimize
  pass: `values` learns from its own loss term, the residual learns from the combined-output term.

| Variant                  | Prior fuser    | F   | Notes                                                |
|--------------------------|----------------|-----|------------------------------------------------------|
| `optimize-hybrid-F32`    | Trilinear      | 32  | Smooth prior + small residual; baseline hybrid       |
| `optimize-hybrid-F64`    | Trilinear      | 64  | Larger residual capacity                             |
| `optimize-hybrid-F32-rbf`| RBF            | 32  | Smoother prior gradient flow + small residual        |

`NearestVertex` is intentionally excluded from the hybrid sweep too: its discrete vertex
assignment breaks gradient sharing on the prior-loss path the same way it does in
optimize-explicit, so the prior would learn poorly. If a piecewise-constant prior is wanted,
prefer freezing it (scatter-init then `vl_loss_prior_weight=0`) -- listed below as a follow-up if
the basic hybrid variants underperform.

## Metrics

Per variant report:

- **Reconstruction quality** on held-out frames: mean cosine similarity and L2 between gathered
  features and ground-truth CLIP features (cosine is the primary metric; L2 is for parity with
  L2-trained variants).
- **Memory**: `octree.capacity * (D + F)` floats at end of training; absolute MB.
- **Wall time per frame** in scatter mode; per iteration in optimize mode (use the existing
  `trainer.timer_*` instrumentation).
- **Open-vocabulary retrieval** (future work, out of scope for this study): top-k retrieval
  accuracy with a text prompt set on labelled Replica frames. Same protocol across variants.
  Deferred until the structural sweep settles on a winner.

## Phase 1: structural sweep

Run every variant in the concrete-variants tables above using the *defaults* inherited from
`configs/trainer-replica-vl.yaml` (no knob tuning yet). Goal: rank variants within each group on
the metrics above, then pick a single winner per group to carry into Phase 2.

Layout:

```
configs_local/replica/room0/vl/
    base.yaml                            # data_path + exp_name root; inherits trainer-replica-vl.yaml
    scatter-nearest-runavg.yaml
    scatter-nearest-overwrite.yaml
    scatter-nearest-ema.yaml
    scatter-trilinear.yaml               # supersedes the current replica-room0-vl.yaml
    scatter-invdist.yaml
    scatter-rbf.yaml
    scatter-ou.yaml
    optimize-explicit-trilinear.yaml
    optimize-explicit-rbf.yaml
    optimize-explicit-invdist.yaml
    optimize-implicit-F16.yaml
    optimize-implicit-F32.yaml
    optimize-implicit-F64.yaml
    optimize-implicit-F128.yaml
    optimize-implicit-F32-multilevel.yaml
    optimize-hybrid-F32.yaml
    optimize-hybrid-F64.yaml
    optimize-hybrid-F32-rbf.yaml
```

The current `configs_local/replica-room0-vl.yaml` is equivalent to
`configs_local/replica/room0/vl/scatter-trilinear.yaml` and can be retained as a thin alias or
deleted once the new tree lands.

Winner-selection rule per group (4 groups total: scatter, optimize-explicit, optimize-implicit,
optimize-hybrid): the variant with the best held-out cosine similarity, with memory as a
tiebreaker (smaller wins on near-ties, defined as within 1% relative cosine). Record runner-ups
in case the winner is dominated on memory or wall time for downstream use.

## Phase 2: knob tuning on winners

Once Phase 1 has named four winners (`W_scatter`, `W_opt_explicit`, `W_opt_implicit`,
`W_opt_hybrid`), sweep the cross-cutting knobs *only* on those winners. Each knob is orthogonal
and is best swept in isolation rather than as a full grid.

Cross-cutting knobs (relevance column says which winners the knob meaningfully changes):

| Knob                             | Sweep values                          | Relevance                                                                |
|----------------------------------|---------------------------------------|--------------------------------------------------------------------------|
| `criterion.vl_loss_type`         | `l2`, `l1`, `cosine`                  | All optimize-* winners                                                   |
| `criterion.vl_loss_prior_weight` | `0`, `0.1`, `1.0`                     | Hybrid winner only (in explicit, `prior == pred` -> double-count)        |
| `model.octree_cfg.resolution`    | `0.025`, `0.05`, `0.1`                | All winners; tie `bandwidth = 0.5 * resolution` for kernel fusers        |
| `lr`                             | `1e-3`, `5e-3`, `1e-2`                | Optimize winners                                                         |
| `num_iterations_per_frame`       | `1`, `4`, `16`                        | Optimize winners                                                         |
| `offline_epochs`                 | `1`, `3`, `5`                         | Optimize winners; cheapest way to push the head further on fixed data    |
| `offline_batch_frames`           | `5`, `10`, `20`                       | Optimize winners                                                         |

Hybrid-specific note: with `vl_loss_prior_weight=0`, the prior `values` stays at initialization;
combine with a scatter-init pass if a piecewise-constant prior is desired. (Phase 1 hybrid runs
optimize both prior and residual jointly.)

Layout (per winner, knob sweeps live in a sibling subdirectory so they're easy to skip when
re-running Phase 1):

```
configs_local/replica/room0/vl/
    phase2/
        scatter-winner/
            res-0.025.yaml
            res-0.1.yaml
        optimize-explicit-winner/
            loss-cosine.yaml
            lr-1e-2.yaml
            iters-16.yaml
            epochs-5.yaml
            res-0.025.yaml
        optimize-implicit-winner/
            loss-cosine.yaml
            lr-1e-2.yaml
            iters-16.yaml
            epochs-5.yaml
            res-0.025.yaml
        optimize-hybrid-winner/
            loss-cosine.yaml
            prior-weight-0.0.yaml          # freezes prior at init
            prior-weight-0.1.yaml
            lr-1e-2.yaml
            iters-16.yaml
            epochs-5.yaml
            res-0.025.yaml
```

Each leaf YAML inherits from the *winner's* Phase 1 file (not from `base.yaml`), so the only
delta is the swept knob itself.

Reporting deliverable at the end of Phase 2: a single best config per group (the winner with its
best knob settings folded in) plus an overall recommendation across groups, framed against the
memory / quality / wall-time tradeoff space.

## Tradeoffs to call out

- **Memory of `optimize-explicit`**: 1024 floats per vertex. At resolution=0.05 m over Replica
  room0 the octree easily reaches >1e6 vertices, i.e. >4 GB just for `values` in float32. The
  `optimize-implicit-F32` variant cuts this 32x at the cost of an MLP decode per query.
- **Round-trip exactness**: only the `scatter-*` + same-fuser-gather combinations round-trip a
  scattered feature back exactly. Nearest-vertex with `running_average` is exact only at the
  scattered point's nearest vertex; trilinear / RBF / etc. are exact only when the scatter has
  fully converged the per-vertex weighted mean.
- **OU in optimize mode**: the kernel's C^0 nonsmoothness at vertices produces unstable gradients
  on `values` for points-on-surface data. Listed in scatter-only sweep; intentionally excluded
  from optimize-explicit.

## Open tasks

- Decide on the held-out evaluation split for Replica room0 (frames not used during training) and
  surface it through `VLFeaturesDataset` so the same eval works across all variants.

## Dataset and parameter sizing

Numbers below come from the `vl_features_manifest.json` for Replica room0 and from the saved
`final.pth` checkpoints under `logs/replica/room0/vl/`.

### Dataset

- 3040 frames; per-frame feature map is a 16x16 grid (CLIP visual tokens, EVA02-L-14, feature dim
  D = 1024). Total feature points: 3040 * 16 * 16 = **778,240** (of which **778,230** survive the
  depth-validity filter; matches the eval `n` exactly).
- Raw feature payload at float32: 778,230 * 1024 * 4 B = **3.04 GB** on disk.
- Trajectory: 3040 poses, T_wc 4x4 row-major. Depth resampled to the same 16x16 grid for feature
  back-projection.

### Octree (resolution = 0.05 m, room0)

- Leaf voxels at end of training: **167,022**.
- Unique vertices referenced by the vertex-index table: **256,097 (32.91%)**. The `FieldStorage` value
  buffer is rounded up to **262,144** rows (= 256K) by the resize observer's growth schedule, so
  every per-vertex tensor below is reported at the buffer size (= what is stored on disk).
- Vertex utilisation differs by fuser, since each fuser writes to different corners per query:
  - `scatter-nearest-runavg`: 132,093 / 262,144 (50%) -- single nearest corner per point.
  - `scatter-trilinear`: 230,717 / 262,144 (88%) -- all 8 corners written per point.

### Per-variant parameter counts (from final.pth)

`Vertex` = sum of explicit `values` (D=1024) and implicit `features` (F) buffers at the octree's
pow-2 buffer capacity (262,144); `Used vertex` = subset of that buffer actually touched by the
fuser during training, read from the sparse `*_used_indices` keys saved by
`FieldStorage._save_to_state_dict` / `FeatureBank._save_to_state_dict` (percentage is of the dense
`Vertex` count above). `MLP` = decoder head weights+biases. All counts are float32 (counts buffer
is int64; not included). `scatter-*` is split into the 1-corner `nearest` row (132,093 touched
vertices) and the 4-fuser 8-corner row (230,717 touched vertices); `optimize-explicit-*` and the
implicit/hybrid groups all touch 230,717 vertices via the 8-corner gather.

| Variant                              | Vertex params | Used vertex params (% of dense) | MLP params | Total params | Float32 size |
|--------------------------------------|--------------:|--------------------------------:|-----------:|-------------:|-------------:|
| scatter-nearest-* (3 variants)       |   268,435,456 |          135,263,232 (50.4%) |          0 |  268,435,456 |     1024 MiB |
| scatter-{trilinear,rbf,invdist,ou}   |   268,435,456 |          236,254,208 (88.0%) |          0 |  268,435,456 |     1024 MiB |
| optimize-explicit-* (all 3)          |   268,435,456 |          236,254,208 (88.0%) |          0 |  268,435,456 |     1024 MiB |
| optimize-implicit-F16                |     4,194,304 |            3,691,472 (88.0%) |     75,968 |    4,270,272 |      16.3 MiB |
| optimize-implicit-F32                |     8,388,608 |            7,382,944 (88.0%) |     76,992 |    8,465,600 |      32.3 MiB |
| optimize-implicit-F64                |    16,777,216 |           14,765,888 (88.0%) |     79,040 |   16,856,256 |      64.3 MiB |
| optimize-implicit-F128               |    33,554,432 |           29,531,776 (88.0%) |    361,984 |   33,916,416 |     129.4 MiB |
| optimize-implicit-F32-multilevel     |     8,388,608 |            7,386,368 (88.1%) |    353,792 |    8,742,400 |      33.3 MiB |
| optimize-hybrid-F32                  |   276,824,064 |          243,637,152 (88.0%) |    142,528 |  276,966,592 |    1056.6 MiB |
| optimize-hybrid-F64                  |   285,212,672 |          251,020,096 (88.0%) |    144,576 |  285,357,248 |    1088.6 MiB |
| optimize-hybrid-F32-rbf              |   276,824,064 |          243,637,152 (88.0%) |    142,528 |  276,966,592 |    1056.6 MiB |

MLP layer breakdowns for the three head variants:

- `optimize-implicit-F32` head `[32 -> 64 -> 64 -> 64 -> 1024]`: 2,048 + 64 + 4,096 + 64 + 4,096 +
  64 + 65,536 + 1,024 = **76,992 params**.
- `optimize-implicit-F128` head `[128 -> 256 -> 256 -> 1024]` (the `hidden_dims: [256, 256]`
  override drops one layer vs the F<=64 default `[64, 64, 64]`): 32,768 + 256 + 65,536 + 256 +
  262,144 + 1,024 = **361,984 params**.
- `optimize-implicit-F32-multilevel` head `[96 -> 256 -> 256 -> 1024]` (input is `cat` of 3 levels
  of F=32): 24,576 + 256 + 65,536 + 256 + 262,144 + 1,024 = **353,792 params**.

The MLP is essentially free compared to the per-vertex storage: even F128's 362K-param head is
~93x smaller than its 33.55M-param feature bank. The bottleneck is the vertex tensor, not the
decoder.

### Memory-cost framing

For Replica room0 at 0.05 m resolution:

- Explicit / hybrid storage = ~1 GiB (the 1024-d per-vertex CLIP values dominate).
- Implicit-F32 storage = ~32 MiB -- **32x smaller**.
- Implicit-F128 storage = ~130 MiB -- 8x smaller than explicit; 4x bigger than F32 for **+0.4%**
  cosine on this Phase 1 run, so capacity is *not* what's holding implicit back at these defaults.

The runtime GPU peaks reported in the Phase 1 table come in two columns. `GPU delta` is the
per-frame baselined peak above what the `train with frame` block already had allocated at entry
(historically the only number we reported); it captures the *additional* working set the block
needs. `Total GPU` is `raw_stats.end_peak.max_bytes` of the same record -- the absolute peak from
`torch.cuda.max_memory_allocated()` across the block, including the resident model parameters and
optimizer state that were already on the device before the block started. The dataset's CLIP
feature cache lives on CPU (CPU `end_rss` peaks at ~3 GiB for every variant); only a per-frame
slice ever moves to GPU, so it does not dominate the GPU total. Concretely:

- **optimize-implicit-F32**: ~233 MiB total. Roughly 32 MiB feature bank + 64 MiB Adam state +
  ~137 MiB per-iteration activations / grads / batch tensors / octree buffers. The block-entry
  baseline already includes params + Adam, so the delta column (~167 MiB) reports just the
  per-frame working set on top.
- **optimize-explicit / optimize-hybrid**: ~5.2-5.5 GiB total. Dominated by 1 GiB `values` +
  2 GiB Adam (`exp_avg` + `exp_avg_sq`); the delta column (~4.1-4.4 GiB) is the per-frame backward
  pass plus the (capacity, D) scratch that the 8-corner scatter / gather kernels still allocate
  in dense form.
- **scatter-nearest-***: ~1.06 GiB total but only ~6 MiB delta. The 1 GiB `values` buffer is
  baseline; the sparse Welford / EMA / overwrite path only touches a tiny scratch per frame.
- **scatter-{trilinear,rbf,invdist,ou}**: ~4.15 GiB total with ~3.09 GiB delta -- the dense
  `(capacity, D)` scratch is reallocated on every frame and shows up almost entirely in the delta.

## Phase 1 results (2026-05-20)

All 18 variants completed against Replica room0 EVA02-L-14 dump (3040 frames, resolution=0.05 m,
offline_epochs=1, num_iterations_per_frame=4 for optimize-* variants). Evaluation is on the same
data stream as training (no held-out split for Phase 1, per the agreed scope). Configs live under
`configs_local/replica/room0/vl/`; runs land in `logs/replica/room0/vl/<variant>/<timestamp>/`;
aggregated CSV at `logs/replica/room0/vl/_run_logs/phase1_results.csv`.

`GPU peak delta` is the per-frame peak above the train-frame block's entry baseline
(`max_peak_bytes` of the GPU `MemoryRecord`, i.e. the historical "GPU peak (MiB)" column).
`Total GPU (MiB)` is the absolute peak observed inside any train-frame block --
`raw_stats.end_peak.max_bytes` of the GPU `MemoryRecord`, i.e. `torch.cuda.max_memory_allocated()`
across the block including the pre-existing model + optimizer + dataset-cache baseline. Std
across blocks is 0 within rounding for every variant (the per-block working set is deterministic
once the octree is full), so we omit it.

Highest cosine per group is **bolded** (best of all rows is the scatter group winner).

| Group              | Variant                          | Cosine  | L1     | L2 RMSE | Wall (s) | GPU delta (MiB) | Total GPU (MiB) |
|--------------------|----------------------------------|---------|--------|---------|----------|-----------------|-----------------|
| scatter            | **scatter-nearest-runavg**       | **0.9637** | 0.2306 | 0.3933  | 21.0     | 6               | 1065            |
| scatter            | scatter-nearest-ema              | 0.9615  | 0.5360 | 0.7873  | 17.3     | 6               | 1064            |
| scatter            | scatter-trilinear                | 0.9596  | 0.2576 | 0.4164  | 25.1     | 41              | 1100            |
| scatter            | scatter-rbf                      | 0.9578  | 0.2640 | 0.4253  | 23.7     | 41              | 1100            |
| scatter            | scatter-ou                       | 0.9551  | 0.2729 | 0.4385  | 20.4     | 41              | 1100            |
| scatter            | scatter-invdist                  | 0.9535  | 0.2775 | 0.4457  | 18.8     | 41              | 1100            |
| scatter            | scatter-nearest-overwrite        | 0.9190  | 0.3205 | 0.5998  | 19.0     | 2               | 1060            |
| optimize-explicit  | **optimize-explicit-invdist**    | **0.8632** | 0.5900 | 0.9647  | 84.4     | 4132            | 5191            |
| optimize-explicit  | optimize-explicit-rbf            | 0.8604  | 0.6052 | 0.9867  | 79.2     | 4132            | 5191            |
| optimize-explicit  | optimize-explicit-trilinear      | 0.8578  | 0.6165 | 1.0015  | 76.7     | 4132            | 5191            |
| optimize-implicit  | **optimize-implicit-F32-multilevel** | **0.8983** | 0.4180 | 0.6532  | 31.0     | 174             | 242             |
| optimize-implicit  | optimize-implicit-F128           | 0.8981  | 0.4098 | 0.6495  | 28.4     | 555             | 719             |
| optimize-implicit  | optimize-implicit-F32            | 0.8945  | 0.4204 | 0.6592  | 29.1     | 167             | 233             |
| optimize-implicit  | optimize-implicit-F64            | 0.8929  | 0.4209 | 0.6635  | 23.4     | 295             | 393             |
| optimize-implicit  | optimize-implicit-F16            | 0.8919  | 0.4247 | 0.6681  | 19.5     | 105             | 155             |
| optimize-hybrid    | **optimize-hybrid-F32**          | **0.9319** | 0.3432 | 0.5404  | 87.7     | 4284            | 5375            |
| optimize-hybrid    | optimize-hybrid-F32-rbf          | 0.9300  | 0.3448 | 0.5473  | 85.2     | 4284            | 5375            |
| optimize-hybrid    | optimize-hybrid-F64              | 0.9283  | 0.3466 | 0.5510  | 82.0     | 4412            | 5535            |

### Per-group winners (winner-selection rule: max cosine; Total GPU tiebreaker within 1% relative)

- **scatter**: `scatter-nearest-ema` by the strict rule -- cosine 0.9615 (within 0.23% of runavg's
  0.9637) and both share an ~1064 MiB Total GPU (1 GiB `values` dominates). *However*, on L1
  (0.236 vs 0.536) and L2 RMSE (0.393 vs 0.787) runavg is 2x better -- the cosine near-tie hides a
  real reconstruction-quality gap, because cosine is scale-invariant and EMA's recency bias
  produces smaller-magnitude vertex values without changing direction much. If the eventual
  downstream task is text retrieval (which is cosine-based), EMA's tiny advantage stands; if any
  downstream task uses absolute features (regression, geometry), runavg is the right pick. The two
  are essentially co-winners on memory after the runavg sparse-update optimization landed (see
  the second bullet in "Observations" below). Runner-up: `scatter-nearest-runavg`.
- **optimize-explicit**: `optimize-explicit-invdist` -- cosine 0.8632, Total GPU 5191 MiB. All three
  kernels are within 0.6% relative cosine of each other and equal on memory; InvDist edges out RBF
  and Trilinear by a hair. Runner-up: `optimize-explicit-rbf`.
- **optimize-implicit**: `optimize-implicit-F32-h64-256` -- cosine 0.9008, Total GPU 237 MiB
  (see "Decoder MLP sweep" below). Multilevel was the Phase 1 default-head winner (cosine 0.8983
  at 242 MiB) but the widened head pushes single-level F=32 ahead. All ten implicit variants land
  within 0.9% of each other on cosine, so the head -- not the latent dimension or aggregation --
  was the bottleneck. Runner-up under the default head: `optimize-implicit-F32-multilevel`.
- **optimize-hybrid**: `optimize-hybrid-F32` -- cosine 0.9319, Total GPU 5375 MiB. RBF prior + F=32
  is within noise (0.9300). Runner-up: `optimize-hybrid-F32-rbf`. (Group shelved -- see "Carrying
  into Phase 2" below.)

### Observations

- **Scatter dominates quality on a static dataset evaluated on its training data.** Best scatter
  hits 0.964 vs best optimize-hybrid 0.932, best optimize-implicit 0.898, best optimize-explicit
  0.863. This is partly structural: scatter assigns GT features directly to the nearest vertex /
  trilinear corners, so on training-data evaluation it has a round-trip advantage that optimize-*
  has to learn. Phase 2 / held-out eval should narrow this gap.
- **Sparse-update optimization on `NearestVertexFuser.running_average` and `_CornerWeightFuser`
  landed mid-study.** The original implementations allocated `(capacity, D)` scratch on every frame,
  so per-frame delta was driven by octree capacity (~3 GiB) and wall time by allocator traffic.
  Reducing per-unique-vidx first (Welford form unchanged) brought runavg from 80.7 s / 3079 MiB
  delta to 21.0 s / 6 MiB delta with bit-identical metrics; the same fix on `_CornerWeightFuser`
  brought the four 8-corner scatter variants (trilinear / rbf / invdist / ou) from ~80 s /
  3092 MiB delta to ~22 s / 41 MiB delta -- a 3.5-4x wall-time speedup and a 75x delta-memory cut,
  also bit-identical to the dense path at the reported precision.
- **Optimize-explicit is undertrained at these defaults.** All three explicit kernels plateau
  around 2.2 reconstruction loss with cosine 0.86 -- the per-vertex 1024-d values start at 0 and
  ~1200 Adam steps are not enough to fit ~778K query points x 1024 dims. Phase 2's `iters-16`,
  `epochs-5`, and `lr-1e-2` knobs are the right levers to push here.
- **Implicit's memory advantage is large and quality is competitive with hybrid.** 333 MiB at
  cosine 0.898 vs hybrid 4283 MiB at cosine 0.932 -- a 12.8x memory cut for ~3.5% cosine drop. The
  flat cosine across F=16..128 says the head, not the latent, is the current bottleneck.
- **Hybrid beats explicit by 7% cosine** at essentially the same memory budget -- the small (F=32)
  implicit residual is doing real work despite the prior `.detach()` cutting its gradient flow
  into the explicit `values`.

### Decoder MLP sweep (2026-05-21)

The Phase 1 implicit defaults use `hidden_dims=[64, 64, 64]` for F<=64 and `[256, 256]` for F=128
(`optimize-implicit-F32-multilevel` overrides to `[256, 256]` too). All five plateau within 0.7%
cosine, so the head -- not the latent -- is the suspect bottleneck. We tested a
progressive-widening structure that goes `F -> first-hidden -> 256 -> 1024` instead of the flat
constant-width stack:

- `hidden_dims=[64, 256]` for F in {16, 32, 64} -- "h64-256" suffix.
- `hidden_dims=[128, 256]` for F=128 -- "h128-256" suffix.
- Both heads applied to F=32-multilevel (head input 96 from `cat` of 3 levels of F=32).

Configs: `configs_local/replica/room0/vl/optimize-implicit-F{16,32,64,128,32-multilevel}-h*.yaml`.
Run with `run_phase1_implicit_head_sweep.sh`. Same per-vertex feature buffer; only the head
changes.

`Delta` is the train-frame baselined peak (historical "GPU (MiB)"); `Total` is the absolute peak
during any block (`raw_stats.end_peak.max_bytes`). Highest cosine in the table is **bolded**.

| Variant                                  | Head structure           | Cosine  | dCosine  | L1     | L2 RMSE | Wall (s) | Delta (MiB) | Total (MiB) | MLP params |
|------------------------------------------|--------------------------|---------|----------|--------|---------|----------|-------------|-------------|-----------:|
| optimize-implicit-F16                    | 16  -> 64x3 -> 1024      | 0.8919  |          | 0.4247 | 0.6681  | 27.6     | 105         | 155         |     75,968 |
| optimize-implicit-F16-h64-256            | 16  -> 64 -> 256 -> 1024 | 0.8974  | +0.0055  | 0.4111 | 0.6519  | 20.3     | 112         | 163         |    280,896 |
| optimize-implicit-F32                    | 32  -> 64x3 -> 1024      | 0.8945  |          | 0.4204 | 0.6592  | 20.0     | 167         | 233         |     76,992 |
| **optimize-implicit-F32-h64-256**        | 32  -> 64 -> 256 -> 1024 | **0.9008** | +0.0063  | 0.4110 | 0.6449  | 22.9     | 170         | 237         |    281,920 |
| optimize-implicit-F64                    | 64  -> 64x3 -> 1024      | 0.8929  |          | 0.4209 | 0.6635  | 36.4     | 295         | 393         |     79,040 |
| optimize-implicit-F64-h64-256            | 64  -> 64 -> 256 -> 1024 | 0.8990  | +0.0061  | 0.4118 | 0.6491  | 27.7     | 298         | 397         |    283,968 |
| optimize-implicit-F128                   | 128 -> 256x2 -> 1024     | 0.8981  |          | 0.4098 | 0.6495  | 28.0     | 555         | 719         |    361,984 |
| optimize-implicit-F128-h128-256          | 128 -> 128 -> 256 -> 1024| 0.8974  | -0.0007  | 0.4159 | 0.6538  | 28.5     | 554         | 718         |    312,704 |
| optimize-implicit-F32-multilevel         | 96  -> 256x2 -> 1024     | 0.8983  |          | 0.4180 | 0.6532  | 26.6     | 174         | 242         |    353,792 |
| optimize-implicit-F32-multilevel-h64-256 | 96  -> 64 -> 256 -> 1024 | 0.8977  | -0.0006  | 0.4180 | 0.6539  | 26.9     | 173         | 240         |    286,016 |
| optimize-implicit-F32-multilevel-h128-256| 96  -> 128 -> 256 -> 1024| 0.8973  | -0.0010  | 0.4189 | 0.6554  | 27.0     | 174         | 241         |    308,608 |

Observations:

- **Single-level F<=64 wins clearly**: the progressive head lifts cosine by 0.55-0.63% across F in
  {16, 32, 64}, drops L1 by ~3% and L2 RMSE by ~2%, with no measurable change in GPU peak. The
  +205K head parameters (~75K -> ~282K) are still tiny next to the per-vertex buffer (e.g. 7.4 M
  for F=32), so the memory budget is unchanged in practice. **`optimize-implicit-F32-h64-256` is
  the new implicit winner at cosine 0.9008**, edging out the previous winner
  (`optimize-implicit-F32-multilevel` at 0.8983).
- **F=128 is a wash**: -0.0007 cosine with the proposed head. The original `[256, 256]` already
  widens enough; replacing the first 256 with 128 narrows the head without gain.
- **Multilevel does not benefit**: both proposed heads land 0.06-0.10% below the original. The
  multilevel 96-d input already encodes per-level structure; collapsing it to 64 (or even 128)
  before going up to 256 throws away some of that.
- Capacity is *not* the bottleneck even with the wider head: F=16-h64-256 and F=64-h64-256 differ
  by 0.0016 cosine -- a 4x change in F costs almost nothing once the head can fit the mapping.

Carrying forward: the implicit-winner for Phase 2 is `optimize-implicit-F32-h64-256`, not the
multilevel variant. The new YAMLs stay alongside the originals so the comparison is reproducible.

### Carrying into Phase 2

- **`optimize-hybrid` is shelved** (decision 2026-05-21). The 4.3 GiB GPU peak for cosine 0.932 is
  not worth the memory cost when `optimize-implicit` hits 0.898 at 333 MiB; the prior-loss path
  also adds a config knob (`vl_loss_prior_weight`) the other groups don't need. The 3 hybrid
  YAMLs and their Phase 1 numbers stay in this doc as historical record, but Phase 2 sweeps
  and any downstream work target scatter / optimize-explicit / optimize-implicit only.
- The remaining three winners (`scatter-nearest-ema` / `scatter-nearest-runavg`,
  `optimize-explicit-invdist`, `optimize-implicit-F32-h64-256`) become the
  `phase2/*-winner/` knob-sweep targets. The implicit winner moved from
  `optimize-implicit-F32-multilevel` (cosine 0.8983) to `optimize-implicit-F32-h64-256`
  (cosine 0.9008) after the head-structure sweep below.
- The optimize-explicit and optimize-implicit groups both look undertrained; the `iters-16` and
  `epochs-5` knobs are the highest-leverage knobs to try first.
- Add a held-out split (every Nth frame) before Phase 2 to break the scatter round-trip advantage.

## Phase 1.5: loss-function sweep on optimize-implicit

`criterion.vl_loss_type` only applies to the `optimize-*` variants -- the scatter group has no
loss / optimizer / criterion in the pipeline and is unaffected by this knob. Among the optimize
variants, every Phase 1 row was trained with `vl_loss_type: l2` (the
`configs/trainer-replica-vl.yaml` default). That is a structural mismatch with the evaluation
metric, which is cosine similarity: L2 couples direction and magnitude, so a vertex feature whose
direction is correct but whose magnitude undershoots is penalised even though it is a perfect
prediction under cosine. Optimize-implicit is the group most likely to benefit from fixing this --
its memory advantage is the headline finding of Phase 1 (~237 MiB at cosine 0.9008 vs ~5.2 GiB for
optimize-explicit, ~1.06 GiB for scatter-nearest), and a 6.3% cosine gap to scatter-nearest-runavg
is what's keeping it from being the recommended default. The point of Phase 1.5 is to land
whatever cosine gain comes from a zero-engineering change to the loss before the heavier Phase 2
knobs (lr / iters / epochs) start running.

### Scope

Every existing optimize-implicit variant -- the five Phase 1 originals plus the six head-sweep
variants -- is paired with two new YAMLs that only override `criterion.vl_loss_type`:

| Source variant                              | F   | Head structure           | Loss sweep targets |
|---------------------------------------------|-----|--------------------------|--------------------|
| `optimize-implicit-F16`                     | 16  | `[64, 64, 64]` default   | `-l1`, `-cosine`   |
| `optimize-implicit-F32`                     | 32  | `[64, 64, 64]` default   | `-l1`, `-cosine`   |
| `optimize-implicit-F64`                     | 64  | `[64, 64, 64]` default   | `-l1`, `-cosine`   |
| `optimize-implicit-F128`                    | 128 | `[256, 256]` default     | `-l1`, `-cosine`   |
| `optimize-implicit-F32-multilevel`          | 96  | `[256, 256]` default     | `-l1`, `-cosine`   |
| `optimize-implicit-F16-h64-256`             | 16  | `[64, 256]` progressive  | `-l1`, `-cosine`   |
| `optimize-implicit-F32-h64-256`             | 32  | `[64, 256]` progressive  | `-l1`, `-cosine`   |
| `optimize-implicit-F64-h64-256`             | 64  | `[64, 256]` progressive  | `-l1`, `-cosine`   |
| `optimize-implicit-F128-h128-256`           | 128 | `[128, 256]` progressive | `-l1`, `-cosine`   |
| `optimize-implicit-F32-multilevel-h64-256`  | 96  | `[64, 256]` progressive  | `-l1`, `-cosine`   |
| `optimize-implicit-F32-multilevel-h128-256` | 96  | `[128, 256]` progressive | `-l1`, `-cosine`   |

That's 11 sources x 2 losses = 22 new YAMLs. The L2 numbers from Phase 1 / Phase 1's
"Decoder MLP sweep" subsection serve as the comparison baseline -- no need to re-run those.
Optimize-explicit and scatter are *not* part of this sweep: explicit is already known to be
undertrained at these defaults and Phase 2 will exercise it more aggressively; scatter has no
loss at all.

### Layout

```
configs_local/replica/room0/vl/phase1.5/
    optimize-implicit-F16-l1.yaml
    optimize-implicit-F16-cosine.yaml
    optimize-implicit-F32-l1.yaml
    optimize-implicit-F32-cosine.yaml
    optimize-implicit-F64-l1.yaml
    optimize-implicit-F64-cosine.yaml
    optimize-implicit-F128-l1.yaml
    optimize-implicit-F128-cosine.yaml
    optimize-implicit-F32-multilevel-l1.yaml
    optimize-implicit-F32-multilevel-cosine.yaml
    optimize-implicit-F16-h64-256-l1.yaml
    optimize-implicit-F16-h64-256-cosine.yaml
    optimize-implicit-F32-h64-256-l1.yaml
    optimize-implicit-F32-h64-256-cosine.yaml
    optimize-implicit-F64-h64-256-l1.yaml
    optimize-implicit-F64-h64-256-cosine.yaml
    optimize-implicit-F128-h128-256-l1.yaml
    optimize-implicit-F128-h128-256-cosine.yaml
    optimize-implicit-F32-multilevel-h64-256-l1.yaml
    optimize-implicit-F32-multilevel-h64-256-cosine.yaml
    optimize-implicit-F32-multilevel-h128-256-l1.yaml
    optimize-implicit-F32-multilevel-h128-256-cosine.yaml
    run_phase1.5.sh
```

Each YAML inherits from its same-named Phase 1 source (not from `base.yaml`) so the only delta is
the loss type. `exp_name` adds the loss-type suffix so logs land under
`logs/replica/room0/vl/optimize-implicit-<...>-<loss>/<timestamp>/` (mirrors the Phase 1
convention -- the aggregator pattern works as-is).

### Reporting

Append a table to the Phase 1 results section -- one row per `(F, head, loss)` triple, with the
existing Phase 1 `L2` rows reused as the baseline column rather than duplicated. Bold the highest
cosine within each `(F, head)` pair so the per-row L1 / L2 / Cosine comparison is immediate.

### Hypotheses to check

- **Cosine should win on cosine.** If the loss matches the metric, we expect `-cosine` to lead. A
  flat result would imply the head saturates well before either loss's gradient runs out, and the
  next lever is iters/epochs.
- **L1 may help with the L1 / L2 RMSE columns.** L1 is the natural fit for the L1 metric column;
  if a downstream task cares about absolute reconstruction (regression, not retrieval) L1 may
  beat L2 there even where it loses on cosine.
- **Head structure x loss interaction.** Wider heads might absorb more of the loss's gradient
  signal. We'll see this as either parallel improvements across all five `-h*-256` variants under
  `-cosine`, or as a divergence (e.g. only F=32 benefits).

### Carry-forward rule

After Phase 1.5 the implicit Phase 2 winner is the best `(F, head, loss)` triple, not just the
best `(F, head)` pair. Update "Carrying into Phase 2" to point at it when the table lands.

## Future work

- Open-vocabulary retrieval metric: pick a fixed text-prompt set on labelled Replica frames and
  report top-k retrieval accuracy across variants. Out of scope for this study; revisit once the
  structural sweep has produced a winning configuration.
