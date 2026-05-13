# `FieldStorage` design

A unified per-vertex storage abstraction for any spatial field over a
`SemiSparseOctree`: SDF, occupancy, VL features, color, semantics, ...

## Goals

1. **Decouple field state from the octree.** The octree owns geometry only
   (`voxels`, `voxel_centers`, `vertex_indices`, `structure`); every field
   moves into its own `FieldStorage` instance.
2. **One abstraction for all fields.** `SdfNetwork` and `OccNetwork` collapse
   to thin wrappers around a `FieldStorage` of `output_dim=1`. VL features
   become `output_dim=C` (e.g. 512 for CLIP). Future fields plug in without
   touching the octree.
3. **Express the three regimes already in use.**
   - **explicit**: per-vertex value, trilinear interpolated (current SDF
     priors, occupancy priors, scattered VL features).
   - **implicit**: per-vertex feature, aggregated across levels, decoded by an
     MLP into a `D`-dim output (current "implicit only" path in `OccNetwork`).
   - **hybrid**: both — the implicit branch's output is added on top of the
     explicit prior (current default `SdfNetwork` / `OccNetwork`).

## Octree responsibility (after refactor)

`SemiSparseOctreeBase` keeps geometry buffers and exposes geometry queries:

```python
class OctreeGeometry:
    voxel_indices: Tensor   # (...,)              leaf index per point
    voxel_centers: Tensor   # (..., 3) metric
    voxel_sizes:   Tensor   # (..., 1) grid units
    vertex_indices: Tensor  # (..., 8) per-leaf 8 vertices
    voxel_offsets:  Tensor  # (..., 3) in [0,1]^3 — unit-cube point

# new octree.forward (no field math):
def query(points, voxel_indices=None, level=1) -> OctreeGeometry: ...
```

Field math (interpolation + MLP) moves into `FieldStorage`. Multiple fields
re-use the same `OctreeGeometry`, so per-point lookups happen once per
forward pass.

The `enable_sdf` / `enable_occupancy` / `enable_implicit` /
`implicit_feature_dim` / `implicit_num_levels` flags leave `OctreeConfig`.

## Config

```python
@dataclass
class FieldStorageConfig(ConfigABC):
    name: str                                # "sdf", "occ", "vl", ... — output key prefix
    output_dim: int = 1                      # D
    mode: Literal["explicit", "implicit", "hybrid"] = "hybrid"

    # ---- Explicit branch (mode in {explicit, hybrid}) ----
    explicit_prior_init: float = 0.0         # scalar init for the (V, D) buffer
    gradient_augmentation: bool = False      # Hermite-style GA; only when D == 1

    # ---- Implicit branch (mode in {implicit, hybrid}) ----
    implicit_feature_dim: int = 4            # F per level
    implicit_feature_level: int = 1          # # of levels (leaf upward) to sample
    implicit_feature_aggregation: Literal["cat", "sum", "max", "mean"] = "cat"
    implicit_net_cfg: ImplicitNetConfig = field(default_factory=ImplicitNetConfig)

    # ---- Sharing ----
    # Default: each field owns its (V, F) bank.
    # Opt-in: point at a named FeatureBank in the parent FieldBank to share the
    # *entire* feature vector with other fields pointed at the same bank. All
    # fields sharing a bank must agree on `implicit_feature_dim`.
    shared_bank: Optional[str] = None        # name of a FeatureBank in the parent FieldBank

    # Output composition is determined by `mode`:
    #   explicit -> pred = prior
    #   implicit -> pred = implicit
    #   hybrid   -> pred = prior.detach() + implicit
```

Per-level feature dim is **uniform** (matches current behavior). If
heterogeneous per-level dims are needed later, promote
`implicit_feature_dim` to `list[int]`.

### Sharing semantics

Sharing is whole-vector: a `FeatureBank` named in `shared_bank` is consumed
by every pointing field through the bank's full `(V, F)` tensor — no per-field
slicing. If a field needs its own implicit features, leave `shared_bank=None`
and it gets a private bank automatically.

A bank may also carry an **optional shared trunk** (`shared_net`, an
`ImplicitNet`) that decodes the trilinear-interpolated per-point feature into
a shared intermediate of dim `H`. When present, each field's own
`implicit_net` (the **decoding head**) consumes this intermediate instead of
the raw `F`-dim feature, and produces its `D`-dim output (or correction in
hybrid mode). When absent, the field's head consumes the raw feature directly
— this matches the per-field-bank default behavior.

```python
# Joint SDF + OCC sharing one 12-dim implicit bank, plus a shared 32-d trunk;
# each field's implicit_net is a small decoding head on top of the trunk.
FieldStorageConfig(name="sdf", mode="hybrid", implicit_feature_dim=12, shared_bank="geo")
FieldStorageConfig(name="occ", mode="hybrid", implicit_feature_dim=12, shared_bank="geo")
# FieldBank constructed with shared_banks=[("geo", 12, ImplicitNetConfig(out_dim=32))]
```

`FieldBank` checks at construction that all fields naming a bank agree on
`implicit_feature_dim` and that the dim matches the bank's declared size.
Heterogeneous-channel sharing (slicing) is intentionally out of scope; if a
future field needs only part of a shared representation it should either use
its own bank or get a dedicated bank with its own dim.

## Renames

| Old (today)         | New (after refactor) | Why                                      |
| ------------------- | -------------------- | ---------------------------------------- |
| `ResidualNet`       | `ImplicitNet`        | Describes the branch type, not just its role in hybrid mode |
| `ResidualNetConfig` | `ImplicitNetConfig`  | Matches the class rename                 |
| `FieldOutput.residual` (proposed) | `FieldOutput.implicit` | Same — "implicit" applies in both implicit and hybrid modes |

The rename is mechanical (no behavior change). The single existing file is
`oren/oren/residual_net.py`; it moves to `oren/oren/implicit_net.py` with
matching symbol updates in `sdf_network.py`, `occ_network.py`, and any YAML
references.

## Storage layout

| mode     | parameters                              | shape           |
| -------- | --------------------------------------- | --------------- |
| explicit | `values`                                | (V, D)          |
| explicit | `grads` (if `gradient_augmentation`)    | (V, 3)          |
| implicit | `features` + `implicit_net`             | (V, F) + MLP    |
| hybrid   | `values` + `features` + `implicit_net` + optional `grads` | as above |

`V` is `init_voxel_num` initially, grown by the octree on insertion. Fields
hook into the octree's resize callback to grow in lock-step.

When a `FeatureBank` carries a shared trunk, `features` lives on the bank
(once) and `shared_net` is owned by the bank; each field still owns its own
`implicit_net` as a per-field decoding head. The field's `implicit_net` input
dim is the trunk's output dim `H` (plus `D` for hybrid), not `F`.

Gradient augmentation is restricted to scalar fields (`D == 1`). For vector
fields it would require per-channel vertex gradients which blows up the
parameter count; the per-channel Eikonal regularizer also doesn't generalize.

## Mode diagrams

Flow of a single `FieldStorage` for each mode (no sharing — bank trunks are
covered in "Sharing semantics" above).

### explicit

```
                points  (N, 3)
                   │
                   ▼
        ┌────────────────────────┐
        │  octree.query(points)  │
        └───────────┬────────────┘
                    │  OctreeGeometry
                    │  (voxel_indices, voxel_centers,
                    │   voxel_sizes, vertex_indices,
                    │   voxel_offsets)
                    ▼
        ┌────────────────────────┐         params owned by FieldStorage
        │  gather vertex values  │◄─────── values (V, D)
        │  [+ vertex grads]      │◄─────── grads  (V, 3)   if GA & D==1
        └───────────┬────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │  (GA-)trilinear interp │
        └───────────┬────────────┘
                    ▼
              prior  (N, D)
                    │
                    ▼
              pred = prior            implicit = None
```

### implicit

```
                points  (N, 3)
                   │
                   ▼
        ┌────────────────────────┐
        │  octree.query(points)  │── (× L levels if multi-level)
        └───────────┬────────────┘
                    ▼
        ┌────────────────────────┐         params owned by FieldStorage
        │ gather vertex features │◄─────── features (V, F)
        └───────────┬────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │  trilinear interp +    │
        │  multi-level aggregate │   cat → (N, L·F)
        │  {cat,sum,mean,max}    │   else → (N,  F )
        └───────────┬────────────┘
                    ▼
        ┌────────────────────────┐
        │     ImplicitNet        │   (decoding head)
        │     in→D               │
        └───────────┬────────────┘
                    ▼
              implicit (N, D)
                    │
                    ▼
              pred = implicit         prior = None
```

### hybrid

```
                points  (N, 3)
                   │
                   ▼
        ┌────────────────────────┐
        │  octree.query(points)  │── (× L levels for implicit branch)
        └─────────┬──┬───────────┘
                  │  │
       ┌──────────┘  └──────────────┐
       │                            │
       ▼                            ▼
┌──────────────┐            ┌──────────────────┐
│ values (V,D) │            │ features (V, F)  │
│ grads  (V,3) │            └────────┬─────────┘
└──────┬───────┘                     │
       │                             ▼
       ▼                    ┌──────────────────┐
 (GA-)trilinear             │ trilinear + agg  │
       │                    └────────┬─────────┘
       ▼                             ▼
 prior (N, D) ──── .detach() ────► feats
       │                             │
       │                             ▼
       │                    ┌──────────────────┐
       │                    │   ImplicitNet    │  (decoding head;
       │                    │  (prior.detach,  │   takes both inputs
       │                    │   feats) → D     │   in hybrid)
       │                    └────────┬─────────┘
       │                             ▼
       │                       implicit (N, D)
       └──────────────┬──────────────┘
                      ▼
        pred = prior.detach() + implicit
```

Two structural points the diagrams make explicit:

- **`prior.detach()` is the gradient barrier in hybrid mode.** The explicit
  prior buffer is updated only by the explicit-branch loss term; the implicit
  branch learns a correction without back-propagating into the prior. This
  matches current `SdfNetwork` / `OccNetwork` behavior.
- **`octree.query` is called once per level used.** Implicit and hybrid modes
  with `implicit_feature_level > 1` need higher-level queries; explicit-only
  mode never asks for `level > 1`.

## Forward

```python
@dataclass
class FieldOutput:
    voxel_indices: Tensor    # (...,)
    prior:    Tensor | None  # (..., D) — None when mode == implicit
    implicit: Tensor | None  # (..., D) — None when mode == explicit
    pred:     Tensor         # (..., D)

class FieldStorage(nn.Module):
    def __init__(
        self,
        cfg: FieldStorageConfig,
        octree: SemiSparseOctreeBase,
        bank: Optional["FeatureBank"] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.octree = octree  # geometry provider; not registered as submodule
        # `bank` is the shared implicit-feature bank when cfg.shared_bank is set;
        # otherwise None and the field owns a private (V, F) `features` parameter.
        # When `bank` is present the field reads `bank.features` directly (whole
        # vector, no slicing); `bank.features.shape[1]` must equal
        # `cfg.implicit_feature_dim`.
        self.bank = bank
        # parameters per mode (see table above)
        # implicit_net built only when mode in {implicit, hybrid}, sized for the
        # aggregation choice: input dim = (L*F if cat else F) [+ D if hybrid]

    def forward(self, points, voxel_indices=None, prior_only=False) -> FieldOutput:
        geom = self.octree.query(points, voxel_indices, level=1)
        prior = self._explicit(geom)           if self.cfg.mode != "implicit" else None
        feats = self._implicit_feature(points, geom) if self.cfg.mode != "explicit"  else None

        # If the bank carries a shared trunk, the field's head consumes the
        # trunk's output rather than the raw per-point feature.
        if feats is not None and self.bank is not None and self.bank.shared_net is not None:
            feats = self.bank.shared_net(feats)

        if self.cfg.mode == "explicit" or prior_only or feats is None:
            pred, implicit = prior, None
        elif self.cfg.mode == "implicit":
            implicit = self.implicit_net(feats)        # decoding head
            pred = implicit
            prior = None
        else:  # hybrid
            implicit = self.implicit_net(prior.detach(), feats)   # decoding head
            pred = prior.detach() + implicit

        return FieldOutput(geom.voxel_indices, prior, implicit, pred)
```

`_implicit_feature` walks `cfg.implicit_feature_level` levels of the octree
(reusing the leaf-level `OctreeGeometry` for level 1, calling
`octree.query(points, level=k)` for k>1) and aggregates per the chosen op:

- `cat` → concat along last dim, implicit-net input dim grows to `L*F`.
- `sum` / `mean` → reduce across levels, implicit-net input dim stays `F`.
- `max` → element-wise max across levels (pooling), implicit-net input dim stays `F`.

## Multi-field on one octree

```python
class FeatureBank(nn.Module):
    """A (V, F) implicit-feature parameter plus an optional shared trunk.

    One or more FieldStorage instances may point at the same bank; they all
    consume the *whole* feature vector (no slicing). When `shared_net` is
    provided, fields receive the trunk's output instead of the raw feature
    and their own implicit_net acts as a decoding head."""
    def __init__(
        self,
        name: str,
        feature_dim: int,
        init_voxel_num: int,
        shared_net_cfg: Optional[ImplicitNetConfig] = None,
    ):
        super().__init__()
        self.name = name
        self.features = nn.Parameter(torch.zeros(init_voxel_num, feature_dim))
        self.shared_net = ImplicitNet(shared_net_cfg) if shared_net_cfg else None


class FieldBank(nn.Module):
    """A SemiSparseOctree + N FieldStorage instances + optional shared
    FeatureBanks. The trainer consumes this object."""
    def __init__(
        self,
        octree: SemiSparseOctreeBase,
        fields:        list[FieldStorageConfig],
        # (name, feature_dim, optional shared trunk cfg)
        shared_banks:  list[tuple[str, int, Optional[ImplicitNetConfig]]] = (),
    ):
        super().__init__()
        self.octree = octree
        self.banks = nn.ModuleDict({
            n: FeatureBank(n, d, octree.cfg.init_voxel_num, trunk_cfg)
            for n, d, trunk_cfg in shared_banks
        })
        self.fields = nn.ModuleDict()
        for f in fields:
            bank = self.banks[f.shared_bank] if f.shared_bank else None
            self.fields[f.name] = FieldStorage(f, octree, bank=bank)

    def forward(self, points, voxel_indices=None) -> dict[str, FieldOutput]:
        geom = self.octree.query(points, voxel_indices, level=1)
        # Higher levels lazily fetched by individual fields if they need them.
        return {name: field(geom) for name, field in self.fields.items()}
```

The bank pays the leaf-level `octree.query` once and dispatches it to every
field. This lets one octree carry, e.g., `sdf + vl` for joint mapping with no
redundant lookups.

## Migration

| Today                          | After                                                                 |
| ------------------------------ | --------------------------------------------------------------------- |
| `SdfNetwork(cfg)`              | wraps `FieldStorage(name="sdf", D=1, mode=hybrid, ga=True, agg="cat")` |
| `OccNetwork(cfg)` (hybrid)     | wraps `FieldStorage(name="occ", D=1, mode=hybrid, ga=False)`           |
| `OccNetwork` (`enable_occ=F`)  | `FieldStorage(name="occ", mode="implicit")`                            |
| `OccNetwork` (`enable_impl=F`) | `FieldStorage(name="occ", mode="explicit")`                           |
| VL scatter demo                | `FieldStorage(name="vl", D=C, mode="explicit", ga=False)` (no MLP)    |

Each wrapper preserves the current `forward()` return tuple
`(voxel_indices, prior, implicit, pred)` so `SdfTrainer` and `OccTrainer`
don't break during the transition. After the wrappers land, trainers can be
migrated to consume `FieldOutput` directly.

(Existing `SdfTrainer` / `OccTrainer` unpack the tuple as `(voxel_indices,
prior, residual, pred)`; the position is preserved so only the local
variable name changes — `sdf_residual` → `sdf_implicit`, etc. — when the
trainers are touched.)

## Trainer model

Composable, both layers in use:

- **Per-field trainer.** `SdfTrainer`, `OccTrainer`, and a future `VlTrainer`
  keep their current shape — they own a `FieldBank` with a single field, an
  optimizer over that field's params, and their own criterion. The trainer
  reads the named `FieldOutput` from the bank and dispatches to its
  criterion.
- **`MultiFieldTrainer`.** Composes existing per-field trainers. Each step:
  1. Run `FieldBank.forward(points)` once (shared geometry lookup).
  2. For each child trainer, call `child.criterion(...)` on the corresponding
     `FieldOutput`. Each child returns `(loss, loss_dict)`.
  3. Sum losses (configurable per-field weight), `loss.backward()`,
     `optimizer.step()` on a single optimizer that covers the whole bank
     (octree geometry buffers excluded; only field params).
  4. Merge `loss_dict`s with field-name prefix into the log.

  This is what enables joint SDF+VL learning where the VL field's gradient
  flows back into the shared implicit bank and constrains surface placement.

Single-field workflows use the per-field trainer directly; joint workflows
construct a `MultiFieldTrainer` from the same child instances.

## OctreeConfig cleanup

Remove from `OctreeConfig`: `enable_sdf`, `enable_occupancy`,
`enable_implicit`, `init_occ_prior`, `implicit_feature_dim`,
`implicit_num_levels`. What remains is pure geometry config (`resolution`,
`tree_depth`, `semi_sparse_depth`, `init_voxel_num`, `insertion_threshold`,
`skip_insertion_if_exists`, `independent_smallest_leaf_vertex`,
`gradient_augmentation` — also moves to `FieldStorageConfig`).

YAML migration (one-shot script):

```
old                                        new
──────────────────────────────────────── │ ────────────────────────────────────────
octree_cfg.enable_sdf: true              │ fields: [{name: sdf, mode: hybrid, ...}]
octree_cfg.enable_occupancy: true        │ fields: [{name: occ, mode: hybrid, ...}]
octree_cfg.implicit_feature_dim: 4       │ field.implicit_feature_dim: 4
octree_cfg.implicit_num_levels: 3        │ field.implicit_feature_level: 3
octree_cfg.gradient_augmentation: true   │ field.gradient_augmentation: true
octree_cfg.init_occ_prior: 0.0           │ field.explicit_prior_init: 0.0
```

The migration script lives at `scripts/migrate_field_configs.py`, walks
`configs/` and `oren_ros/configs/`, and rewrites in place with a `.bak`.

## Implementation order

1. `FieldOutput` dataclass + new `OctreeGeometry` return type.
2. Move geometry-only buffers; deprecate field flags on `OctreeConfig` with
   warnings (one release of overlap before deletion).
3. `FeatureBank` + `FieldStorage` skeleton (explicit mode first — covers VL
   demo). Verify scattered-feature parity with `demo_semi_sparse_octree_vl`.
4. Implicit + hybrid modes; apply the renames in the "Renames" section above
   and extend `ImplicitNet`'s signature for the aggregation choices. Verify
   SDF parity with `SdfTrainer`.
5. `FieldBank` + per-field trainer ports (`SdfTrainer`, `OccTrainer` thin
   over the bank).
6. `MultiFieldTrainer`; first joint workflow = SDF + scattered VL on the
   same octree.
7. YAML migration script; drop the deprecated `OctreeConfig` flags.
