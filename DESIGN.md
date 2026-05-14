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

`SemiSparseOctreeBase` keeps geometry buffers and exposes geometry queries.
The result is a two-tier object: per-level snapshots (immutable, the data a
field actually consumes) wrapped in a per-call cache (memoizes higher-level
queries so multiple fields on the same octree share the work).

```python
@dataclass
class LevelGeometry:
    """Per-level snapshot for a single query batch."""
    level:         int
    voxel_indices: Tensor   # (...,)              leaf index per point at this level
    voxel_centers: Tensor   # (..., 3) metric
    voxel_sizes:   Tensor   # (..., 1) grid units
    vertex_indices: Tensor  # (..., 8) per-leaf 8 vertices
    voxel_offsets: Tensor   # (..., 3) in [0,1]^3 — unit-cube point


class OctreeGeometry:
    """Per-call multi-level geometry cache.

    Built once per forward pass by `octree.query(points)`, then passed to
    every FieldStorage on the bank. Level 1 is populated eagerly (every
    field needs it); levels > 1 are computed on first `at_level(k)` call
    and cached so subsequent fields (or aux-bank lookups) reuse the
    result. Cache lifetime = one forward pass; no global state.
    """
    octree:  "SemiSparseOctreeBase"  # back-ref, not registered as submodule
    points:  Tensor                  # (..., 3)
    _levels: dict[int, LevelGeometry]

    def at_level(self, level: int = 1) -> LevelGeometry:
        if level not in self._levels:
            self._levels[level] = self.octree._compute_level_geometry(self.points, level)
        return self._levels[level]


# octree entry point (no field math):
def query(points, voxel_indices=None) -> OctreeGeometry:
    """Returns a cache pre-populated with level 1 (voxel_indices honored
    if provided; otherwise resolved internally). Higher levels lazy."""
```

Field math (interpolation + MLP) moves into `FieldStorage`. Two layers of
sharing fall out:

- **Across fields, same level.** `FieldBank.forward(points)` constructs one
  `OctreeGeometry` per call and hands the same object to every field, so
  level-1 (and every other level any field has needed) is computed exactly
  once per forward.
- **Across levels within a field.** A field with `implicit_feature_level > 1`
  walks levels via `geom.at_level(k)`; aux-bank gathers reuse the same
  accessor. The first walk pays the C++ `find_voxel_indices` cost; later
  walks return cached `LevelGeometry`s. This matters most for joint workflows
  (SDF + OCC + VL all asking for `level=1..L` from the same `geo` bank).

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

    # ---- Auxiliary read-only inputs to the decoder ----
    # Each entry names another *shared* bank whose per-point features are
    # concatenated to this field's own features before the decoding head.
    # `detach=True` blocks the gradient flow from this field's losses into the
    # aux bank's parameters (one-way coupling: this field reads but does not
    # shape the aux bank).
    auxiliary_banks: list["AuxiliaryBankSpec"] = field(default_factory=list)

    # Output composition is determined by `mode`:
    #   explicit -> pred = prior
    #   implicit -> pred = implicit
    #   hybrid   -> pred = prior.detach() + implicit


@dataclass
class AuxiliaryBankSpec(ConfigABC):
    name: str          # must reference a shared FeatureBank on the FieldBank
    detach: bool = False
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

#### Auxiliary banks (read-only decoder inputs)

A field's `auxiliary_banks` list names other *shared* banks whose per-point
features get concatenated to this field's own features before the decoding
head. This is for the pattern "field X has its own implicit storage but its
decoder also wants to see field Y's implicit features" — e.g., a VL field
that wants geometry-aware grounding without contributing its own learnable
geometry features.

Each entry is an `AuxiliaryBankSpec(name, detach)`. `detach=True` blocks the
gradient flow from this field's losses into the aux bank's parameters,
making the coupling one-way: the field reads the aux features but does not
shape them. This is the symmetric counterpart to `prior.detach()` in hybrid
mode — same rationale (blame attribution; keep the aux bank's representation
shaped by its own owners only).

Constraint: aux entries can only name **shared** banks (banks declared on
the `FieldBank`'s `shared_banks`). Pointing at another field's private bank
is rejected at construction — it would create coupling that's invisible at
the call site and surprising to readers of the YAML.

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
`implicit_net` as a per-field decoding head, called as a single-input MLP
on whatever tensor FieldStorage assembles for it. The field's
`implicit_net.in_dim` is the trunk's output dim `H` (plus `Σ aux_dim` if any
aux banks, plus `D` in hybrid mode), not `F`.

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
 prior (N, D) ─── .detach() ───┐    feats
       │                       │     │
       │                       ▼     ▼
       │                    ┌────────────────────┐
       │                    │ cat(prior.detach,  │   (concat in FieldStorage)
       │                    │     feats)         │
       │                    │     (N, D + ·)     │
       │                    └─────────┬──────────┘
       │                              ▼
       │                    ┌────────────────────┐
       │                    │     ImplicitNet    │   (decoding head;
       │                    │   single tensor in │    single-input MLP in
       │                    │       → D          │    every mode)
       │                    └─────────┬──────────┘
       │                              ▼
       │                       implicit (N, D)
       └──────────────┬───────────────┘
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
        # implicit_net built only when mode in {implicit, hybrid}. It is always
        # a single-input MLP; FieldStorage concatenates the prior (in hybrid)
        # and any aux-bank features before calling it. The head's input dim is
        # therefore the size of the tensor FieldStorage hands it:
        #     base = (L*F if agg == "cat" else F)        # implicit feats
        #     base += sum(aux_bank.feature_dim for aux in auxiliary_banks)
        #     in_dim = D + base if mode == "hybrid" else base
        # When the bank carries a shared trunk, swap `F`/`L*F` for the trunk's
        # output dim `H`.

    def forward(
        self,
        geom: OctreeGeometry,
        prior_only: bool = False,
    ) -> FieldOutput:
        # FieldBank builds `geom` once per call and hands the same cache to
        # every field. Single-field callers can still call:
        #   geom = octree.query(points); field(geom)
        leaf = geom.at_level(1)
        prior = self._explicit(leaf)            if self.cfg.mode != "implicit" else None
        feats = self._implicit_feature(geom)    if self.cfg.mode != "explicit"  else None

        # If the bank carries a shared trunk, the field's head consumes the
        # trunk's output rather than the raw per-point feature.
        if feats is not None and self.bank is not None and self.bank.shared_net is not None:
            feats = self.bank.shared_net(feats)

        if self.cfg.mode == "explicit" or prior_only or feats is None:
            pred, implicit = prior, None
        elif self.cfg.mode == "implicit":
            # head is a single-input MLP; feats already carries any aux-bank
            # concatenation done upstream in _implicit_feature.
            implicit = self.implicit_net(feats)
            pred = implicit
            prior = None
        else:  # hybrid
            # Same single-input head; FieldStorage prepends prior.detach()
            # to feats, so the decoder signature is identical across modes.
            x = torch.cat([prior.detach(), feats], dim=-1)
            implicit = self.implicit_net(x)
            pred = prior.detach() + implicit

        return FieldOutput(leaf.voxel_indices, prior, implicit, pred)
```

`_implicit_feature(geom)` walks `cfg.implicit_feature_level` levels via
`geom.at_level(k)` — never `octree.query(...)` directly — so the per-level
geometry is computed once per forward pass even when several fields walk
overlapping level ranges. Each level's vertex features are trilinearly
interpolated; results are aggregated per the chosen op:

- `cat` → concat along last dim, implicit-net input dim grows to `L*F`.
- `sum` / `mean` → reduce across levels, implicit-net input dim stays `F`.
- `max` → element-wise max across levels (pooling), implicit-net input dim stays `F`.

Aux-bank gathers (when a field's `auxiliary_banks` is non-empty) also go
through `geom.at_level(k)`, so they share the same cache.

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
        # One OctreeGeometry per call, shared across every field. Level 1 is
        # populated eagerly; higher levels are filled lazily on first request
        # and cached on `geom` for the rest of this forward pass.
        geom = self.octree.query(points, voxel_indices)
        return {name: field(geom) for name, field in self.fields.items()}
```

The bank pays the leaf-level `octree.query` once and dispatches it to every
field. Higher-level lookups (`implicit_feature_level > 1`, aux-bank gathers)
hit the per-call cache on `OctreeGeometry`, so the second field asking for
level 3 just reads the dict — no second C++ trip. This is what lets one
octree carry, e.g., `sdf + occ + vl` for joint mapping with no redundant
geometry work.

### Worked example: shared-feature SDF + OCC

SDF in hybrid mode, OCC in implicit mode, both pointing at one
`FeatureBank "geo"` (no shared trunk). The diagram shows what's shared
versus per-field and where gradients meet.

```
                              points (N, 3)
                                  │
                                  ▼
                  ┌────────────────────────────┐
                  │    octree.query(points)    │── (× L levels if implicit_feature_level>1)
                  └─────────────┬──────────────┘
                                │ OctreeGeometry  ── shared by both fields
                                ▼
                  ╔════════════════════════════╗
                  ║      FeatureBank "geo"     ║
                  ║   features (V, F)          ║
                  ║   shared_net = None        ║
                  ╚═════════════┬══════════════╝
                                │  gather 8 vertices per leaf →
                                │  trilinear interp +
                                │  multi-level aggregate {cat,sum,mean,max}
                                ▼
                          feats  (N, F)  or  (N, L·F)        ── shared
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
        SDF (hybrid)                        OCC (implicit)
              │                                   │
              ▼                                   │
  ┌──────────────────────┐                        │
  │ values (V, 1)        │  ← per-field params    │
  │ grads  (V, 3)  [GA]  │                        │
  └──────────┬───────────┘                        │
             │ (GA-)trilinear                     │
             ▼                                    │
        prior  (N, 1)                             │
             │                                    │
             ├── .detach() ───┐                   │
             │                ▼                   ▼
             │      ┌─────────────────┐  ┌─────────────────┐
             │      │   ImplicitNet   │  │   ImplicitNet   │
             │      │   (SDF head)    │  │   (OCC head)    │
             │      │  in: 1 + F → 1  │  │  in: F → 1      │
             │      │  (cat done in   │  │                 │
             │      │   FieldStorage) │  │                 │
             │      └────────┬────────┘  └────────┬────────┘
             │               ▼                    ▼
             │      sdf_implicit (N, 1)    occ_implicit (N, 1)
             │               │                    │
             └───────┬───────┘                    │
                     ▼                            ▼
         sdf_pred = prior.detach()     occ_pred = occ_implicit
                  + sdf_implicit       (occ_prior = None)
         sdf_prior = prior
```

**What's shared, what's per-field**

| Resource | Owner | Note |
| --- | --- | --- |
| `OctreeGeometry` | octree | one `query()` per level, dispatched to both fields |
| `features (V, F)` | `FeatureBank "geo"` | optimized by *both* SDF's and OCC's losses through their respective heads |
| `feats (N, ·)` | bank's gather+interp step | computed once per forward, used by both heads |
| `values (V, 1)`, `grads (V, 3)` | `SDF FieldStorage` | only SDF has explicit storage; OCC has none |
| `ImplicitNet (SDF head)`, `(OCC head)` | each field | independent decoders, both consume the same `feats` |

**Coupling notes**

- The shared `features` parameter is the only place SDF and OCC really meet
  during backprop. Each field's loss flows into the bank through its own
  head, then back into the same `(V, F)` tensor — so the implicit feature
  field learns a representation that has to serve both predictions
  simultaneously.
- If a `shared_net` (trunk) were added to the bank, it would sit between
  `feats` and the two heads, and would also receive gradients from both
  fields. The two heads would then be predicting from a jointly-decoded
  intermediate. Useful when you want the trunk to absorb geometry-common
  structure and leave head-specific shaping to the per-field decoders.
- The Eikonal-coupling decision (see TODO.md "Hybrid mode: pick coupling for
  spatial-gradient losses") only applies to SDF here; OCC has no `prior` in
  implicit mode, so there is no detach choice to make on the OCC side.

### Worked example 2: + VL with auxiliary geometry features

Same SDF (hybrid) + OCC (implicit) on `FeatureBank "geo"` as above, plus a
VL field in implicit mode that owns its own private `FeatureBank "vl_feat"`
and reads `"geo"` as an **auxiliary bank**. This lets the VL decoder ground
its prediction in the same geometry features SDF and OCC are learning,
without contributing its own learnable geometry storage.

```
                                points (N, 3)
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │     octree.query(points)      │── (× L levels)
                     └───────────────┬───────────────┘
                                     │ OctreeGeometry
                     ┌───────────────┴───────────────┐
                     │                               │
                     ▼                               ▼
            ╔════════════════════╗        ╔══════════════════════╗
            ║ FeatureBank "geo"  ║        ║ FeatureBank "vl_feat"║
            ║ features (V, F_g)  ║        ║ features (V, F_v)    ║
            ║ shared_net = None  ║        ║ shared_net = None    ║
            ╚═════════╤══════════╝        ╚═════════╤════════════╝
                      │ gather+interp+agg           │ gather+interp+agg
                      ▼                             ▼
              feats_geo (N, F_g)              feats_vl (N, F_v)
                      │                             │
       ┌──────────────┼─────────────────┐           │
       │              │                 │           │
       ▼              ▼                 ▼           ▼
  SDF (hybrid)   OCC (implicit)   VL (implicit, auxiliary_banks=[
  shared_bank    shared_bank        AuxiliaryBankSpec("geo", detach=…)])
   ="geo"        ="geo"          shared_bank=None (private "vl_feat")
   ──────        ──────            ─────────────────────────────────
   private:           │            decoder input =
     values(V,1)      │              cat(feats_geo [.detach()?], feats_vl)
     grads (V,3)      │                          (N, F_g + F_v)
       │              └─────────────────┐                    │
       │ (GA-)trilinear                 │                    │
       ▼                                │                    │
   prior (N, 1)                         │                    │
       │                                │                    │
       ├── .detach() ──┐                │                    │
       │               ▼                ▼                    ▼
       │      ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐
       │      │ ImplicitNet  │   │ ImplicitNet  │   │   ImplicitNet      │
       │      │  (SDF head)  │   │  (OCC head)  │   │    (VL head)       │
       │      │ in: 1 + F_g  │   │  in: F_g     │   │  in: F_g + F_v     │
       │      │      → 1     │   │      → 1     │   │       → C          │
       │      └──────┬───────┘   └──────┬───────┘   └─────────┬──────────┘
       │             ▼                  ▼                     ▼
       │      sdf_implicit         occ_implicit         vl_implicit (N, C)
       │             │                  │                     │
       └──────┬──────┘                  │                     │
              ▼                         ▼                     ▼
       sdf_pred = prior.detach()   occ_pred = occ_implicit  vl_pred = vl_implicit
                + sdf_implicit
```

**What's shared, what's per-field**

| Resource                        | Owner                  | Consumers                                |
| ------------------------------- | ---------------------- | ---------------------------------------- |
| `OctreeGeometry`                | octree                 | all three fields                         |
| `features (V, F_g)` ("geo")     | `FeatureBank "geo"`    | SDF head, OCC head, VL head (auxiliary)  |
| `feats_geo (N, F_g)`            | bank gather step       | computed once, dispatched to 3 heads     |
| `features (V, F_v)` ("vl_feat") | `FeatureBank "vl_feat"`| VL head only                             |
| `values (V, 1)`, `grads (V, 3)` | SDF FieldStorage       | SDF head                                 |
| `ImplicitNet` per field         | each field             | independent decoders                     |

**Coupling notes**

- **`AuxiliaryBankSpec("geo", detach=False)`** lets VL's losses flow back
  into `geo.features`. The geometry bank then learns a representation that
  has to serve SDF priors, OCC priors, *and* VL prediction — strongest
  joint-feature-learning regime, but bad VL labels can perturb geometry.
- **`AuxiliaryBankSpec("geo", detach=True)`** makes VL a *consumer only*:
  it reads geometry features but its losses do not update `geo.features`.
  Same blame-attribution rationale as `prior.detach()` in hybrid mode.
  Useful when VL supervision is noisy or when geometry should converge
  unaffected by the VL head.
- **`vl_feat` is always private and one-way.** VL's bank is owned by VL
  alone, and no other field declares it as auxiliary, so the gradient flow
  into `vl_feat.features` comes only from the VL head — no cross-field leak
  on the VL side regardless of the `detach` choice for `geo`.
- **Aux banks must be shared banks.** `FieldBank.__init__` rejects an
  `AuxiliaryBankSpec` whose `name` points at a private bank (a bank not
  declared on `shared_banks`); pulling from another field's private storage
  would create implicit coupling that's invisible at the call site.

### Worked example 3: distilled multi-teacher VL

Extend example 2 with a *second* VL field. Both VL fields share the same
`vl_feat` bank with different decoder heads, distilling multiple VL teachers
(e.g., CLIP-512 and DINO-768) into one compact `(V, F_v)` latent — with
`F_v` deliberately much smaller than either teacher's dim so the bank is
forced to encode information that serves both reconstructions.

```
                                  points (N, 3)
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │     octree.query(points)      │── (× L levels)
                       └───────────────┬───────────────┘
                                       │ OctreeGeometry
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
              ╔══════════════════════╗     ╔══════════════════════╗
              ║ FeatureBank "geo"    ║     ║ FeatureBank "vl_feat"║
              ║ features (V, F_g)    ║     ║ features (V, F_v)    ║
              ║ shapers: SDF, OCC,   ║     ║ shapers: VL-1, VL-2  ║
              ║          VL-1, VL-2  ║     ║ readers: VL-1, VL-2  ║
              ║          (if aux not ║     ║                      ║
              ║          detached)   ║     ║  distilled compact   ║
              ║                      ║     ║  latent              ║
              ╚══════════╤═══════════╝     ╚══════════╤═══════════╝
                         │ gather+interp+agg          │ gather+interp+agg
                         ▼                            ▼
                  feats_geo (N, F_g)             feats_vl (N, F_v)
                         │                            │
       ┌─────┬───────────┼───────────┐                │
       │     │           │           │                │
       ▼     ▼           │           │           ┌────┴────┐
      SDF   OCC          │           │           │         │
     (hyb)(impl)         │           │           │         │
       │    │            ▼           ▼           ▼         ▼
       │    │       VL-1 aux    VL-2 aux       VL-1      VL-2
       │    │       "geo"       "geo"          own       own
       │    │       (detach=?)  (detach=?)     feats     feats
       │    │            │           │           │         │
       │    │            │           │           │         │
       │    │            └────┐  ┌───┘           │         │
       │    │                 ▼  ▼               │         │
       │    │       ┌─────────────────┐     ┌────┴────┐    │
       │    │       │ cat(feats_geo,  │◄────┘         │    │
       │    │       │     feats_vl)   │               │    │
       │    │       └────────┬────────┘               │    │
       │    │                │      (and the          │    │
       │    │                │       symmetric cat    │    │
       │    │                │       for VL-2 ─────────────┘
       │    │                │       with feats_vl    │
       │    │                │       from the same    │
       │    │                │       bank gather)     │
       │    │                ▼                        ▼
       │    │       ┌───────────────┐      ┌───────────────┐
       │    │       │  ImplicitNet  │      │  ImplicitNet  │
       │    │       │  (VL-1 head)  │      │  (VL-2 head)  │
       │    │       │  → C1 = 512   │      │  → C2 = 768   │
       │    │       │   (CLIP)      │      │   (DINO)      │
       │    │       └───────┬───────┘      └───────┬───────┘
       │    │               ▼                      ▼
       ▼    ▼        vl1_implicit (N, C1)   vl2_implicit (N, C2)
   (SDF/OCC               │                      │
    heads as              ▼                      ▼
    in ex. 2)        vl1_pred = vl1_implicit   vl2_pred = vl2_implicit
```

**Config sketch**

```python
FieldStorageConfig(
    name="vl_clip",  mode="implicit", output_dim=512,
    shared_bank="vl_feat", implicit_feature_dim=F_v,
    auxiliary_banks=[AuxiliaryBankSpec(name="geo", detach=False)],
    implicit_net_cfg=ImplicitNetConfig(out_dim=512, ...),
)
FieldStorageConfig(
    name="vl_dino", mode="implicit", output_dim=768,
    shared_bank="vl_feat", implicit_feature_dim=F_v,   # same as vl_clip
    auxiliary_banks=[AuxiliaryBankSpec(name="geo", detach=False)],
    implicit_net_cfg=ImplicitNetConfig(out_dim=768, ...),
)
# FieldBank.shared_banks = [("geo", F_g, None), ("vl_feat", F_v, None)]
```

**What's new vs. example 2**

| Resource                        | Owner                  | Consumers (this example)                     |
| ------------------------------- | ---------------------- | -------------------------------------------- |
| `features (V, F_v)` ("vl_feat") | `FeatureBank "vl_feat"`| **both VL-1 and VL-2 heads** (was: VL only)  |
| `feats_vl (N, F_v)`             | bank gather step       | computed once, dispatched to both VL heads   |
| `ImplicitNet (VL-1)`, `(VL-2)`  | each VL field          | independent decoders, both consume `cat(feats_geo, feats_vl)` |

**Distillation coupling notes**

- **`vl_feat` is a capacity-bottlenecked joint latent.** `F_v` is much
  smaller than `C1` and `C2`. Both VL losses flow into the same `(V, F_v)`
  tensor, so the bank learns whichever subspace serves both reconstructions
  — the common information between the two VL feature spaces. This *is* the
  distillation. If `F_v ≥ C1 + C2` there is no compression pressure and the
  bank can simply concatenate the two targets; the design only works when
  `F_v` is intentionally narrow.
- **The heads are the per-teacher "decoders" of distillation.** Each
  `ImplicitNet (VL-k)` projects the shared latent (plus geometry context)
  into one teacher's space. Replacing a teacher = swapping a head and a
  target tensor; the geometry bank and the distilled bank are reusable.
- **Aux `geo` is the geometry crutch.** Without it, `vl_feat` would have to
  encode both VL content *and* the spatial structure needed to vary
  smoothly across the octree. With `geo` available as auxiliary input, the
  VL bank can focus on the semantic residual and let `geo` carry the
  geometry side. `detach=False` lets the VL losses also push `geo` toward
  "good for VL" geometry; `detach=True` keeps `geo` purely SDF/OCC-shaped.
- **Coupling between the two VL tasks is purely representational.** Each
  VL head has its own `ImplicitNet` params that receive gradients only from
  their own loss; the cross-task coupling flows entirely through the shared
  `(V, F_v)` tensor. The two tasks negotiate over the bank's contents but
  do not share decoder weights.
- **Output dims may differ; `implicit_feature_dim` must agree.**
  `vl_clip.output_dim=512` and `vl_dino.output_dim=768` are fine, but both
  fields must declare the same `implicit_feature_dim=F_v` because that's
  the bank's storage dim. Same constraint as the SDF/OCC shared-bank case,
  surfaced again here.

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

Strategy: build the abstraction bottom-up; verify it outside-in (explicit →
implicit → hybrid), each verification stage gated by a YAML migration so the
existing trainers can drive the new code unmodified. The four phases below
correspond to three observable checkpoints (VL demo, `OccTrainer`,
`SdfTrainer`) plus a finalization phase.

### Summary: files & symbols

All paths relative to `src/oren_vl/`. "new (rename)" = move/rename existing
file. Wrappers preserve the existing `(voxel_indices, prior, residual, pred)`
return tuple so `SdfTrainer`/`OccTrainer` unpacking is untouched until
phase 4.

| File | Change | Symbols introduced / modified | Phase |
| --- | --- | --- | --- |
| `oren/oren/field_output.py` | new | `LevelGeometry`, `OctreeGeometry` (per-call multi-level cache), `FieldOutput` | 0 |
| `oren/oren/implicit_net.py` | new (rename of `residual_net.py`) | `ImplicitNet`, `ImplicitNetConfig` (+ per-mode signature) | 0 |
| `oren/oren/semi_sparse_octree_base.py` | modify | `+ query(points, voxel_indices=None) -> OctreeGeometry`, `+ _compute_level_geometry(points, level) -> LevelGeometry`; legacy `forward()` routes through them | 0 |
| `oren/oren/octree_config.py` | modify | `DeprecationWarning` on legacy field flags | 0 |
| `oren/oren/field_storage_config.py` | new | `FieldStorageConfig`, `AuxiliaryBankSpec` | 0 |
| `oren/oren/feature_bank.py` | new | `FeatureBank` ((V, F) + optional `shared_net`) | 1 |
| `oren/oren/field_storage.py` | new | `FieldStorage` (explicit only in phase 1; implicit in 2; hybrid in 3) | 1–3 |
| `oren/oren/field_bank.py` | new | `FieldBank` (octree + fields + shared banks + aux wiring) | 2 |
| `scripts/migrate_field_configs.py` | new | YAML rewriter (writes `.bak`) | 2 |
| `oren_vl/oren_vl/demo_semi_sparse_octree_vl.py` | modify | swap to `FieldStorage(mode="explicit")`; also fix stale `semi_sparse_octree_v2` import | 1 |
| `oren/oren/occ_network.py` | modify | thin wrapper around `FieldStorage`; `OccNetworkConfig` carries one `FieldStorageConfig` | 2 |
| `oren/oren/sdf_network.py` | modify | thin wrapper around `FieldStorage`; `SdfNetworkConfig` carries one `FieldStorageConfig` | 3 |
| `oren/oren/multi_field_trainer.py` | new | `MultiFieldTrainer` (composes per-field trainers, one shared optimizer) | 4 |
| `oren/oren/semi_sparse_octree_base.py` | modify (phase 4) | remove `ModelOutput`, legacy `forward()`, field buffers | 4 |
| `oren/oren/octree_config.py` | modify (phase 4) | delete deprecated flags | 4 |

### Phase 0 — Scaffolding (no behavior change)

1. Add `OctreeGeometry` and `FieldOutput` dataclasses in `field_output.py`.
   Use `FieldOutput.implicit` from the start (not `.residual`).
2. Rename `ResidualNet` → `ImplicitNet`; move `residual_net.py` →
   `implicit_net.py`. The head stays a **single-input MLP** — `forward(x)` —
   in every mode; FieldStorage is responsible for any concatenation (prior
   in hybrid mode, aux-bank features when present). Compute the head's
   `in_dim` at construction from the (mode, aggregation, aux-banks)
   combination: `in_dim = (L·F if cat else F) + Σ aux_dim + (D if hybrid)`.
   This actually *reverts* today's `ResidualNet.forward(prior, features)`
   (which concats internally) back to a plain MLP signature. Update
   `sdf_network.py` / `occ_network.py` imports mechanically and move their
   `torch.cat([prior, features], dim=-1)` call to the wrapper site —
   runtime behavior unchanged at this step.
3. Add `query(points, voxel_indices=None) -> OctreeGeometry` to
   `SemiSparseOctreeBase`. The returned object eagerly populates level 1
   (using `voxel_indices` if provided), and exposes
   `at_level(level) -> LevelGeometry` which lazily fills and caches higher
   levels via the new `_compute_level_geometry` helper (extracted from
   today's `forward(...)` body around the existing
   `find_voxel_indices(points, False, level)` site at
   `semi_sparse_octree_base.py:289`). Keep the legacy
   `forward(...) -> ModelOutput` alive but route its geometry lookups
   through this same accessor so the new cache exercises the existing
   path. Emit `DeprecationWarning` when `enable_sdf` / `enable_occupancy`
   / `enable_implicit` / `implicit_feature_dim` / `implicit_num_levels` /
   `init_occ_prior` are read.
4. Add `FieldStorageConfig` + `AuxiliaryBankSpec` in
   `field_storage_config.py`. No callers yet.

### Phase 1 — Explicit mode + VL demo verification

5. Implement `FeatureBank` in `feature_bank.py`: `features (init_voxel_num, F)`
   parameter + optional `shared_net: ImplicitNet`. Implement a resize hook
   that grows `features` in lock-step with the C++ tree.

   **C++ side requires no changes.** The `erl_geometry` bindings already
   expose `num_vertices` as a read-only property
   (`pybind11_semi_sparse_octree.hpp:59`), every `*_tensor` accessor
   `.clone()`s the buffer (so Python tensors are snapshots, decoupled from
   C++ internal growth), and only one Python site mutates the tree:
   `SemiSparseOctree.insert_voxels:38` (every other caller routes through
   `insert_points`). The hook is therefore pure Python:

   - Add a `_resize_observers: list[Callable[[int], None]]` list to
     `SemiSparseOctreeBase`, with `register_resize_observer(fn)` /
     `unregister_resize_observer(fn)` methods.
   - Wrap `insert_voxels` in `SemiSparseOctreeBase`: after the subclass's
     `_insert_voxels_impl(...)` returns, read `self.sso.num_vertices`,
     compare against `self._last_known_capacity`, and if it grew, iterate
     observers with the new capacity.
   - `FeatureBank.__init__` registers a callback that allocates a new
     `(new_capacity, F)` parameter, copies the old contents into the
     prefix, and replaces `self.features` (use `nn.Parameter` rebinding +
     `nn.utils.parametrize`-style optimizer-state migration if the bank
     is already attached to an optimizer; for the demo path the bank is
     not optimized so a plain replacement suffices).
   - `FieldStorage` registers its own callback for `values (V, D)` and
     (optional) `grads (V, 3)`, same pattern.

   This also fixes a latent bug in today's code: `sdf_priors` etc. on
   `SemiSparseOctreeBase` are sized once at `init_voxel_num` and never
   grow, so if a dataset pushes `num_vertices` past 200 000 (the default),
   `vertex_indices ≥ init_voxel_num` would index out of bounds. The
   trainers have been getting away with it because the default is
   conservatively large; the resize hook is what makes the new field
   abstraction safe to grow unbounded.

   *Rejected alternative:* a C++-side observer list (pybind
   `std::function` registered into `InsertKeys`) would re-enter Python
   from the C++ insertion path on every grow event. That adds GIL
   re-acquisition cost on what is meant to be a hot path, and buys
   nothing — every insertion already goes through exactly one Python
   entry point (`insert_voxels`), so the polling check has nowhere to
   miss. **Do not** add a Python callback to the C++ runtime for this.
6. Implement `FieldStorage` explicit branch in `field_storage.py`:
   `_explicit(geom)` calls `ga_trilinear` when `D == 1 and gradient_augmentation`,
   plain trilinear otherwise. Implicit/hybrid paths raise `NotImplementedError`.
   Same resize-hook plumbing for `values (V, D)` and `grads (V, 3)`.
7. Migrate `oren_vl/oren_vl/demo_semi_sparse_octree_vl.py`:
   - Fix the broken import `from oren.semi_sparse_octree_v2 import SemiSparseOctree`
     → `from oren.semi_sparse_octree import SemiSparseOctree` (the `_v2`
     module no longer exists; only a stale `.pyc` remains).
   - Construct one `FieldStorage(name="vl", output_dim=dataset.channels,
     mode="explicit", gradient_augmentation=False)` and scatter the
     per-pixel VL features into `field.values` instead of
     `octree.implicit_features`.
8. **Checkpoint — VL demo parity.** Rerun the demo on a fixed dataset, both
   `overwrite` and `running_average` modes. Compare the saved tensors
   against a pre-refactor snapshot vertex-by-vertex (bit-exact for
   `overwrite`, ≤ 1e-6 for `running_average`).

### Phase 2 — Implicit mode + `OccTrainer` verification

9. Implement `FieldStorage._implicit_feature(geom)`. Always go through
   `geom.at_level(k)` — never `octree.query(...)` directly — so the per-call
   `OctreeGeometry` cache mediates: the first field on the bank to need
   level k pays for it, every subsequent field (or aux-bank gather) on the
   same forward pass reads the cached `LevelGeometry` from the dict.
   Aggregate across levels with `cat` / `sum` / `mean` / `max`. Apply
   `bank.shared_net` (if present) before the per-field head.
10. Implement `FieldStorage` implicit forward: `pred = implicit_net(feats)`,
    `prior = None`.
11. Implement `FieldBank` in `field_bank.py`: holds `octree`, `ModuleDict`
    of fields, `ModuleDict` of named `FeatureBank`s. At construction,
    validate:
    - every field naming `shared_bank=X` agrees on `implicit_feature_dim`
      and that dim matches `banks[X].features.shape[1]`;
    - every `AuxiliaryBankSpec.name` references a **shared** bank (rejects
      pointing at another field's private bank);
    - the field's head input dim accounts for aux-bank concatenation.
    Wire `auxiliary_banks` per field: gather the aux bank's per-point
    features via `geom.at_level(k)` (so the cache absorbs any overlap with
    the owning fields' own multi-level walks), apply `.detach()` per
    `AuxiliaryBankSpec.detach`, concat to this field's features before the
    head.
12. Land `scripts/migrate_field_configs.py`. Mappings:
    - `octree_cfg.enable_sdf|enable_occupancy: true` → emit one
      `fields[{name: sdf|occ, mode: hybrid|explicit|implicit, ...}]` entry
      derived from `enable_implicit` and `enable_*` combinations (see
      table in the "Migration" section).
    - `octree_cfg.implicit_feature_dim` → `fields[*].implicit_feature_dim`.
    - `octree_cfg.implicit_num_levels` → `fields[*].implicit_feature_level`.
    - `octree_cfg.gradient_augmentation` → `fields[*].gradient_augmentation`.
    - `octree_cfg.init_occ_prior` → `fields[occ].explicit_prior_init`.
    - `residual_net_cfg` block → `fields[*].implicit_net_cfg`.
    Writes `<file>.bak` next to each rewritten file.
13. **Migrate the OCC YAMLs first**: run the script on
    `src/oren_vl/configs/trainer-replica-occ.yaml`,
    `src/oren_vl/oren_ros/configs/trainer-ros-occ.yaml`,
    `src/oren_vl/oren_ros/configs/trainer-ros-newer-college-occ.yaml`.
    Spot-check each diff before deleting `.bak`s.
14. Migrate `OccNetwork` to a thin wrapper around `FieldStorage`.
    `OccNetworkConfig` now holds `octree_cfg` (geometry-only) and one
    `FieldStorageConfig`. The wrapper's `forward()` still returns
    `(voxel_indices, occ_prior, occ_residual, occ_pred)` so
    `occ_trainer.py:107, 117` unpack sites are untouched. The local-name
    rename `occ_residual` → `occ_implicit` waits until phase 4.
15. **Checkpoint — OCC parity.** Run
    `python oren/oren/occ_trainer.py --config configs/trainer-replica-occ.yaml`
    on a fixed seed and confirm parity against a pre-refactor snapshot:
    loss curves (tolerance e.g. ≤ 1e-4), occupancy-mesh metrics, and a
    sample of `occ_prior`/`occ_pred` values at fixed query points.

### Phase 3 — Hybrid mode + `SdfTrainer` verification

16. Implement `FieldStorage` hybrid forward:
    `x = torch.cat([prior.detach(), feats], dim=-1)`;
    `implicit = implicit_net(x)`;
    `pred = prior.detach() + implicit`.
    The head stays single-input — concat lives in FieldStorage — so its
    `in_dim` just grows by `D` over the implicit-mode value.
17. **Migrate the SDF YAMLs**: run the migration script on
    `src/oren_vl/configs/trainer-replica.yaml`,
    `src/oren_vl/configs/trainer-newer_college.yaml`,
    `src/oren_vl/configs/replica.yaml`,
    `src/oren_vl/configs/replica/*.yaml`,
    `src/oren_vl/configs/replica_room0_view{0,1}.yaml`,
    `src/oren_vl/oren_ros/configs/trainer-ros-sdf.yaml`,
    `trainer-ros-realsense-sdf.yaml`, `trainer-ros-realsense-lab-sdf.yaml`,
    `trainer-ros-drone-sdf-{1,2,3}.yaml`,
    `trainer-ros-newer-college-sdf.yaml`.
18. Migrate `SdfNetwork` to a thin wrapper around `FieldStorage` (mirrors
    `OccNetwork`). Preserve the 4-tuple return contract so
    `sdf_trainer.py:114, 234, 235` are untouched.
19. **Checkpoint — SDF parity.** Run
    `python oren/oren/sdf_trainer.py --config configs/replica.yaml` and
    `--config configs/trainer-newer_college.yaml` end-to-end and confirm
    SDF parity (loss curves, marching-cubes metrics from `evaluator_oren`).

### Phase 4 — Finalization

20. Trainers consume `FieldBank` directly. `SdfTrainer` / `OccTrainer`
    construct a single-field `FieldBank` and read the named `FieldOutput`,
    dropping the `SdfNetwork` / `OccNetwork` wrapper indirection. The
    4-tuple return contract retires here; rename `*_residual` → `*_implicit`
    at the trainer unpack sites.
21. Implement `MultiFieldTrainer` in `multi_field_trainer.py`. One
    `octree.query(points)` per step — the resulting `OctreeGeometry`
    carries the per-level cache, so every child trainer's
    `FieldBank.forward(geom)` reuses the same memoized levels. Per-child
    criteria, summed loss with per-field weights, one optimizer over
    `bank.parameters()`, log-dict merging with `<field_name>/` prefix.
    First joint workflow: SDF (hybrid) + scattered VL (explicit) on a
    shared octree.
22. Delete the deprecated flags from `OctreeConfig` (`enable_sdf`,
    `enable_occupancy`, `enable_implicit`, `init_occ_prior`,
    `implicit_feature_dim`, `implicit_num_levels`, `gradient_augmentation`).
    Delete `SemiSparseOctreeBase.ModelOutput` and the legacy `forward()`.
    Any YAML not migrated by this point now fails loudly at config load.

### Dependency outline

```
  Phase 0    FieldOutput/OctreeGeometry ── ImplicitNet rename ── octree.query() ── FieldStorageConfig
                 │                                                       │
  Phase 1        └── FeatureBank ── FieldStorage(explicit) ── VL demo ✓
                                                              │
  Phase 2        FieldStorage(implicit) ── FieldBank ── migrate_field_configs.py
                                                         │
                                         OCC YAMLs ── OccNetwork wrapper ── OccTrainer ✓
  Phase 3      FieldStorage(hybrid)
                                         SDF YAMLs ── SdfNetwork wrapper ── SdfTrainer ✓
  Phase 4    trainers consume FieldBank directly ── MultiFieldTrainer ── OctreeConfig cleanup
```
