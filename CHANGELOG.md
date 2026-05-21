# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions apply to all four packages in this monorepo (`oren`, `oren_msgs`,
`oren_ros`, `oren_vl`) unless noted otherwise.

## [Unreleased]

### Added
- `VlEvaluator` (`oren/oren/evaluator_vl.py`): per-point L1 / L2 (MSE, RMSE)
  / cosine-similarity metrics for predicted vs. ground-truth VL features,
  with separate aggregates for the final `vl` output and the `vl_prior`
  branch. Provides `evaluate_test_set` (load `(points, gt_features)` from
  `.npy` files) and `evaluate_dataset` (iterate any `VlFrame` stream)
  entry points.
- `VlTrainer.evaluate()` is now wired up (previously a no-op): writes
  metrics to the text log, TensorBoard, and a `vl_metrics.yaml` snapshot
  under the run's `misc_dir`.
- Per-frame CPU / GPU memory profiling (`CpuMemoryProfiler`,
  `GpuMemoryProfiler`) on `TrainerBase`, with TensorBoard logging after
  each frame / batch.
- `TrainerConfig.final_save_profiling_stats`: dump aggregated timer +
  memory stats to `<misc_dir>/profiling_stats.yaml` at the end of
  `train()`. The headline `total_wall_time_s` is recorded
  unconditionally; per-label breakdowns require `profiling: true`.
- `TimerRecords` export now reports `Std (ms)` and `Max (ms)` columns
  alongside mean / total / count (O(1) bookkeeping per label).
- PCA precomputation in `dataset/generate_vl_features.py`: fits three
  components on a uniform sample of valid (depth > 0) pixels and writes
  `pca.npz` (components, mean, explained variance) into the feature
  bundle. `visualize_octree_vl.py` / `visualize_pcd_vl.py` auto-detect
  and reuse it so PCA colors stay consistent across visualizations of
  the same scene.
- `FieldStorageConfig.track_used_vertices` and matching
  `FeatureBankConfig.track_used_vertices`: when enabled, `FieldStorage`
  / `FeatureBank` maintain `values_used` / `features_used` bool masks
  that are OR-marked on every `Fuser.scatter` / `Fuser.gather`.
  State-dict serialization then stores only `(indices, rows_at_indices)`
  for touched vertices, substantially shrinking checkpoints
  (particularly in optimize modes where many vertices are never
  gathered).
- `STUDY-VL-Fusion.md`: writeup of the VL-fusion ablation results.

### Changed
- `oren_vl` package removed. The VL code already lives under
  `oren/oren/` after the 0.2.0 reorganization; the `oren_vl/` directory
  only contained empty scaffolding (`package.xml`, `setup.py`, empty
  `__init__.py`s) and a single demo, now relocated to
  `scripts/demo_semi_sparse_octree_vl.py`. The workspace now ships
  three packages: `oren`, `oren_msgs`, `oren_ros`.
- `Fuser.scatter` / `Fuser.gather` accept an optional `touched_mask`
  argument that subclasses OR-mark with the global vertex indices they
  accessed; this drives sparse checkpoint storage. Subclasses that
  ignore the mask produce dense state-dicts as before.
- `NearestVertexFuser.scatter` (`running_average` mode) rewritten to
  operate on the unique touched vertices only (`O(U*D)`), replacing the
  dense scratch + masked write-back that scaled with octree capacity.
- `visualize_octree_vl.py` now loads `VlTrainer` checkpoints directly
  (`ckpt/final.pth` + sibling `bak/config.yaml`) instead of the legacy
  demo `.pt` file. Picks the field to render via `--field-name`
  (default `vl`) and uses the per-vertex `prior_fuser.weight_sum` (when
  present) to decide which leaves received scatter updates.
- `deps/erl_geometry` bumped to a newer revision (logging refactor;
  `init.cpp` / `eigen.cpp` split out from headers).
- `configs/trainer-replica-sdf.yaml`: `exp_name` renamed from
  `replica_for_planning` to `replica_sdf`.

### Fixed
- `FieldStorage` implicit branch: `ImplicitNet.output_dim` is now
  propagated from `FieldStorageConfig.output_dim` at construction time
  (previously left at the user-supplied `implicit_net_cfg` default,
  which silently produced the wrong output width for multi-dim fields).
- `SemiSparseOctree._load_from_state_dict` fires registered resize
  observers before delegating to the parent loader, so sibling
  per-vertex buffers (`FieldStorage.values`, `FeatureBank.features`,
  fuser scratch tensors) grow to the saved capacity in time for the
  in-place `copy_`. Without this, checkpoints saved past the initial
  capacity failed to load with a shape-mismatch error.
- Off-by-one in the offline-epoch tqdm message (`Epoch N/M` instead of
  `Epoch N-1/M`).

## [0.2.0] - 2026-05-20

### Added
- `oren_vl` package: vision-language extension that fuses CLIP features into the
  octree field via `FieldStorage` (see `DESIGN-VL-Fusion.md`).
- `OccNetwork` / `occ_trainer.py`: occupancy-field training alongside SDF.
- `MultiFieldTrainer`: joint training of SDF + VL (and other) fields through
  `FieldBank` / `FeatureBank`.
- `oren_ros` package: ROS 2 nodes (`mapping_node`, `sdf_query_node`,
  `clock_node`), launch file, and configs for online SDF mapping on bags.
- `oren_msgs` package: `QueryScalarField`, `SaveMesh`, `SaveModel` services.
- Bag preprocessing script `scripts/split_posed_depth_bag.py` for
  `grad_sdf_interface/msg/PosedDepth` bags.

### Changed
- **FieldStorage refactor**: split per-vertex field state out of the octree.
  `semi_sparse_octree*.py` now holds geometry only; `FieldStorage` owns values
  and features and selects one of three regimes (`explicit`, `implicit`,
  `hybrid`). See `DESIGN-FieldStorage.md`.
- `SdfNetwork` / `OccNetwork` reduced to thin adapters that bind a single
  `FieldStorage` to the training loop; regime is chosen in YAML.
- Repository reorganised from a single package into a ROS 2 colcon workspace
  with four packages; non-ROS algorithm code moved under `oren/`.
- Removed `tiny-cuda-nn` dependency.

### Removed
- Legacy `sparse_octree` implementation (superseded by `semi_sparse_octree`).

[Unreleased]: https://github.com/daizhirui/oren_vl/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/daizhirui/oren_vl/releases/tag/v0.2.0
