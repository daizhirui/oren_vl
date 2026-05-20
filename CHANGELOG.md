# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions apply to all four packages in this monorepo (`oren`, `oren_msgs`,
`oren_ros`, `oren_vl`) unless noted otherwise.

## [Unreleased]

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
