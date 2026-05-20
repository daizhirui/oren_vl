"""One-shot YAML migration: legacy octree-cfg field flags  FieldStorageConfig.

Translates the pre-FieldStorage-refactor model config shape:

    model:
      octree_cfg:
        enable_sdf|enable_occupancy|enable_implicit: ...
        init_occ_prior, gradient_augmentation,
        implicit_feature_dim, implicit_num_levels: ...
      residual_net_cfg:
        cfg_identifier: oren.residual_net.ResidualNetConfig
        ...

into the new shape:

    model:
      octree_cfg:                        # geometry-only after this migration
        resolution, tree_depth, ...
      field:                             # new FieldStorageConfig block
        cfg_identifier: oren.field_storage_config.FieldStorageConfig
        name: sdf|occ
        mode: hybrid|explicit|implicit
        gradient_augmentation, explicit_prior_init,
        implicit_feature_dim, implicit_feature_level,
        implicit_feature_aggregation, auxiliary_banks
        implicit_net_cfg:                # renamed from residual_net_cfg
          cfg_identifier: oren.implicit_net.ImplicitNetConfig
          ...

Behavior:
  * Operates per-file with no `base_config:` resolution. For inheriting
    configs, run the script separately on the base and on each child; the
    child gets a partial `field` block with whatever legacy keys it
    overrode. Phase 4's final cleanup is a good time to flatten the
    inheritance hierarchies if desired.

  * Writes `<file>.bak` next to the migrated YAML.

  * Idempotent: a YAML that already has `model.field` is left alone.

Mode mapping for `(enable_sdf, enable_occupancy, enable_implicit)`:
   (T, _, T)  sdf  hybrid          (the SDF default — explicit prior + implicit residual)
   (T, _, F)  sdf  explicit
   (F, T, T)  occ  hybrid          (OCC with learned per-vertex prior)
   (F, T, F)  occ  explicit
   (F, F, T)  occ  hybrid          (legacy implicit-only OCC — preserved as hybrid with
                                     `explicit_prior_init=0` + `gradient_augmentation=false` so
                                     the "+1 zero-prior" head architecture is preserved bit-equal.
                                     A *cleaner* post-refactor target is `mode="implicit"`, but
                                     that changes the head's input dim by 1 and is not parity-
                                     preserving — left as a manual edit for users who want it.)
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from typing import Optional

import ruamel.yaml


# DEFAULTS — must match OctreeConfig.__init__ so we infer the right field
# spec for YAMLs that don't explicitly set every legacy flag.
_LEGACY_DEFAULTS = {
    "enable_sdf": True,
    "enable_occupancy": False,
    "enable_implicit": True,
    "init_occ_prior": 0.0,
    "implicit_feature_dim": 4,
    "implicit_num_levels": 3,
    "gradient_augmentation": True,
}


def _make_yaml() -> ruamel.yaml.YAML:
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=4, sequence=4, offset=2)
    yaml.width = 120
    return yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` onto `base`. Dict values are merged
    key-by-key; everything else is replaced (override wins for scalars and
    lists). Returns a new dict, leaving inputs untouched."""
    out = {}
    for k, v in base.items():
        if k in override:
            if isinstance(v, dict) and isinstance(override[k], dict):
                out[k] = _deep_merge(v, override[k])
            else:
                out[k] = override[k]
        else:
            out[k] = v
    for k, v in override.items():
        if k not in base:
            out[k] = v
    return out


def _resolve_inheritance(yaml_path: pathlib.Path, yaml: ruamel.yaml.YAML) -> tuple[dict, bool]:
    """Resolve `base_config:` chains. Returns (merged_dict, inherits).
    `inherits` is True iff the file had `base_config:` (and is now flattened)."""
    with open(yaml_path, "r") as f:
        data = yaml.load(f)
    if not isinstance(data, dict):
        return data, False

    base_ref = data.pop("base_config", None)
    if base_ref is None:
        return data, False

    # Resolve the base path relative to this file's directory.
    base_path = (yaml_path.parent / str(base_ref)).resolve()
    base_data, _ = _resolve_inheritance(base_path, yaml)
    return _deep_merge(base_data, data), True


def _migrate_octree_to_field(model: dict) -> Optional[str]:
    """Mutate `model` in place. Returns a one-line description of what was
    done (or None if there was nothing to migrate)."""
    if "field" in model:
        return None  # already migrated

    octree_cfg = model.get("octree_cfg")
    legacy_keys = {
        "enable_sdf", "enable_occupancy", "enable_implicit",
        "init_occ_prior", "implicit_feature_dim", "implicit_num_levels",
        "gradient_augmentation",
    }

    if octree_cfg is None or not (legacy_keys & set(octree_cfg.keys())):
        # No octree_cfg block, or block has none of the legacy keys — but
        # residual_net_cfg might still need renaming. Handle that separately.
        if "residual_net_cfg" in model:
            return _rename_residual_only(model)
        return None

    # Pull legacy values, defaulting to OctreeConfig defaults when absent.
    legacy = {}
    for k in legacy_keys:
        if k in octree_cfg:
            legacy[k] = octree_cfg.pop(k)
        else:
            legacy[k] = _LEGACY_DEFAULTS[k]

    enable_sdf = legacy["enable_sdf"]
    enable_occ = legacy["enable_occupancy"]
    enable_impl = legacy["enable_implicit"]

    # Determine field name and mode. See mapping in module docstring.
    explicit_prior_init = 0.0
    gradient_augmentation = legacy["gradient_augmentation"]

    if enable_sdf:
        name = "sdf"
        mode = "hybrid" if enable_impl else "explicit"
        # SDF doesn't use init_occ_prior; explicit_prior_init defaults to 0.
    elif enable_occ:
        name = "occ"
        mode = "hybrid" if enable_impl else "explicit"
        explicit_prior_init = legacy["init_occ_prior"]
    elif enable_impl:
        # Legacy "implicit-only OCC": preserve the +1 zero-prior architecture
        # for bit-equal parity. The MLP still sees a zero prior concatenated.
        name = "occ"
        mode = "hybrid"
        explicit_prior_init = 0.0
        gradient_augmentation = False
    else:
        # Pure geometry octree — no field block to emit. Caller handled by
        # falling through (the new code path supports this; the legacy code
        # path's all-disabled assertion was relaxed in phase 1).
        return "geometry-only (no field block emitted)"

    field: dict = {
        "cfg_identifier": "oren.field_storage_config.FieldStorageConfig",
        "name": name,
        "output_dim": 1,
        "mode": mode,
        "explicit_prior_init": float(explicit_prior_init),
        "gradient_augmentation": bool(gradient_augmentation),
        "implicit_feature_dim": int(legacy["implicit_feature_dim"]),
        "implicit_feature_level": int(legacy["implicit_num_levels"]),
        "implicit_feature_aggregation": "cat",
        "shared_bank": None,
        "auxiliary_banks": [],
    }

    if "residual_net_cfg" in model:
        rn = model.pop("residual_net_cfg")
        rn["cfg_identifier"] = "oren.implicit_net.ImplicitNetConfig"
        # Inner `implicit_feature_dim` was renamed to `input_dim` when the
        # field migrated from `ResidualNetConfig` to `ImplicitNetConfig`. The
        # value is overwritten at runtime by FieldStorage anyway, so the
        # rename is purely surface-level.
        if "implicit_feature_dim" in rn:
            rn["input_dim"] = rn.pop("implicit_feature_dim")
        field["implicit_net_cfg"] = rn

    model["field"] = field
    return f"{name}/{mode} field emitted; legacy octree flags removed"


def _rename_residual_only(model: dict) -> str:
    """When the YAML has only residual_net_cfg (no legacy octree flags),
    just rename the identifier in place — this is the case for child YAMLs
    that override the MLP architecture without touching field flags."""
    rn = model["residual_net_cfg"]
    if rn.get("cfg_identifier") == "oren.residual_net.ResidualNetConfig":
        rn["cfg_identifier"] = "oren.implicit_net.ImplicitNetConfig"
        if "implicit_feature_dim" in rn:
            rn["input_dim"] = rn.pop("implicit_feature_dim")
        return "residual_net_cfg cfg_identifier renamed only"
    return "residual_net_cfg already on new identifier — no-op"


def migrate_file(yaml_path: pathlib.Path, dry_run: bool = False) -> Optional[str]:
    yaml = _make_yaml()
    # Resolving inheritance up front guarantees mode-detection sees every
    # legacy flag (rather than defaulting them based on what the override
    # file alone contained). The trade-off: post-migration the file is flat
    # — `base_config:` no longer present. For phase 2 this only affects the
    # one OCC verification config; broader inheritance recovery is a later
    # cleanup.
    data, was_inheriting = _resolve_inheritance(yaml_path, yaml)

    if not isinstance(data, dict):
        return None  # not a config dict (e.g. a list-typed YAML)

    model = data.get("model")
    if not isinstance(model, dict):
        return None

    summary = _migrate_octree_to_field(model)
    if summary is None:
        return None
    if was_inheriting:
        summary = "flattened inheritance + " + summary

    if dry_run:
        return f"[dry-run] {yaml_path}: {summary}"

    backup = yaml_path.with_suffix(yaml_path.suffix + ".bak")
    shutil.copy(yaml_path, backup)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    return f"{yaml_path}: {summary} (backup at {backup.name})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yamls", nargs="+", type=pathlib.Path,
                        help="YAML files to migrate (use shell globs in invocation).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing.")
    args = parser.parse_args()

    n_changed = 0
    n_skipped = 0
    for path in args.yamls:
        if not path.exists():
            print(f"  SKIP (missing): {path}", file=sys.stderr)
            n_skipped += 1
            continue
        result = migrate_file(path, dry_run=args.dry_run)
        if result is None:
            print(f"  SKIP (no legacy keys): {path}")
            n_skipped += 1
        else:
            print(result)
            n_changed += 1

    print(f"\n{n_changed} migrated, {n_skipped} skipped.")


if __name__ == "__main__":
    main()
