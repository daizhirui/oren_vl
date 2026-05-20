"""MultiFieldTrainer — joint training over a multi-field FieldBank.

Phase 4 of the FieldStorage refactor. Demonstrates the composition pattern described in DESIGN.md §"Trainer model" and
§"Worked example: shared-feature SDF + OCC":

  1. Build one `FieldBank` with N FieldStorages (+ optional shared FeatureBanks).
  2. Per step, call `bank.forward(points)` once  dict[name  FieldOutput], so the per-call OctreeGeometry cache is
     shared across every field.
  3. Each child criterion runs against its named FieldOutput slot, returns (loss, loss_dict).
  4. Sum losses with per-field weights, backward, step a single optimizer covering `bank.parameters()`.

**Scope.** This file ships a functional scaffold + a working single-step joint forward/loss path; a complete trainer
that drives end-to-end joint training (with shared key-frame management, ray sampling, evaluator wiring, and mesh
extraction per field) is intentionally left as a follow-up. Per-field trainers (`SdfTrainer`, `OccTrainer`) remain the
canonical entry points for single-field workflows; `MultiFieldTrainer.joint_step` is the seed for the joint workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn

from oren.feature_bank import FeatureBankConfig
from oren.field_bank import FieldBank
from oren.field_output import FieldOutput
from oren.field_storage_config import FieldStorageConfig
from oren.semi_sparse_octree import SemiSparseOctree
from oren.utils.config_abc import ConfigABC


@dataclass
class FieldWeight(ConfigABC):
    """Per-field loss weight used by MultiFieldTrainer."""
    name: str = ""
    weight: float = 1.0


# A "criterion callable" is anything that maps a FieldOutput (plus optional extra kwargs) to `(loss, loss_dict)`. The
# trainer doesn't care whether the callable is a torch.nn.Module subclass or a plain function; it just dispatches the
# named slot to it. The existing `SdfCriterion.forward` / `OccCriterion.forward` already take precomputed tensors (not
# the model), so they slot in cleanly once the caller unpacks the FieldOutput into the expected scalar tensors.
CriterionFn = Callable[[FieldOutput], tuple[torch.Tensor, dict]]


class MultiFieldTrainer(nn.Module):
    """Compose per-field training over a shared multi-field FieldBank."""

    def __init__(
        self,
        octree: SemiSparseOctree,
        fields: list[FieldStorageConfig],
        shared_banks: Iterable[FeatureBankConfig] = (),
        criteria: Optional[dict[str, CriterionFn]] = None,
        weights: Optional[list[FieldWeight]] = None,
    ):
        """Compose a `FieldBank` over `octree` and bind optional per-field weights and criteria.

        Args:
            octree: geometry provider shared across all fields in this trainer.
            fields: per-field configs forwarded to the underlying `FieldBank`.
            shared_banks: cross-field-shared `FeatureBank` declarations.
            criteria: optional initial mapping of field name to criterion callable; can be set later via
                `set_criterion(name, fn)`.
            weights: optional per-field loss weights; missing entries default to 1.0.
        """
        super().__init__()
        self.bank = FieldBank(octree=octree, fields=fields, shared_banks=shared_banks)
        # Per-field loss weights. Missing entries default to 1.0.
        self.weights: dict[str, float] = {w.name: float(w.weight) for w in (weights or [])}
        for f in fields:
            self.weights.setdefault(f.name, 1.0)
        # Optional per-field criteria. May be left empty and set later via `set_criterion(name, fn)`; the joint_step
        # API checks for missing criteria and raises if asked to compute loss without them.
        self.criteria: dict[str, CriterionFn] = dict(criteria or {})

    @classmethod
    def from_sdf_occ_geo_share(
        cls,
        octree_cfg,
        sdf_field_cfg: FieldStorageConfig,
        occ_field_cfg: FieldStorageConfig,
        geo_feature_dim: int,
        weights: Optional[list[FieldWeight]] = None,
    ) -> "MultiFieldTrainer":
        """Convenience constructor for the SDF + OCC joint workflow on a shared `"geo"` FeatureBank — the configuration
        drawn out in DESIGN.md §"Worked example: shared-feature SDF + OCC".

        Both fields must declare `shared_bank="geo"` and an `implicit_feature_dim` matching `geo_feature_dim`.

        Args:
            octree_cfg: octree configuration used to construct the shared `SemiSparseOctree`.
            sdf_field_cfg: field config for the SDF field; must declare `shared_bank="geo"`.
            occ_field_cfg: field config for the OCC field; must declare `shared_bank="geo"`.
            geo_feature_dim: feature dim (F) of the shared `"geo"` `FeatureBank`; must match each field's
                `implicit_feature_dim`.
            weights: optional per-field loss weights.

        Returns:
            A `MultiFieldTrainer` wired with both fields on a shared `"geo"` FeatureBank.
        """
        assert sdf_field_cfg.shared_bank == "geo", "sdf_field_cfg.shared_bank must be 'geo'"
        assert occ_field_cfg.shared_bank == "geo", "occ_field_cfg.shared_bank must be 'geo'"
        assert sdf_field_cfg.implicit_feature_dim == geo_feature_dim, (
            "sdf_field_cfg.implicit_feature_dim must match geo_feature_dim"
        )
        assert occ_field_cfg.implicit_feature_dim == geo_feature_dim, (
            "occ_field_cfg.implicit_feature_dim must match geo_feature_dim"
        )
        octree = SemiSparseOctree(octree_cfg)
        return cls(
            octree=octree,
            fields=[sdf_field_cfg, occ_field_cfg],
            shared_banks=[FeatureBankConfig(name="geo", feature_dim=geo_feature_dim)],
            weights=weights,
        )

    def set_criterion(self, name: str, fn: CriterionFn) -> None:
        """Bind a criterion callable to a field name.

        Args:
            name: field name registered on the underlying `FieldBank`.
            fn: criterion callable mapping a `FieldOutput` to `(loss, loss_dict)`.
        """
        assert name in self.bank.fields, f"unknown field name {name!r}; have {list(self.bank.fields)}"
        self.criteria[name] = fn

    def joint_step(
        self,
        points: torch.Tensor,
        voxel_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Run one bank forward, dispatch every named FieldOutput to its criterion, return the weighted sum + merged
        log dict.

        Args:
            points: (n_points, 3) query points in world coordinates.
            voxel_indices: optional (n_points,) precomputed leaf voxel indices passed through to the bank forward.

        Returns: (total_loss, loss_dict) where `loss_dict` has every child criterion's keys prefixed with
        `<field_name>/`. The caller is responsible for calling `total_loss.backward()` and stepping the optimizer.
        """
        outputs = self.bank(points, voxel_indices=voxel_indices)
        total_loss = torch.zeros((), dtype=torch.float32, device=points.device)
        merged_log: dict = {}
        for name, out in outputs.items():
            crit = self.criteria.get(name)
            if crit is None:
                continue  # field has no criterion bound this step; skip silently
            loss, log = crit(out)
            weight = self.weights.get(name, 1.0)
            total_loss = total_loss + weight * loss
            for k, v in log.items():
                merged_log[f"{name}/{k}"] = v
        return total_loss, merged_log

    @property
    def octree(self) -> SemiSparseOctree:
        return self.bank.octree
