"""Phase-1 parity check: FieldStorage(explicit, D=1, GA=True) vs SdfNetwork.prior.

Loads a Phase 0 SDF checkpoint, copies its per-vertex `sdf_priors` and
`grad_priors` into a `FieldStorage(name="sdf", mode="explicit",
gradient_augmentation=True)`, queries at the test-set grid points, and diffs
against the baseline's `prior` tensor.

The baseline's `prior` is exactly the legacy `ga_trilinear` path inside
the pre-refactor `SemiSparseOctree.forward()` (since deleted along with
the `Base` subclass). FieldStorage's `_explicit` branch is the
extracted, decoupled version of that path. They should be bit-equal — if
they are, the new abstraction is numerically faithful for explicit/GA
fields, which is the only mode FieldStorage exposes in phase 1.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import torch

from oren.field_storage import FieldStorage
from oren.field_storage_config import FieldStorageConfig
from oren.semi_sparse_octree import SemiSparseOctree
from oren.trainer_config import TrainerConfig
from oren.utils.registry import get_model


def build_model_and_extract_octree(cfg: TrainerConfig, ckpt_path: pathlib.Path):
    """Reconstruct the SdfNetwork and load its checkpoint. Returns the model
    plus a back-reference to its octree, so we can borrow its vertex values
    for the FieldStorage probe."""
    model_cls = get_model(cfg.model_identifier)
    model = model_cls(cfg.model)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(cfg.device)
    model.eval()
    return model, model.octree


@torch.no_grad()
def extract_sdf_field_from_model(model) -> FieldStorage:
    """Return the trained SDF FieldStorage held inside the model wrapper.

    Phase 4 made `SdfNetwork` a thin adapter around a single-field FieldBank,
    so the trained `FieldStorage` is already a submodule at
    `model.field_bank.fields["sdf"]`. We don't need to reconstruct it from
    legacy octree-Parameter state anymore (that path was deleted in phase 4
    cleanup); we just hand back the existing instance.
    """
    return model.field_bank.fields["sdf"]


@torch.no_grad()
def query_in_batches(
    octree: SemiSparseOctree,
    fs: FieldStorage,
    points: torch.Tensor,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Query the field with `prior_only=True` so the explicit (trilinear)
    branch is the only thing that runs. Even when the field is configured
    in hybrid mode, this isolates the explicit output for parity comparison
    against the legacy SdfNetwork.prior tensor."""
    voxel_indices_chunks = []
    pred_chunks = []
    n = points.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = points[start:end]
        geom = octree.query(chunk)
        out = fs.forward(geom, prior_only=True)
        voxel_indices_chunks.append(out.voxel_indices.cpu())
        # FieldOutput.pred is (N, D=1); squeeze to match the legacy (N,) prior layout.
        pred_chunks.append(out.pred.squeeze(-1).cpu())
    return {
        "voxel_indices": torch.cat(voxel_indices_chunks, dim=0),
        "pred": torch.cat(pred_chunks, dim=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=pathlib.Path,
                        help="Trainer YAML that produced the SDF checkpoint.")
    parser.add_argument("--ckpt", required=True, type=pathlib.Path,
                        help="Path to final.pth (SDF model.state_dict()).")
    parser.add_argument("--test-set", required=True, type=pathlib.Path,
                        help="Directory with grid_points.npy.")
    parser.add_argument("--baseline-predictions", required=True, type=pathlib.Path,
                        help="Path to baseline predictions.pt produced by dump_field_predictions.py "
                             "(used as the source of truth for the bit-equal comparison).")
    parser.add_argument("--batch-size", type=int, default=204800)
    args = parser.parse_args()

    cfg: TrainerConfig = TrainerConfig.from_yaml(args.config)
    batch_size = args.batch_size or cfg.batch_size
    print(f"config         : {args.config}")
    print(f"ckpt           : {args.ckpt}")
    print(f"test_set       : {args.test_set}")
    print(f"baseline       : {args.baseline_predictions}")
    print(f"device         : {cfg.device}")
    print(f"batch_size     : {batch_size}")

    print("Loading SdfNetwork checkpoint...")
    model, octree = build_model_and_extract_octree(cfg, args.ckpt)
    print("Extracting FieldStorage(explicit, D=1, GA=True) from the SdfNetwork wrapper...")
    fs = extract_sdf_field_from_model(model)

    grid_points_np = np.load(args.test_set / "grid_points.npy")
    points = torch.from_numpy(grid_points_np).to(cfg.device).float().view(-1, 3)
    print(f"grid_points    : {tuple(grid_points_np.shape)}  flattened to {tuple(points.shape)}")

    t0 = time.time()
    fs_out = query_in_batches(octree, fs, points, batch_size)
    dt = time.time() - t0
    print(f"FieldStorage probe: {points.shape[0]} points in {dt:.2f}s "
          f"({points.shape[0] / max(dt, 1e-6):.0f} pts/s)")

    baseline = torch.load(args.baseline_predictions, map_location="cpu", weights_only=False)

    # The baseline's `prior` is the legacy SdfNetwork.prior — exactly what
    # FieldStorage._explicit should reproduce.
    for k_fs, k_base in [("voxel_indices", "voxel_indices"), ("pred", "prior")]:
        a = fs_out[k_fs]
        b = baseline[k_base]
        if torch.equal(a, b):
            print(f"  {k_fs:14s} vs baseline.{k_base:14s} bit-equal  shape={tuple(a.shape)} dtype={a.dtype}")
        else:
            if a.is_floating_point():
                d = (a.float() - b.float()).abs()
                print(f"  {k_fs:14s} vs baseline.{k_base:14s} DIFF  max={d.max().item():.3e} "
                      f"mean={d.mean().item():.3e} count_nonzero={(d > 0).sum().item()}/{a.numel()}")
            else:
                print(f"  {k_fs:14s} vs baseline.{k_base:14s} DIFF  n_mismatch={(a != b).sum().item()}/{a.numel()}")


if __name__ == "__main__":
    main()
