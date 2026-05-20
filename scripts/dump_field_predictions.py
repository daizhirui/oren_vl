"""Dump per-point field predictions from a trained checkpoint.

Used to capture a baseline before the FieldStorage refactor: the predicted
(prior, residual, pred) tuple at a fixed grid of points, plus the
ground-truth SDF values from the test set. After the refactor, re-run this
script with the new model on the same checkpoint location (or its
re-trained equivalent) and compare tensors element-wise.

The script is model-agnostic across SdfNetwork / OccNetwork because both
return the same 4-tuple (voxel_indices, prior, residual, pred) today.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
import torch

from oren.trainer_config import TrainerConfig
from oren.utils.registry import get_model


def sha256_of_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(cfg: TrainerConfig, ckpt_path: pathlib.Path) -> torch.nn.Module:
    model_cls = get_model(cfg.model_identifier)
    model = model_cls(cfg.model)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys when loading ckpt: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys when loading ckpt: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    model.to(cfg.device)
    model.eval()
    return model


@torch.no_grad()
def query_in_batches(
    model: torch.nn.Module,
    points: torch.Tensor,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    voxel_indices_chunks = []
    prior_chunks = []
    residual_chunks = []
    pred_chunks = []

    n = points.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = points[start:end]
        vi, prior, residual, pred = model(chunk)
        voxel_indices_chunks.append(vi.cpu())
        prior_chunks.append(prior.cpu())
        residual_chunks.append(residual.cpu())
        pred_chunks.append(pred.cpu())

    return {
        "voxel_indices": torch.cat(voxel_indices_chunks, dim=0),
        "prior": torch.cat(prior_chunks, dim=0),
        "residual": torch.cat(residual_chunks, dim=0),
        "pred": torch.cat(pred_chunks, dim=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=pathlib.Path,
                        help="Trainer YAML that produced the checkpoint.")
    parser.add_argument("--ckpt", required=True, type=pathlib.Path,
                        help="Path to final.pth (model.state_dict()).")
    parser.add_argument("--test-set", required=True, type=pathlib.Path,
                        help="Directory with grid_points.npy and gt_*.npy.")
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="Output .pt path for the dumped predictions.")
    parser.add_argument("--batch-size", type=int, default=204800,
                        help="Points per forward pass (defaults to trainer batch_size if 0).")
    parser.add_argument("--device", type=str, default=None,
                        help="Override cfg.device (e.g. cuda:0).")
    args = parser.parse_args()

    args.config = args.config.resolve()
    args.ckpt = args.ckpt.resolve()
    args.test_set = args.test_set.resolve()
    args.output = args.output.resolve()

    cfg: TrainerConfig = TrainerConfig.from_yaml(args.config)
    if args.device is not None:
        cfg.device = args.device

    batch_size = args.batch_size or cfg.batch_size
    print(f"config         : {args.config}")
    print(f"ckpt           : {args.ckpt}")
    print(f"test_set       : {args.test_set}")
    print(f"output         : {args.output}")
    print(f"device         : {cfg.device}")
    print(f"batch_size     : {batch_size}")
    print(f"model_identifier: {cfg.model_identifier}")

    print("Building model and loading checkpoint...")
    model = build_model(cfg, args.ckpt)

    grid_points_np = np.load(args.test_set / "grid_points.npy")
    gt_sdf_np = np.load(args.test_set / "gt_sdf_values.npy")
    near_mask_np = np.load(args.test_set / "mask_near_surface.npy")
    far_mask_np = np.load(args.test_set / "mask_far_surface.npy")
    print(f"grid_points    : shape={grid_points_np.shape} dtype={grid_points_np.dtype}")

    points = torch.from_numpy(grid_points_np).to(cfg.device).float().view(-1, 3)

    t0 = time.time()
    out = query_in_batches(model, points, batch_size)
    dt = time.time() - t0
    print(f"queried {points.shape[0]} points in {dt:.2f} s "
          f"({points.shape[0] / max(dt, 1e-6):.0f} pts/s)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # Inputs (so a later re-run can verify identical query coordinates)
        "points": points.cpu(),
        "gt_sdf_values": torch.from_numpy(gt_sdf_np),
        "mask_near_surface": torch.from_numpy(near_mask_np),
        "mask_far_surface": torch.from_numpy(far_mask_np),
        # Predictions
        "voxel_indices": out["voxel_indices"],
        "prior": out["prior"],
        "residual": out["residual"],
        "pred": out["pred"],
        # Provenance
        "meta": {
            "config_path": str(args.config),
            "config_sha256": sha256_of_file(args.config),
            "ckpt_path": str(args.ckpt),
            "ckpt_sha256": sha256_of_file(args.ckpt),
            "test_set_dir": str(args.test_set),
            "model_identifier": cfg.model_identifier,
            "batch_size": batch_size,
            "device": cfg.device,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "n_points": int(points.shape[0]),
            "elapsed_seconds": dt,
        },
    }
    torch.save(payload, args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KiB)")

    summary = {
        "prior":    {"min": float(out["prior"].min()),    "max": float(out["prior"].max()),    "mean": float(out["prior"].mean())},
        "residual": {"min": float(out["residual"].min()), "max": float(out["residual"].max()), "mean": float(out["residual"].mean())},
        "pred":     {"min": float(out["pred"].min()),     "max": float(out["pred"].max()),     "mean": float(out["pred"].mean())},
    }
    print("summary statistics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
