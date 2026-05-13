"""Compare contribution of occ_prior vs occ_residual on a trained OccNetwork.

Samples points in the supervised octree region, evaluates the network, and plots
histograms of (prior, residual, prior+residual) plus a scatter of prior vs residual.

Usage:
    PYTHONSAFEPATH=1 python scripts/hist_prior_residual.py --run-dir <run> [--field-source grid|points] [--out out.png]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from oren.evaluator_occ import OccEvaluator
from oren.occ_network import OccNetworkConfig

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--n-points", type=int, default=200_000)
ap.add_argument("--out", default=None, help="Default: <run-dir>/prior_vs_residual.png")
args = ap.parse_args()

with open(os.path.join(args.run_dir, "bak", "config.yaml")) as f:
    cfg = yaml.safe_load(f)
bound_min = np.array(cfg["bound_min"])
bound_max = np.array(cfg["bound_max"])

model_cfg = OccNetworkConfig.from_dict(dict(cfg["model"]))
evaluator = OccEvaluator(
    batch_size=cfg["batch_size"], clean_mesh=True,
    model_cfg=model_cfg,
    model_path=os.path.join(args.run_dir, "ckpt", "final.pth"),
    device="cuda",
)

# Sample points uniformly in the bbox, then filter to ones inside an active leaf voxel
# (where the model has actually been supervised). Points outside any leaf are dead.
n = args.n_points
pts = torch.rand(n, 3, device="cuda", dtype=torch.float32) * \
      torch.tensor(bound_max - bound_min, device="cuda", dtype=torch.float32) + \
      torch.tensor(bound_min, device="cuda", dtype=torch.float32)
out = evaluator.forward_model(evaluator.model, pts, get_grad=False, auto_grad=False, prior_only=False, device="cuda")
voxel_indices = out["voxel_indices"]
mask = voxel_indices >= 0
prior = out["occ_prior"][mask].cpu().numpy()
residual = out["occ_residual"][mask].cpu().numpy() if out.get("occ_residual") is not None else np.zeros_like(prior)
combined = prior + residual
n_active = mask.sum().item()
print(f"sampled {n} points; {n_active} ({100*n_active/n:.1f}%) inside active voxels")
print(f"  prior:     mean={prior.mean():+.3f}  std={prior.std():.3f}  |.|.mean={np.abs(prior).mean():.3f}")
print(f"  residual:  mean={residual.mean():+.3f}  std={residual.std():.3f}  |.|.mean={np.abs(residual).mean():.3f}")
print(f"  combined:  mean={combined.mean():+.3f}  std={combined.std():.3f}  |.|.mean={np.abs(combined).mean():.3f}")

# Fraction of |combined| that comes from prior (vs residual)
prior_frac = np.abs(prior) / (np.abs(prior) + np.abs(residual) + 1e-9)
print(f"  share-of-magnitude — prior: median {np.median(prior_frac)*100:.1f}%   mean {prior_frac.mean()*100:.1f}%")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
bins = 80

ax = axes[0, 0]
xmax = max(np.abs(prior).max(), np.abs(residual).max(), np.abs(combined).max())
xrange = (-xmax, xmax)
ax.hist(prior, bins=bins, range=xrange, alpha=0.55, color="C0", label=f"prior  (|.|={np.abs(prior).mean():.2f})")
ax.hist(residual, bins=bins, range=xrange, alpha=0.55, color="C1", label=f"residual  (|.|={np.abs(residual).mean():.2f})")
ax.set_xlabel("logit value")
ax.set_ylabel("count")
ax.set_yscale("log")
ax.legend()
ax.set_title("Distribution of prior vs residual logits")
ax.axvline(0, color="k", lw=0.5, ls="--")

ax = axes[0, 1]
ax.hist(combined, bins=bins, range=xrange, alpha=0.7, color="C2", label=f"prior+residual  (|.|={np.abs(combined).mean():.2f})")
ax.hist(prior, bins=bins, range=xrange, alpha=0.4, color="C0", label="prior")
ax.set_xlabel("logit value")
ax.set_ylabel("count")
ax.set_yscale("log")
ax.legend()
ax.set_title("How the residual reshapes the prior")
ax.axvline(0, color="k", lw=0.5, ls="--")

ax = axes[1, 0]
ax.hist(prior_frac, bins=50, range=(0, 1), color="C3")
ax.set_xlabel("share of |combined| coming from prior  (1 = prior dominates, 0 = residual dominates)")
ax.set_ylabel("count")
ax.set_title(f"Per-point magnitude share — median {np.median(prior_frac)*100:.1f}%, mean {prior_frac.mean()*100:.1f}%")
ax.axvline(0.5, color="k", lw=0.5, ls="--")

ax = axes[1, 1]
# Scatter prior vs residual (subsample for clarity)
sub_n = min(20_000, len(prior))
idx = np.random.choice(len(prior), sub_n, replace=False)
ax.scatter(prior[idx], residual[idx], s=0.5, alpha=0.3, color="C4")
lim = max(np.abs(prior).max(), np.abs(residual).max())
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.plot([-lim, lim], [lim, -lim], "r--", lw=0.5, alpha=0.4, label="prior + residual = 0")
ax.set_xlabel("prior logit"); ax.set_ylabel("residual logit")
ax.set_title(f"Scatter: prior vs residual ({sub_n} points)")
ax.legend(loc="upper right")
ax.set_aspect("equal")

plt.suptitle(f"OccNetwork: prior vs residual contribution\n{args.run_dir}", fontsize=10)
plt.tight_layout()
out_path = args.out or os.path.join(args.run_dir, "prior_vs_residual.png")
plt.savefig(out_path, dpi=110)
print(f"saved: {out_path}")
