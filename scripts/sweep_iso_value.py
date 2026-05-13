"""Sweep mesh_iso_value on a trained OccNetwork.

Loads the model from `final.pth` of a chosen run, runs marching cubes at multiple
iso_values, computes Chamfer distance / precision / recall against the GT mesh,
and writes one mesh.ply per iso into out_dir.

Usage:
    PYTHONSAFEPATH=1 python scripts/sweep_iso_value.py
"""
import argparse
import os
import yaml
import numpy as np
import open3d as o3d

# Rely on the editable install (PYTHONSAFEPATH=1 avoids the cwd shadowing the oren package)
from oren.evaluator_occ import OccEvaluator
from oren.occ_network import OccNetworkConfig

DEFAULT_ISO = [-3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
THRESHOLDS = [0.01, 0.025, 0.05]  # 1cm, 2.5cm, 5cm
GT_MESH = "/home/daizhirui/DataArchive/Replica-SDF-aug3/room0_mesh.ply"

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True, help="Run directory containing bak/config.yaml and ckpt/final.pth")
ap.add_argument("--out-dir", default=None, help="Where to save meshes (default: <run-dir>/iso_sweep)")
ap.add_argument("--iso", nargs="+", type=float, default=DEFAULT_ISO)
ap.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
ap.add_argument("--gt-mesh", default=GT_MESH)
ap.add_argument("--quiet", action="store_true", help="Print only summary rows")
ap.add_argument("--field", default="occ", choices=["occ", "occ_prior"],
                help="Which field to extract mesh from (occ=prior+residual, occ_prior=octree only)")
args = ap.parse_args()

RUN_DIR = args.run_dir
OUT_DIR = args.out_dir or os.path.join(RUN_DIR, "iso_sweep")
ISO_VALUES = args.iso
GT_MESH = args.gt_mesh

os.makedirs(OUT_DIR, exist_ok=True)

# Load the saved config to get bounds + model config
with open(os.path.join(RUN_DIR, "bak", "config.yaml")) as f:
    cfg_dict = yaml.safe_load(f)

bound_min = cfg_dict["bound_min"]
bound_max = cfg_dict["bound_max"]
mesh_resolution = cfg_dict["mesh_resolution"]
batch_size = cfg_dict["batch_size"]
clean_mesh = cfg_dict["clean_mesh"]

# Reconstruct the model config via the ConfigABC loader so nested fields are filled
model_cfg = OccNetworkConfig.from_dict(dict(cfg_dict["model"]))

print(f"Loading model from {RUN_DIR}/ckpt/final.pth ...")
evaluator = OccEvaluator(
    batch_size=batch_size,
    clean_mesh=clean_mesh,
    model_cfg=model_cfg,
    model_path=os.path.join(RUN_DIR, "ckpt", "final.pth"),
    device="cuda",
)
print(f"Model loaded. bound_min={bound_min}, bound_max={bound_max}, res={mesh_resolution}")

# Load GT once
gt_mesh = o3d.io.read_triangle_mesh(GT_MESH)
gt_pts = np.asarray(gt_mesh.sample_points_uniformly(300_000).points)
gt_t = o3d.geometry.PointCloud(); gt_t.points = o3d.utility.Vector3dVector(gt_pts)


def f1(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

hdr = f"{'iso':>5s}  {'CD':>6s}"
for t in args.thresholds:
    cm = int(t * 100 * 10) / 10
    hdr += f"  {f'P@{cm}':>6s}  {f'R@{cm}':>6s}  {f'F1@{cm}':>6s}"
print(hdr)

results = []
for iso in ISO_VALUES:
    [mesh] = evaluator.extract_mesh(
        bound_min=bound_min, bound_max=bound_max,
        grid_resolution=mesh_resolution, fields=[args.field], iso_value=iso,
    )
    out_path = os.path.join(OUT_DIR, f"mesh_{args.field}_iso{iso:+.2f}.ply")
    o3d.io.write_triangle_mesh(out_path, mesh)
    if len(mesh.vertices) == 0:
        print(f"{iso:>5.2f}  empty mesh")
        continue
    pred_pts = np.asarray(mesh.sample_points_uniformly(300_000).points)
    pr_t = o3d.geometry.PointCloud(); pr_t.points = o3d.utility.Vector3dVector(pred_pts)
    d1 = np.asarray(pr_t.compute_point_cloud_distance(gt_t))  # pred -> GT (precision)
    d2 = np.asarray(gt_t.compute_point_cloud_distance(pr_t))  # GT -> pred (recall)
    cd = 0.5 * (d1.mean() + d2.mean())

    row = f"{iso:>5.2f}  {cd*100:6.3f}"
    metrics = {"iso": iso, "cd": cd}
    for t in args.thresholds:
        p = (d1 < t).mean(); r = (d2 < t).mean(); f = f1(p, r)
        row += f"  {p*100:5.1f}%  {r*100:5.1f}%  {f*100:5.1f}%"
        metrics[f"p{t}"] = p; metrics[f"r{t}"] = r; metrics[f"f{t}"] = f
    print(row)
    results.append(metrics)

def _fmt_cm(t: float) -> str:
    cm = t * 100
    return f"{cm:g}"


# Rankings
print("\nBest 3 by CD:")
for r in sorted(results, key=lambda r: r["cd"])[:3]:
    parts = [f"  iso={r['iso']:+.2f}  CD={r['cd']*100:.3f}cm"]
    for t in args.thresholds:
        parts.append(f"F1@{_fmt_cm(t)}={r[f'f{t}']*100:.1f}%")
    print("  ".join(parts))

for t in args.thresholds:
    cm = _fmt_cm(t)
    print(f"\nBest 3 by F1@{cm}cm:")
    for r in sorted(results, key=lambda r: -r[f"f{t}"])[:3]:
        print(f"  iso={r['iso']:+.2f}  F1@{cm}={r[f'f{t}']*100:.2f}%  "
              f"P@{cm}={r[f'p{t}']*100:.1f}%  R@{cm}={r[f'r{t}']*100:.1f}%  CD={r['cd']*100:.3f}cm")
