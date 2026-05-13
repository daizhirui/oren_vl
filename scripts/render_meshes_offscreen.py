"""Offscreen-render a list of meshes from a fixed camera view (Open3D ViewTrajectory JSON).
Saves one PNG per mesh into output_dir/<label>.png."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def render(mesh_path: str, view: dict, width: int, height: int, out_path: str,
           bg_color=(0.05, 0.05, 0.05), backface_culling: bool = True,
           crop_min=None, crop_max=None) -> None:
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if crop_min is not None and crop_max is not None:
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_min, max_bound=crop_max)
        mesh = mesh.crop(aabb)
    mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.background_color = np.asarray(bg_color)
    opt.mesh_show_back_face = not backface_culling
    opt.light_on = True

    vc = vis.get_view_control()
    vc.set_lookat(view["lookat"])
    vc.set_front(view["front"])
    vc.set_up(view["up"])
    vc.set_zoom(view["zoom"])
    if "field_of_view" in view:
        try:
            cur_fov = vc.get_field_of_view()
            delta = view["field_of_view"] - cur_fov
            vc.change_field_of_view(step=delta / 5.0)
        except Exception:
            pass

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--view-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--width", type=int, default=900)
    p.add_argument("--height", type=int, default=900)
    # entries: "label::mesh_path" repeated
    p.add_argument("--mesh", action="append", required=True,
                   help="label::path/to/mesh.ply (repeatable)")
    p.add_argument("--crop-min", nargs=3, type=float, default=None,
                   help="Crop each mesh to this AABB min (x y z) before rendering")
    p.add_argument("--crop-max", nargs=3, type=float, default=None,
                   help="Crop each mesh to this AABB max (x y z) before rendering")
    args = p.parse_args()

    with open(args.view_json) as f:
        traj = json.load(f)
    view = traj["trajectory"][0]

    os.makedirs(args.out_dir, exist_ok=True)
    for entry in args.mesh:
        label, path = entry.split("::", 1)
        out = os.path.join(args.out_dir, f"{label}.png")
        if not os.path.exists(path):
            print(f"[skip] {label}: {path} missing")
            continue
        print(f"[render] {label} <- {path}")
        render(path, view, args.width, args.height, out,
               crop_min=args.crop_min, crop_max=args.crop_max)
    print(f"\nDone. PNGs in {args.out_dir}")


if __name__ == "__main__":
    main()
