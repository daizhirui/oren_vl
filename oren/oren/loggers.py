import os
import pickle
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from colorama import Fore, Style


class BasicLogger:
    def __init__(self, log_dir: str, exp_name: str, config_dict: dict, for_eva: bool = False) -> None:
        """Create a logger that writes to a timestamped subdirectory under `log_dir/exp_name`.

        Args:
            log_dir: parent directory under which the experiment folder is created.
            exp_name: experiment name; used as a subdirectory under `log_dir`.
            config_dict: configuration to dump as YAML into the backup folder (unless `for_eva`).
            for_eva: if True, only set up paths without creating directories or a TensorBoard writer; used for
                read-only evaluation runs.
        """
        self.log_dir = os.path.join(
            log_dir,
            exp_name,
            self.get_random_time_str(),
        )
        self.img_dir = os.path.join(self.log_dir, "imgs")
        self.mesh_dir = os.path.join(self.log_dir, "mesh")
        self.ckpt_dir = os.path.join(self.log_dir, "ckpt")
        self.backup_dir = os.path.join(self.log_dir, "bak")
        self.misc_dir = os.path.join(self.log_dir, "misc")
        self.for_eva = for_eva
        self.tb = None
        if not for_eva:
            os.makedirs(self.img_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.mesh_dir, exist_ok=True)
            os.makedirs(self.misc_dir, exist_ok=True)
            os.makedirs(self.backup_dir, exist_ok=True)

            self.log_config(config_dict)
            self.tb = SummaryWriter(self.log_dir)

    @staticmethod
    def get_random_time_str():
        return datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")

    def log_ckpt(self, state_dict: dict, name: str):
        """Save a checkpoint state_dict to disk.

        Args:
            state_dict: dictionary of tensors / scalars to torch.save.
            name: target file name; treated as absolute if it starts with `/`, otherwise placed under `ckpt_dir`.
        """
        path = name if os.path.isabs(name) else os.path.join(self.ckpt_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def log_config(self, config_dict: dict):
        """Dump the configuration dictionary to `<backup_dir>/config.yaml`.

        Args:
            config_dict: configuration to dump.
        """
        out_path = os.path.join(self.backup_dir, "config.yaml")
        yaml.dump(config_dict, open(out_path, "w"))

    def log_mesh(self, mesh, name="final_mesh.ply"):
        """Write a triangle mesh to disk.

        Args:
            mesh: an Open3D triangle mesh.
            name: file name; treated as absolute if it starts with `/`, otherwise placed under `mesh_dir`.
        """
        out_path = name if os.path.isabs(name) else os.path.join(self.mesh_dir, name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        o3d.io.write_triangle_mesh(out_path, mesh)

    def log_point_cloud(self, pcd, name="final_points.ply"):
        """Write a point cloud to disk.

        Args:
            pcd: an Open3D point cloud.
            name: file name; treated as absolute if it starts with `/`, otherwise placed under `mesh_dir`.
        """
        out_path = name if os.path.isabs(name) else os.path.join(self.mesh_dir, name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        o3d.io.write_point_cloud(out_path, pcd)

    def log_numpy_data(self, data, name, ind=None):
        """Save a numpy array (or torch tensor) under `misc_dir`.

        Args:
            data: array-like; torch tensors are detached, moved to CPU, and converted.
            name: file stem (no extension).
            ind: optional integer index; if provided, the file is named `name-<ind:05d>.npy`.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if ind is not None:
            np.save(os.path.join(self.misc_dir, "{}-{:05d}.npy".format(name, ind)), data)
        else:
            np.save(os.path.join(self.misc_dir, f"{name}.npy"), data)

    def log_debug_data(self, data, idx):
        """Pickle arbitrary debug data under `misc_dir/scene_data_<idx>.pkl`.

        Args:
            data: object to pickle.
            idx: index appended to the file name.
        """
        with open(os.path.join(self.misc_dir, f"scene_data_{idx}.pkl"), "wb") as f:
            pickle.dump(data, f)

    def log_rgb(self, rgb, name: str):
        """Save an RGB image (float in [0, 1] or torch tensor) under `img_dir`.

        Args:
            rgb: (H, W, 3) image in float [0, 1]; torch tensors are detached and moved to CPU.
            name: file name; treated as absolute if it starts with `/`, otherwise placed under `img_dir`.
        """
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        rgb = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img_path = name if os.path.isabs(name) else os.path.join(self.img_dir, name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        cv2.imwrite(img_path, rgb)

    def log_depth(self, depth, name: str):
        """Save a depth image (meters) as 16-bit PNG in millimeters under `img_dir`.

        Args:
            depth: (H, W) depth in meters; torch tensors are detached and moved to CPU.
            name: file name; treated as absolute if it starts with `/`, otherwise placed under `img_dir`.
        """
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        depth = (depth * 1000).astype(np.uint16)  # Convert to mm and uint16
        img_path = name if os.path.isabs(name) else os.path.join(self.img_dir, name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        cv2.imwrite(img_path, depth)

    @staticmethod
    def info(msg: str):
        tqdm.write(msg)

    @staticmethod
    def warn(msg: str):
        tqdm.write(Fore.YELLOW + msg + Style.RESET_ALL)
