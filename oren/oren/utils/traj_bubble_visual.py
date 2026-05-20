from typing import List, Optional, Union

import numpy as np
import vedo


class TrajBubbleVisualizer:
    """Visualize trajectory and bubbles inside a mesh using vedo."""

    def __init__(
        self,
        mesh_color: Union[str, List[float]] = "grey",
        mesh_opacity: float = 0.3,
        traj_color: Union[str, List[float]] = "red",
        traj_point_size: int = 8,
        traj_line_width: int = 3,
        bubble_color: Union[str, List[float]] = "green",
        bubble_transparency: float = 0.5,
    ):
        """Configure rendering styles for the mesh / trajectory / bubble visualization.

        Args:
            mesh_color: vedo-compatible color spec (name or RGB list) for the background mesh.
            mesh_opacity: Alpha in [0, 1] applied to the mesh.
            traj_color: Color used for trajectory points and connecting line.
            traj_point_size: Radius in pixels for trajectory point glyphs.
            traj_line_width: Width in pixels for the trajectory polyline.
            bubble_color: Color used for the bubble spheres.
            bubble_transparency: Bubble transparency in [0, 1]; alpha is `1 - bubble_transparency`.
        """
        self.mesh_color = mesh_color
        self.mesh_opacity = mesh_opacity
        self.traj_color = traj_color
        self.traj_point_size = traj_point_size
        self.traj_line_width = traj_line_width
        self.bubble_color = bubble_color
        self.bubble_transparency = bubble_transparency

        self.mesh: Optional[vedo.Mesh] = None
        self.traj: Optional[np.ndarray] = None
        self.bubble: Optional[np.ndarray] = None

    def load_mesh(self, mesh_path: str) -> vedo.Mesh:
        """Load mesh from file.

        Args:
            mesh_path: Path to a mesh file readable by `vedo.Mesh` (e.g., .ply, .obj).

        Returns:
            The loaded `vedo.Mesh`; also stored on `self.mesh`.
        """
        self.mesh = vedo.Mesh(mesh_path)
        return self.mesh

    def load_traj(self, traj_path: str) -> np.ndarray:
        """Load trajectory (n, 3) from text file.

        Args:
            traj_path: Path to a whitespace-separated text file with one xyz row per waypoint.

        Returns:
            (n, 3) trajectory array; also stored on `self.traj`.
        """
        traj = np.loadtxt(traj_path)
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        self.traj = traj
        return self.traj

    def load_bubble(self, bubble_path: str) -> np.ndarray:
        """Load bubbles (n, 4) from text file. Each row: x, y, z, radius.

        Args:
            bubble_path: Path to a whitespace-separated text file with rows `x y z radius`.

        Returns:
            (n, 4) bubble array; also stored on `self.bubble`.
        """
        bubble = np.loadtxt(bubble_path)
        if bubble.ndim == 1:
            bubble = bubble.reshape(1, -1)
        self.bubble = bubble
        return self.bubble

    def set_data(
        self,
        mesh: Optional[vedo.Mesh] = None,
        traj: Optional[np.ndarray] = None,
        bubble: Optional[np.ndarray] = None,
    ):
        """Set data directly instead of loading from files.

        Each argument is only assigned when it is not None, so callers may update individual fields independently.

        Args:
            mesh: Optional pre-loaded `vedo.Mesh` to assign to `self.mesh`.
            traj: Optional (n, 3) trajectory array to assign to `self.traj`.
            bubble: Optional (n, 4) bubble array (`x y z radius` rows) to assign to `self.bubble`.
        """
        if mesh is not None:
            self.mesh = mesh
        if traj is not None:
            self.traj = traj
        if bubble is not None:
            self.bubble = bubble

    def visualize(self):
        """Visualize the loaded mesh, trajectory, and bubbles."""
        assert self.mesh is not None, "Mesh not loaded. Call load_mesh() or set_data() first."
        assert self.traj is not None, "Trajectory not loaded. Call load_traj() or set_data() first."
        assert self.bubble is not None, "Bubbles not loaded. Call load_bubble() or set_data() first."

        actors = []

        # --- Mesh ---
        self.mesh.color(self.mesh_color).alpha(self.mesh_opacity)
        actors.append(self.mesh)

        # --- Bubbles (render first so trajectory appears on top) ---
        if self.bubble.shape[0] > 0:
            for i in range(self.bubble.shape[0]):
                center = self.bubble[i, :3]
                radius = self.bubble[i, 3]
                sphere = vedo.Sphere(pos=center, r=radius, res=16)
                sphere.color(self.bubble_color).alpha(1.0 - self.bubble_transparency)
                actors.append(sphere)

        # --- Trajectory ---
        if self.traj.shape[0] > 0:
            # Draw trajectory points
            traj_points = vedo.Points(self.traj, c=self.traj_color, r=self.traj_point_size)
            actors.append(traj_points)

            # Draw trajectory line if more than 1 point
            if self.traj.shape[0] > 1:
                traj_line = vedo.Line(self.traj, c=self.traj_color, lw=self.traj_line_width)
                actors.append(traj_line)

            # Mark start and end points
            start_pt = vedo.Point(self.traj[0], c=self.traj_color, r=self.traj_point_size * 2)
            end_pt = vedo.Point(self.traj[-1], c=self.traj_color, r=self.traj_point_size * 2)
            actors.append(start_pt)
            actors.append(end_pt)

        # --- Show ---
        plotter = vedo.Plotter()
        plotter.show(*actors, axes=1, viewup="z", title="Trajectory & Bubbles")
        plotter.close()


if __name__ == "__main__":
    mesh_path = "logs/oren_ros/2026-02-20-10-39-58/mesh/mesh.ply"
    traj_path = "logs/oren_ros/2026-02-20-10-39-58/misc/traj.txt"
    bubble_path = "logs/oren_ros/2026-02-20-10-39-58/misc/bubbles.txt"

    viz = TrajBubbleVisualizer(bubble_color=[0, 1, 0], bubble_transparency=0.5)
    viz.load_mesh(mesh_path)
    viz.load_traj(traj_path)
    viz.load_bubble(bubble_path)
    viz.visualize()
