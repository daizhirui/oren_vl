"""Streaming `DataLoader` that pulls (depth or LiDAR) frames + poses from ROS 2 topics.

Constructed by `RosTrainer` with a shared `rclpy.node.Node`; subscriptions and the
no-data watchdog timer are registered on that node. The trainer interacts with
this loader through the same duck-typed interface as `replica.DataLoader` /
`newer_college.DataLoader`:

  - `bound_min`, `bound_max` (torch.Tensor) — for `Trainer.__init__` bound auto-set.
  - `__len__()` returns -1 to signal an unbounded streaming source.
  - `__getitem__(idx)` blocks until a synchronized frame is available, or returns
    `None` either when shutdown has been requested OR when `idle_block_sec`
    elapsed without a new frame (so the trainer can keep optimizing existing
    keyframes during sensor pauses). The trainer distinguishes the two cases
    via the `is_shutdown` property.

`on_idle` is an optional callback invoked while `__getitem__` is blocked; the
trainer wires it up to drain queued service requests so they don't starve when
no frames are arriving.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import torch
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, PointCloud2

from oren.frame import DepthFrame, Frame, LiDARFrame


class DataLoader:
    """Streaming data source for `Trainer`. See module docstring."""

    streaming = True

    def __init__(
        self,
        ros_node: Node,
        bound_min: list[float],
        bound_max: list[float],
        intrinsics: Optional[list[list[float]]] = None,
        min_depth: float = 0.0,
        max_depth: float = -1.0,
    ):
        self.node = ros_node
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.intrinsics = (
            torch.tensor(intrinsics, dtype=torch.float32) if intrinsics is not None else None
        )
        self.bound_min = torch.tensor(bound_min, dtype=torch.float32)
        self.bound_max = torch.tensor(bound_max, dtype=torch.float32)

        # ROS interface params (declared here, where they're used)
        ros_node.declare_parameter("modality", "depth")          # "depth" | "lidar"
        ros_node.declare_parameter("depth_topic", "/camera/depth")
        ros_node.declare_parameter("pose_topic", "/pose")
        ros_node.declare_parameter("lidar_topic", "/lidar/points")
        ros_node.declare_parameter("odom_topic", "/odom")
        ros_node.declare_parameter("sync_tolerance", 0.01)
        ros_node.declare_parameter("max_pose_age_sec", 0.1)
        ros_node.declare_parameter("no_data_timeout_sec", 5.0)
        ros_node.declare_parameter("frame_queue_maxsize", 32)
        # If > 0, __getitem__ returns None after this many seconds of no frame
        # so the trainer can run train_with_frame(None) on existing keyframes.
        # 0 = block until a frame arrives or shutdown.
        ros_node.declare_parameter("idle_block_sec", 0.0)

        self._modality = ros_node.get_parameter("modality").value
        self._sync_tol = float(ros_node.get_parameter("sync_tolerance").value)
        self._max_pose_age = float(ros_node.get_parameter("max_pose_age_sec").value)
        self._no_data_timeout = float(ros_node.get_parameter("no_data_timeout_sec").value)
        self._idle_block_sec = float(ros_node.get_parameter("idle_block_sec").value)
        maxsize = int(ros_node.get_parameter("frame_queue_maxsize").value)

        # state
        self._frame_q: queue.Queue[Frame] = queue.Queue(maxsize=maxsize)
        self._shutdown = threading.Event()
        self._frame_count = 0
        self._last_msg_time: Optional[float] = None  # wall-clock seconds
        self._lock = threading.Lock()

        # pose pairing buffers (for depth) / latest pose (for lidar)
        self._latest_depth: Optional[tuple[float, torch.Tensor]] = None  # (stamp, depth)
        self._latest_pose: Optional[tuple[float, torch.Tensor]] = None   # (stamp, pose 4x4)

        # idle callback: invoked while __getitem__ is waiting for a frame
        self.on_idle: Optional[Callable[[], None]] = None

        # subscribers + watchdog use a single mutually-exclusive group so the
        # producer side stays single-threaded (avoids races on _latest_*).
        self._sub_cbg = MutuallyExclusiveCallbackGroup()
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        if self._modality == "depth":
            assert self.intrinsics is not None, "intrinsics required for modality='depth'"
            depth_topic = ros_node.get_parameter("depth_topic").value
            pose_topic = ros_node.get_parameter("pose_topic").value
            ros_node.create_subscription(
                Image, depth_topic, self._depth_cb, sensor_qos, callback_group=self._sub_cbg
            )
            ros_node.create_subscription(
                PoseStamped, pose_topic, self._pose_cb, sensor_qos, callback_group=self._sub_cbg
            )
            ros_node.get_logger().info(
                f"[RosDataLoader] depth modality: subscribed to {depth_topic} + {pose_topic}"
            )
        elif self._modality == "lidar":
            lidar_topic = ros_node.get_parameter("lidar_topic").value
            odom_topic = ros_node.get_parameter("odom_topic").value
            ros_node.create_subscription(
                PointCloud2, lidar_topic, self._lidar_cb, sensor_qos, callback_group=self._sub_cbg
            )
            ros_node.create_subscription(
                Odometry, odom_topic, self._odom_cb, sensor_qos, callback_group=self._sub_cbg
            )
            ros_node.get_logger().info(
                f"[RosDataLoader] lidar modality: subscribed to {lidar_topic} + {odom_topic}"
            )
        else:
            raise ValueError(f"unknown modality: {self._modality!r} (expected 'depth' or 'lidar')")

        # 1 Hz watchdog
        self._watchdog = ros_node.create_timer(
            1.0, self._check_no_data_timeout, callback_group=self._sub_cbg
        )

    # ------------------------- Trainer-facing interface -------------------------

    def __len__(self) -> int:
        return -1  # streaming sentinel

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown.is_set()

    def __getitem__(self, idx: int) -> Optional[Frame]:
        """Block until a frame is ready, transient idle expires, or shutdown.

        Returns:
            Frame if one is available.
            None if either (a) `idle_block_sec` elapsed without a new frame, or
            (b) the loader has been shut down. Caller distinguishes the two
            cases via `is_shutdown`.
        """
        deadline = (
            time.monotonic() + self._idle_block_sec if self._idle_block_sec > 0 else None
        )
        while not self._shutdown.is_set():
            try:
                return self._frame_q.get(timeout=0.05)
            except queue.Empty:
                if self.on_idle is not None:
                    self.on_idle()
                if deadline is not None and time.monotonic() >= deadline:
                    return None  # transient idle
        return None  # shutdown

    def shutdown(self) -> None:
        self._shutdown.set()

    # -------------------------- ROS callbacks (executor) -------------------------

    def _depth_cb(self, msg: Image) -> None:
        try:
            self._last_msg_time = time.monotonic()
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            # Assume 32FC1 depth in metres
            depth_np = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            depth = torch.from_numpy(depth_np.copy())
            with self._lock:
                self._latest_depth = (stamp, depth)
            self._try_emit_depth_frame()
        except Exception as e:
            self.node.get_logger().error(f"[RosDataLoader] depth_cb: {e}")

    def _pose_cb(self, msg: PoseStamped) -> None:
        try:
            self._last_msg_time = time.monotonic()
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            pose = _pose_stamped_to_matrix(msg)
            with self._lock:
                self._latest_pose = (stamp, pose)
            self._try_emit_depth_frame()
        except Exception as e:
            self.node.get_logger().error(f"[RosDataLoader] pose_cb: {e}")

    def _lidar_cb(self, msg: PointCloud2) -> None:
        try:
            self._last_msg_time = time.monotonic()
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            with self._lock:
                latest_pose = self._latest_pose
            if latest_pose is None:
                self.node.get_logger().debug("[RosDataLoader] no odom yet, dropping point cloud")
                return
            pose_stamp, pose = latest_pose
            if abs(stamp - pose_stamp) > self._max_pose_age:
                self.node.get_logger().warn(
                    f"[RosDataLoader] odom too old (Δt={abs(stamp - pose_stamp):.3f}s); dropping cloud"
                )
                return
            points_np = np.frombuffer(msg.data, dtype=np.float32).reshape(
                -1, msg.point_step // 4
            )[:, :3]
            points = torch.from_numpy(points_np.copy())
            self._frame_count += 1
            frame = LiDARFrame(fid=self._frame_count, pointcloud=points, ref_pose=pose)
            frame.apply_bound(self.bound_min, self.bound_max)
            self._enqueue(frame)
        except Exception as e:
            self.node.get_logger().error(f"[RosDataLoader] lidar_cb: {e}")

    def _odom_cb(self, msg: Odometry) -> None:
        try:
            self._last_msg_time = time.monotonic()
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            pose = _odom_to_matrix(msg)
            with self._lock:
                self._latest_pose = (stamp, pose)
        except Exception as e:
            self.node.get_logger().error(f"[RosDataLoader] odom_cb: {e}")

    # ------------------------------- internals ----------------------------------

    def _try_emit_depth_frame(self) -> None:
        with self._lock:
            d = self._latest_depth
            p = self._latest_pose
            if d is None or p is None:
                return
            d_stamp, depth = d
            p_stamp, pose = p
            if abs(d_stamp - p_stamp) > self._sync_tol:
                return  # not synchronized yet
            # consume both
            self._latest_depth = None
            self._latest_pose = None
        self._frame_count += 1
        frame = DepthFrame(
            fid=self._frame_count,
            depth=depth,
            intrinsic=self.intrinsics,
            ref_pose=pose,
            min_depth=self.min_depth if self.min_depth > 0 else None,
            max_depth=self.max_depth if self.max_depth > 0 else None,
        )
        # bound clipping is automatic via the valid_mask path inside DepthFrame
        self._enqueue(frame)

    def _enqueue(self, frame: Frame) -> None:
        try:
            self._frame_q.put_nowait(frame)
        except queue.Full:
            try:
                _ = self._frame_q.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            self._frame_q.put_nowait(frame)
            self.node.get_logger().warn("[RosDataLoader] frame queue full; dropped oldest frame")

    def _check_no_data_timeout(self) -> None:
        if self._last_msg_time is None or self._shutdown.is_set():
            return
        elapsed = time.monotonic() - self._last_msg_time
        if elapsed > self._no_data_timeout:
            self.node.get_logger().info(
                f"[RosDataLoader] no data for {elapsed:.1f}s (> {self._no_data_timeout:.1f}s); "
                "shutting down stream."
            )
            self.shutdown()


# ------------------------------ helpers ---------------------------------------


def _pose_stamped_to_matrix(msg: PoseStamped) -> torch.Tensor:
    p = msg.pose.position
    q = msg.pose.orientation
    rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3, 3] = [p.x, p.y, p.z]
    return torch.from_numpy(T)


def _odom_to_matrix(msg: Odometry) -> torch.Tensor:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot
    T[:3, 3] = [p.x, p.y, p.z]
    return torch.from_numpy(T)
