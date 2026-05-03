"""Wrapper around `Trainer` that owns the ROS interface (services + shutdown).

Mirrors the `GuiTrainer` pattern: builds a `Trainer`, attaches to its existing
callback hooks (`training_iteration_end_callback`, `training_frame_start_callback`),
and registers ROS 2 services on a shared node.

Threading: `Trainer.train()` runs on the main thread (CUDA + heavy work).
Subscriber and service callbacks run on a `MultiThreadedExecutor` spinning in
a background daemon thread (set up by the entry-point script). Service handlers
enqueue requests into a `queue.Queue` and block on a `Future`; the main thread
drains the queue between iterations and while idle (via `data_stream.on_idle`),
so all model access is serialized.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from typing import Callable, Optional

import torch
from geometry_msgs.msg import Point, Vector3
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from oren.frame import Frame
from oren.trainer import Trainer
from oren.trainer_config import TrainerConfig
from oren_msgs.srv import QuerySdf, SaveMesh, SaveModel

from oren_ros.ros_wrapper import DataLoader as RosDataLoader


class RosTrainer:
    def __init__(self, trainer_cfg: TrainerConfig, ros_node: Node):
        self.ros_node = ros_node

        # Build the data loader directly so cfg.data.dataset_args never holds a Node.
        loader = RosDataLoader(ros_node=ros_node, **trainer_cfg.data.dataset_args)
        self.trainer = Trainer(trainer_cfg, data_stream=loader)

        # Service request queue + shutdown signal.
        self._service_q: queue.Queue[tuple[Callable[[Trainer], object], Future]] = queue.Queue()
        self._shutdown = threading.Event()

        # Trainer callbacks (run on the main / training thread).
        self.trainer.training_iteration_end_callback = self._drain_services
        self.trainer.training_frame_start_callback = self._on_frame_start
        # Idle drain while __getitem__ is blocked waiting for data.
        loader.on_idle = lambda: self._drain_services(self.trainer)

        # Service registration. Reentrant group so a handler waiting on Future.result()
        # doesn't block other service / subscriber callbacks.
        srv_cbg = ReentrantCallbackGroup()
        ros_node.create_service(
            QuerySdf, "query_sdf", self._srv_query_sdf, callback_group=srv_cbg
        )
        ros_node.create_service(
            SaveMesh, "save_mesh", self._srv_save_mesh, callback_group=srv_cbg
        )
        ros_node.create_service(
            SaveModel, "save_model", self._srv_save_model, callback_group=srv_cbg
        )

        ros_node.get_logger().info(
            "[RosTrainer] services ready: /query_sdf /save_mesh /save_model"
        )

    # ---------------------- main entry point ----------------------------------

    def run(self) -> None:
        try:
            self.trainer.train()
        except KeyboardInterrupt:
            self.ros_node.get_logger().info("[RosTrainer] KeyboardInterrupt — shutting down.")
            self._shutdown.set()
            # Make sure the data loader stops blocking too.
            if hasattr(self.trainer.data_stream, "shutdown"):
                self.trainer.data_stream.shutdown()
            raise

    # ---------------------- callbacks (main thread) ---------------------------

    def _drain_services(self, trainer: Trainer) -> None:
        while True:
            try:
                fn, fut = self._service_q.get_nowait()
            except queue.Empty:
                return
            try:
                fut.set_result(fn(trainer))
            except Exception as e:  # noqa: BLE001 — propagate to the service caller
                fut.set_exception(e)

    def _on_frame_start(self, trainer: Trainer, frame: Frame) -> bool:
        return not self._shutdown.is_set()

    # ---------------------- service plumbing (executor thread) ----------------

    def _submit(self, fn: Callable[[Trainer], object], timeout: float = 60.0):
        fut: Future = Future()
        self._service_q.put((fn, fut))
        return fut.result(timeout=timeout)

    def _srv_query_sdf(self, request, response):
        try:
            points = torch.tensor(
                [[p.x, p.y, p.z] for p in request.points],
                dtype=torch.float32,
            )
            if points.shape[0] == 0:
                response.sdf = []
                response.grad = []
                return response
            out = self._submit(lambda t: t.query_sdf(points, return_grad=True))
            sdf_t: torch.Tensor = out["sdf"].detach().cpu().reshape(-1)
            response.sdf = sdf_t.tolist()
            grad_t: Optional[torch.Tensor] = out.get("sdf_grad")
            if grad_t is not None:
                grad_t = grad_t.detach().cpu().reshape(-1, 3)
                response.grad = [Vector3(x=float(g[0]), y=float(g[1]), z=float(g[2])) for g in grad_t]
            else:
                response.grad = []
        except Exception as e:  # noqa: BLE001
            self.ros_node.get_logger().error(f"[RosTrainer] query_sdf failed: {e}")
            response.sdf = []
            response.grad = []
        return response

    def _srv_save_mesh(self, request, response):
        try:
            self._submit(lambda t: t.save_mesh(request.path, prior=request.prior))
            response.success = True
            response.message = f"mesh saved: {request.path} (prior={request.prior})"
        except Exception as e:  # noqa: BLE001
            response.success = False
            response.message = repr(e)
        return response

    def _srv_save_model(self, request, response):
        try:
            self._submit(lambda t: t.save_model(request.path))
            response.success = True
            response.message = f"model saved: {request.path}"
        except Exception as e:  # noqa: BLE001
            response.success = False
            response.message = repr(e)
        return response
