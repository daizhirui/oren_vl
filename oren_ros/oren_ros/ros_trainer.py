"""Wrapper around `Trainer` that owns the ROS interface (services + shutdown).

Mirrors the `GuiTrainer` pattern: builds a `Trainer`, attaches to its existing callback hooks
(`training_iteration_end_callback`, `training_frame_start_callback`), and registers ROS 2 services on a shared node.

Supports `SdfTrainer` (signed-distance field) and `OccTrainer` (occupancy logits). The trainer class is resolved
from `cfg.trainer_identifier` via `oren.utils.registry.get_trainer`; the `query_scalar_field` service then
dispatches to `query_sdf` or `query_occ` based on which method the resolved trainer exposes, and the response's
`value` field carries SDF metres or occupancy logits accordingly.

Threading: `Trainer.train()` runs on the main thread (CUDA + heavy work).
Subscriber and service callbacks run on a `MultiThreadedExecutor` spinning in a background daemon thread
(set up by the entry-point script). Service handlers enqueue requests into a `queue.Queue` and block on a `Future`;
the main thread drains the queue between iterations and while idle (via `data_stream.on_idle`), so all model access
is serialized.
"""

import queue
import threading
from concurrent.futures import Future
from typing import Callable

import torch
from geometry_msgs.msg import Vector3
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from oren.frame import Frame
from oren.occ_trainer import OccTrainer
from oren.sdf_trainer import SdfTrainer
from oren.utils.registry import get_trainer
from oren.trainer_config import TrainerConfig
from oren_msgs.srv import QueryScalarField, SaveMesh, SaveModel
from oren_ros.ros_dataloader import RosDataLoader


Trainer = SdfTrainer | OccTrainer


class RosTrainer:
    def __init__(self, trainer_cfg: TrainerConfig, ros_node: Node):
        self.ros_node = ros_node

        # Build the data loader directly so cfg.data.dataset_args never holds a Node.
        loader = RosDataLoader(ros_node=ros_node, **trainer_cfg.data.dataset_args)
        self.trainer: Trainer = get_trainer(trainer_cfg.trainer_identifier)(trainer_cfg, data_stream=loader)

        # Field name for the query_scalar_field service response; "sdf" or "occ".
        self._field = self.trainer.model.field_prefix
        self._query_fn = getattr(self.trainer, f"query_{self._field}")

        # Service request queue + shutdown signal.
        self._service_queue: queue.Queue[tuple[Callable[[Trainer], object], Future]] = queue.Queue()
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
            QueryScalarField, "query_scalar_field", self._srv_query_scalar_field, callback_group=srv_cbg
        )
        ros_node.create_service(SaveMesh, "save_mesh", self._srv_save_mesh, callback_group=srv_cbg)
        ros_node.create_service(SaveModel, "save_model", self._srv_save_model, callback_group=srv_cbg)

        ros_node.get_logger().info(
            f"[RosTrainer] services ready: query_scalar_field (field={self._field}) /save_mesh /save_model"
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
                fn, fut = self._service_queue.get_nowait()
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
        self._service_queue.put((fn, fut))
        return fut.result(timeout=timeout)

    def _srv_query_scalar_field(self, request, response):
        field = self._field  # "sdf" or "occ"
        try:
            points = torch.tensor(
                [[p.x, p.y, p.z] for p in request.points],
                dtype=torch.float32,
            )
            if points.shape[0] == 0:
                response.value = []
                response.grad = []
                return response
            out = self._submit(
                lambda t: self._query_fn(points, return_grad=request.return_grad, prior_only=request.prior)
            )
            # When request.prior is True, both query_sdf/query_occ set out[field] = prior values.
            value_t: torch.Tensor = out[field].detach().cpu().reshape(-1)
            response.value = value_t.tolist()
            if request.return_grad:
                grad = out["grad"]
                grad = grad[f"{field}_prior"] if request.prior else grad[field]
                grad = grad.detach().cpu().reshape(-1, 3)
                response.grad = [Vector3(x=float(g[0]), y=float(g[1]), z=float(g[2])) for g in grad]
            else:
                response.grad = []
        except Exception as e:
            self.ros_node.get_logger().error(f"[RosTrainer] query_scalar_field ({field}) failed: {e}")
            response.value = []
            response.grad = []
        return response

    def _srv_save_mesh(self, request, response):
        try:
            bound_min = list(request.bound_min) if request.use_bound else None
            bound_max = list(request.bound_max) if request.use_bound else None
            grid_resolution = request.resolution if request.resolution > 0.0 else None
            self._submit(
                lambda t: t.save_mesh(
                    request.path,
                    prior=request.prior,
                    bound_min=bound_min,
                    bound_max=bound_max,
                    grid_resolution=grid_resolution,
                    iso_value=request.iso_value,
                )
            )
            response.success = True
            response.message = f"mesh saved: {request.path} (prior={request.prior})"
        except Exception as e:
            response.success = False
            response.message = repr(e)
        return response

    def _srv_save_model(self, request, response):
        try:
            self._submit(lambda t: t.save_model(request.path))
            response.success = True
            response.message = f"model saved: {request.path}"
        except Exception as e:
            response.success = False
            response.message = repr(e)
        return response
