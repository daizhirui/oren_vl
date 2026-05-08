"""Query SDF on a 2D grid and publish the result as a GridMap and/or PointCloud2."""

import threading
from typing import Optional

import numpy as np
import rclpy
import tf2_ros
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Point
from grid_map_msgs.msg import GridMap, GridMapInfo
from rclpy.node import Node
from rclpy.task import Future
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from std_srvs.srv import Trigger

from oren_msgs.srv import QuerySdf


class SdfQueryNode(Node):

    def __init__(self):
        super().__init__("sdf_query_node")

        # --- parameters ----------------------------------------------------
        self.declare_parameter("resolution", 0.1)
        self.declare_parameter("x_cells", 101)
        self.declare_parameter("y_cells", 101)
        self.declare_parameter("x", 0.0)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("z", 0.0)
        self.declare_parameter("publish_gradient", False)
        self.declare_parameter("publish_grid_map", True)
        self.declare_parameter("publish_point_cloud", False)
        self.declare_parameter("publish_rate", 2.0)
        self.declare_parameter("attached_to_frame", False)
        self.declare_parameter("attached_frame", "sdf_query_frame")
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("query_prior", False)
        self.declare_parameter("query_service", "query_sdf")
        self.declare_parameter("map_topic", "sdf_grid_map")
        self.declare_parameter("point_cloud_topic", "sdf_point_cloud")
        self.declare_parameter("trigger_service", "sdf_trigger_query")

        self._resolution = float(self.get_parameter("resolution").value)
        self._x_cells = int(self.get_parameter("x_cells").value)
        self._y_cells = int(self.get_parameter("y_cells").value)
        self._x = float(self.get_parameter("x").value)
        self._y = float(self.get_parameter("y").value)
        self._z = float(self.get_parameter("z").value)
        self._publish_gradient = bool(self.get_parameter("publish_gradient").value)
        self._publish_grid_map = bool(self.get_parameter("publish_grid_map").value)
        self._publish_point_cloud = bool(self.get_parameter("publish_point_cloud").value)
        self._publish_rate = float(self.get_parameter("publish_rate").value)
        self._attached_to_frame = bool(self.get_parameter("attached_to_frame").value)
        self._attached_frame = str(self.get_parameter("attached_frame").value)
        self._world_frame = str(self.get_parameter("world_frame").value)
        self._query_prior = bool(self.get_parameter("query_prior").value)

        if self._resolution <= 0.0:
            raise ValueError(f"resolution must be > 0, got {self._resolution}")
        if self._x_cells <= 0 or self._y_cells <= 0:
            raise ValueError("x_cells and y_cells must be positive")
        # Force odd cell counts so the origin sits on a cell center.
        if self._x_cells % 2 == 0:
            self._x_cells += 1
            self.get_logger().warn(f"x_cells must be odd; using {self._x_cells}")
        if self._y_cells % 2 == 0:
            self._y_cells += 1
            self.get_logger().warn(f"y_cells must be odd; using {self._y_cells}")
        if not self._publish_grid_map and not self._publish_point_cloud:
            raise ValueError("at least one of publish_grid_map / publish_point_cloud must be true")
        if self._attached_to_frame and (not self._attached_frame or not self._world_frame):
            raise ValueError("attached_to_frame=true requires non-empty attached_frame and world_frame")

        self._frame_id = self._attached_frame if self._attached_to_frame else self._world_frame

        # --- query points (in attached_frame if attached_to_frame, else world)
        self._query_points_local = self._make_grid_points()  # (N, 3) float64

        # --- ROS plumbing --------------------------------------------------
        query_service = str(self.get_parameter("query_service").value)
        map_topic = str(self.get_parameter("map_topic").value)
        cloud_topic = str(self.get_parameter("point_cloud_topic").value)
        trigger_service = str(self.get_parameter("trigger_service").value)

        self._sdf_client = self.create_client(QuerySdf, query_service)
        self._pub_map = self.create_publisher(GridMap, map_topic, 10) if self._publish_grid_map else None
        self._pub_pcd = self.create_publisher(PointCloud2, cloud_topic, 10) if self._publish_point_cloud else None
        self.create_service(Trigger, trigger_service, self._on_trigger)

        if self._attached_to_frame:
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        else:
            self._tf_buffer = None
            self._tf_listener = None

        self._inflight = False
        self._inflight_stamp: Optional[Time] = None
        self._lock = threading.Lock()

        if self._publish_rate > 0.0:
            self.create_timer(1.0 / self._publish_rate, self._on_timer)
        else:
            self.get_logger().info("publish_rate <= 0, timer disabled (use trigger service)")

        self.get_logger().info(
            f"SdfQueryNode ready: grid {self._x_cells}x{self._y_cells} @ {self._resolution}m, "
            f"frame={self._frame_id}, service={query_service}"
        )

    # ------------------------------------------------------------------ helpers

    def _make_grid_points(self) -> np.ndarray:
        """Generate (N, 3) grid points in column-major (x fastest) order, matching grid_map's layout."""
        half_x = (self._x_cells - 1) // 2
        half_y = (self._y_cells - 1) // 2
        # iterate y outer (high → low), x inner (high → low) so consecutive points share a y row
        ys = np.arange(half_y, -half_y - 1, -1, dtype=np.float64) * self._resolution + self._y
        xs = np.arange(half_x, -half_x - 1, -1, dtype=np.float64) * self._resolution + self._x
        gx, gy = np.meshgrid(xs, ys, indexing="xy")  # both (y_cells, x_cells)
        pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, self._z)], axis=1)
        return pts

    def _query_points_world(self) -> Optional[tuple[np.ndarray, Time]]:
        """Return query points in the world frame plus the timestamp to use on outgoing msgs."""
        if not self._attached_to_frame:
            return self._query_points_local, self.get_clock().now()
        try:
            tf = self._tf_buffer.lookup_transform(self._world_frame, self._attached_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"tf lookup {self._world_frame}<-{self._attached_frame} failed: {e}")
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        pts_world = self._query_points_local @ R.T + np.array([t.x, t.y, t.z])
        return pts_world, Time.from_msg(tf.header.stamp)

    def _has_subscribers(self) -> bool:
        if self._pub_map is not None and self._pub_map.get_subscription_count() > 0:
            return True
        if self._pub_pcd is not None and self._pub_pcd.get_subscription_count() > 0:
            return True
        return False

    def _build_request(self, pts_world: np.ndarray) -> QuerySdf.Request:
        req = QuerySdf.Request()
        req.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts_world]
        req.return_grad = self._publish_gradient
        req.prior = self._query_prior
        return req

    # ------------------------------------------------------------------ callbacks

    def _on_timer(self) -> None:
        if not self._has_subscribers():
            return
        with self._lock:
            if self._inflight:
                return
            if not self._sdf_client.service_is_ready():
                self.get_logger().warn(f"service {self._sdf_client.srv_name} not ready", throttle_duration_sec=5.0)
                return
            qp = self._query_points_world()
            if qp is None:
                return
            pts_world, stamp = qp
            self._inflight = True
            self._inflight_stamp = stamp

        future = self._sdf_client.call_async(self._build_request(pts_world))
        future.add_done_callback(self._on_response)

    def _on_response(self, future: Future) -> None:
        with self._lock:
            stamp = self._inflight_stamp
            self._inflight = False
            self._inflight_stamp = None
        try:
            response: QuerySdf.Response = future.result()
        except Exception as e:
            self.get_logger().warn(f"service call failed: {e}")
            return
        self._publish(response, stamp)

    def _on_trigger(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        if not self._sdf_client.service_is_ready():
            response.success = False
            response.message = f"service {self._sdf_client.srv_name} not ready"
            return response
        qp = self._query_points_world()
        if qp is None:
            response.success = False
            response.message = "tf lookup failed"
            return response
        pts_world, stamp = qp
        future = self._sdf_client.call_async(self._build_request(pts_world))
        # block until the executor delivers the response
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done():
            response.success = False
            response.message = "service call timed out"
            return response
        try:
            query_response: QuerySdf.Response = future.result()
        except Exception as e:
            response.success = False
            response.message = f"service call raised: {e}"
            return response
        try:
            self._publish(query_response, stamp)
        except Exception as e:
            response.success = False
            response.message = f"publish failed: {e}"
            return response
        response.success = True
        response.message = "queried and published"
        return response

    # ------------------------------------------------------------------ publish

    def _publish(self, response: QuerySdf.Response, stamp: Time) -> None:
        n = self._x_cells * self._y_cells
        if len(response.sdf) != n:
            self.get_logger().warn(f"response size {len(response.sdf)} != expected {n}; dropping")
            return
        sdf = np.asarray(response.sdf, dtype=np.float32)
        grad = None
        if self._publish_gradient:
            if len(response.grad) == n:
                grad = np.array([[g.x, g.y, g.z] for g in response.grad], dtype=np.float32)
            else:
                self.get_logger().warn(f"publish_gradient=true but grad size {len(response.grad)} != {n}")

        stamp_msg = stamp.to_msg()
        if self._pub_map is not None:
            self._pub_map.publish(self._make_grid_map(sdf, grad, stamp_msg))
        if self._pub_pcd is not None:
            self._pub_pcd.publish(self._make_point_cloud(sdf, grad, stamp_msg))

    def _make_grid_map(self, sdf: np.ndarray, grad: Optional[np.ndarray], stamp) -> GridMap:
        msg = GridMap()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id

        info = GridMapInfo()
        info.resolution = self._resolution
        info.length_x = self._x_cells * self._resolution
        info.length_y = self._y_cells * self._resolution
        info.pose.position.x = self._x if not self._attached_to_frame else 0.0
        info.pose.position.y = self._y if not self._attached_to_frame else 0.0
        info.pose.position.z = self._z if not self._attached_to_frame else 0.0
        info.pose.orientation.w = 1.0
        msg.info = info

        layers = ["sdf"]
        arrays = [sdf]
        if grad is not None:
            layers += ["gradient_x", "gradient_y", "gradient_z"]
            arrays += [grad[:, 0].copy(), grad[:, 1].copy(), grad[:, 2].copy()]
        msg.layers = layers
        msg.basic_layers = ["sdf"]
        msg.data = [self._layer_to_msg(a) for a in arrays]
        msg.outer_start_index = 0
        msg.inner_start_index = 0
        return msg

    def _layer_to_msg(self, flat: np.ndarray) -> Float32MultiArray:
        # grid_map convention: column-major, dim[0]=column_index (size=cols=y_cells),
        # dim[1]=row_index (size=rows=x_cells). The query points were generated
        # column-major over (rows=x_cells, cols=y_cells), so `flat` is already in
        # the right order.
        m = Float32MultiArray()
        m.layout = MultiArrayLayout()
        m.layout.dim = [
            MultiArrayDimension(
                label="column_index",
                size=self._y_cells,
                stride=self._x_cells * self._y_cells,
            ),
            MultiArrayDimension(
                label="row_index",
                size=self._x_cells,
                stride=self._x_cells,
            ),
        ]
        m.layout.data_offset = 0
        m.data = flat.astype(np.float32, copy=False).tolist()
        return m

    def _make_point_cloud(self, sdf: np.ndarray, grad: Optional[np.ndarray], stamp) -> PointCloud2:
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id

        names = ["x", "y", "z", "sdf"]
        if grad is not None:
            names += ["gradient_x", "gradient_y", "gradient_z"]
        fields = []
        for i, name in enumerate(names):
            fields.append(PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1))
        msg.fields = fields
        msg.point_step = 4 * len(names)
        msg.height = 1
        msg.is_bigendian = False
        msg.is_dense = False

        pts = self._query_points_local  # publish in the (attached or world) reference frame
        valid = np.isfinite(sdf)
        idx = np.nonzero(valid)[0]
        cols = [
            pts[idx, 0].astype(np.float32),
            pts[idx, 1].astype(np.float32),
            pts[idx, 2].astype(np.float32),
            sdf[idx],
        ]
        if grad is not None:
            cols += [grad[idx, 0], grad[idx, 1], grad[idx, 2]]
        # interleave: out[i, k] = cols[k][i]
        out = np.stack(cols, axis=1).astype(np.float32, copy=False)
        msg.width = out.shape[0]
        msg.row_step = msg.point_step * msg.width
        msg.data = out.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = SdfQueryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
