"""Publish simulated /clock at a fixed rate.

Optionally seeds the start time from the first message received on /tf.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage


class ClockNode(Node):

    def __init__(self):
        """Declare parameters, validate them, and either start the clock timer or wait for the first /tf message."""
        super().__init__("clock_node")

        self.declare_parameter("clock_rate", 100.0)
        self.declare_parameter("start_time", 0.0)
        self.declare_parameter("start_time_from_tf", False)

        self._clock_rate = float(self.get_parameter("clock_rate").value)
        self._start_time = float(self.get_parameter("start_time").value)
        self._start_time_from_tf = bool(self.get_parameter("start_time_from_tf").value)

        if self._clock_rate <= 0.0:
            raise ValueError(f"Invalid clock_rate: {self._clock_rate}. Must be > 0.0")
        if self._start_time < 0.0:
            raise ValueError(f"Invalid start_time: {self._start_time}. Must be >= 0.0")

        self.get_logger().info(
            f"Loaded node parameters:\n"
            f"  clock_rate: {self._clock_rate}\n"
            f"  start_time: {self._start_time}\n"
            f"  start_time_from_tf: {self._start_time_from_tf}"
        )

        self._clock_pub = self.create_publisher(Clock, "/clock", 10)
        self._dt = Duration(seconds=1.0 / self._clock_rate)
        self._current_time: Time | None = None
        self._timer = None
        self._sub_tf = None

        if self._start_time_from_tf:
            self.get_logger().info(
                "Waiting for one message on '/tf' to determine start_time ..."
            )
            self._sub_tf = self.create_subscription(
                TFMessage, "/tf", self._callback_tf, 10
            )
        else:
            self._start_clock()

    def _callback_tf(self, msg: TFMessage) -> None:
        if not msg.transforms:
            return
        stamp = msg.transforms[0].header.stamp
        self._start_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        self.get_logger().info(
            f"Got start_time from /tf: {stamp.sec}.{stamp.nanosec:09d} "
            f"({self._start_time:.6f} s)"
        )
        self.destroy_subscription(self._sub_tf)
        self._sub_tf = None
        self._start_clock()

    def _start_clock(self) -> None:
        self.get_logger().info(
            f"Clock node started with rate: {self._clock_rate:.2f} Hz, "
            f"start_time: {self._start_time:.6f}"
        )
        self._current_time = Time(nanoseconds=int(self._start_time * 1e9))
        self._timer = self.create_timer(1.0 / self._clock_rate, self._callback_timer)

    def _callback_timer(self) -> None:
        msg = Clock()
        msg.clock = self._current_time.to_msg()
        self._clock_pub.publish(msg)
        self._current_time = self._current_time + self._dt


def main(args=None):
    """Spin a ClockNode until shutdown.

    Args:
        args: Optional CLI argument list forwarded to ``rclpy.init``.
    """
    rclpy.init(args=args)
    node = ClockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
