"""Single ROS 2 entry point for oren mapping.

Spins rclpy on a background daemon thread and runs `RosTrainer.run()` (which
internally drives `Trainer.train()`) on the main thread. ROS-side parameters
are declared by `RosDataLoader` / `RosTrainer` themselves; override at launch
with `--ros-args -p modality:=lidar -p depth_topic:=/cam/depth` etc.
"""
from __future__ import annotations

import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions

from oren.trainer_config import TrainerConfig
from oren_ros.ros_trainer import RosTrainer
from oren_ros.tqdm_redirect import install_ros_tqdm_redirect


def main(args=None):
    # Disable rclpy's own SIGINT handler so KeyboardInterrupt lands in
    # `Trainer.train()`'s try/finally on the main thread (where cleanup runs).
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("oren_mapping_node")
    install_ros_tqdm_redirect(node)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.declare_parameter("trainer_config_path", "")
        config_path = node.get_parameter("trainer_config_path").value
        assert config_path, "trainer_config_path parameter is required. Pass via --ros-args -p trainer_config_path:=/path/to/config.yaml"
        cfg = TrainerConfig.from_yaml(config_path)
        ros_trainer = RosTrainer(cfg, node)
        ros_trainer.run()
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
