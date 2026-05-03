"""Launch oren_ros mapping_node alongside `ros2 bag play`.

ROS-side parameters (topics, modality, sync) are passed via launch args
`--ros-args -p name:=value` after the launch line, or by editing the
`parameters=[...]` list below.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share_root = Path(get_package_share_directory("oren_ros"))
    default_config_path = Path.cwd() / "configs" / "v2" / "trainer-ros.yaml"
    if not default_config_path.exists():
        default_config_path = share_root / "configs" / "v2" / "trainer-ros.yaml"

    default_bag_path = Path.cwd() / "data" / "newercollege-ros2"
    if not default_bag_path.exists():
        default_bag_path = share_root / "data" / "newercollege-ros2"

    bag_path_arg = DeclareLaunchArgument(
        "bag_path",
        default_value=default_bag_path.as_posix(),
        description="ROS 2 bag directory path.",
    )
    config_path_arg = DeclareLaunchArgument(
        "config_path",
        default_value=default_config_path.as_posix(),
        description="Trainer config YAML path.",
    )
    play_rate_arg = DeclareLaunchArgument(
        "play_rate",
        default_value="1.0",
        description="ros2 bag play rate.",
    )
    bag_delay_arg = DeclareLaunchArgument(
        "bag_delay",
        default_value="1.0",
        description="Seconds to wait before starting ros2 bag play.",
    )

    bag_path = LaunchConfiguration("bag_path")
    config_path = LaunchConfiguration("config_path")
    play_rate = LaunchConfiguration("play_rate")
    bag_delay = LaunchConfiguration("bag_delay")

    mapping_node = Node(
        package="oren_ros",
        executable="mapping_node",
        name="oren_mapping_node",
        output="screen",
        arguments=["--config", config_path],
    )

    bag_play = ExecuteProcess(
        cmd=["ros2", "bag", "play", bag_path, "-r", play_rate],
        output="screen",
    )

    return LaunchDescription(
        [
            bag_path_arg,
            config_path_arg,
            play_rate_arg,
            bag_delay_arg,
            LogInfo(
                msg=[
                    "Starting oren_ros mapping_node with config: ",
                    config_path,
                    " | bag: ",
                    bag_path,
                ]
            ),
            mapping_node,
            TimerAction(period=bag_delay, actions=[bag_play]),
        ]
    )
