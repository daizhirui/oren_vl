"""Launch oren_ros mapping_node alongside `ros2 bag play`.

ROS-side parameters (topics, modality, sync) are passed via launch args
`--ros-args -p name:=value` after the launch line, or by editing the
`parameters=[...]` list below.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def _rviz_action(context, *args, **kwargs):
    """Resolve `rviz_config` at runtime so we can omit `-d` when it is empty."""
    rviz_config = LaunchConfiguration("rviz_config").perform(context)
    rviz_args = ["-d", rviz_config] if rviz_config else []
    return [
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=rviz_args,
            condition=IfCondition(LaunchConfiguration("rviz")),
        )
    ]


def generate_launch_description():
    oren_ros_share = get_package_share_directory("oren_ros")

    trainer_config_path_arg = DeclareLaunchArgument(
        "trainer_config_path",
        default_value=PathJoinSubstitution([oren_ros_share, "configs", "trainer-ros.yaml"]),
        description="Trainer config YAML path.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Whether to use ROS simulation time (e.g. from a ros2 bag).",
    )
    bag_path_arg = DeclareLaunchArgument(
        "bag_path",
        default_value="ros2_bag",
        description="ROS 2 bag directory path.",
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
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="false",
        description="If true, launch RViz alongside the mapping node.",
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value="",
        description="Path to an RViz .rviz config file. Empty -> open RViz with its default layout.",
    )
    visualize_sdf_arg = DeclareLaunchArgument(
        "visualize_sdf",
        default_value="true",
        description="If true, launch sdf_query_node to publish SDF as GridMap / PointCloud2.",
    )

    bag_path = LaunchConfiguration("bag_path")
    trainer_config_path = LaunchConfiguration("trainer_config_path")
    play_rate = LaunchConfiguration("play_rate")
    bag_delay = LaunchConfiguration("bag_delay")
    use_sim_time = LaunchConfiguration("use_sim_time")

    mapping_node = Node(
        package="oren_ros",
        executable="mapping_node",
        name="oren_mapping_node",
        output="screen",
        parameters=[
            PathJoinSubstitution([oren_ros_share, "configs", "ros2-params.yaml"]),
            {"use_sim_time": use_sim_time},
            {"trainer_config_path": trainer_config_path},
        ],
    )

    sdf_query_node = Node(
        package="oren_ros",
        executable="sdf_query_node",
        name="sdf_query_node",
        output="screen",
        parameters=[
            PathJoinSubstitution([oren_ros_share, "configs", "ros2-params.yaml"]),
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(LaunchConfiguration("visualize_sdf")),
    )

    clock_node = Node(
        package="oren_ros",
        executable="clock_node",
        name="clock_node",
        output="screen",
        parameters=[{"start_time_from_tf": True}],
        condition=IfCondition(use_sim_time),
    )

    bag_play = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            bag_path,
            "-r",
            play_rate,
            "--remap",
            "/robot/tf:=/tf",
            "/robot/tf_static:=/tf_static",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            bag_path_arg,
            trainer_config_path_arg,
            play_rate_arg,
            bag_delay_arg,
            use_sim_time_arg,
            rviz_arg,
            rviz_config_arg,
            visualize_sdf_arg,
            LogInfo(
                msg=[
                    "Starting oren_ros mapping_node with config: ",
                    trainer_config_path,
                    " | bag: ",
                    bag_path,
                ]
            ),
            clock_node,
            mapping_node,
            sdf_query_node,
            OpaqueFunction(function=_rviz_action),
            TimerAction(period=bag_delay, actions=[bag_play]),
        ]
    )
