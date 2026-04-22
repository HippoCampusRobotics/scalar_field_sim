from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scenario_path",
                default_value=str(
                    get_package_share_path("scalar_field_sim")
                    / "scenarios/simple_field_1.toml"
                ),
            ),
            DeclareLaunchArgument("grid_step", default_value="0.05"),
            DeclareLaunchArgument("z_mode", default_value="flat"),
            DeclareLaunchArgument("height_scale", default_value="0.5"),
            Node(
                package="scalar_field_sim",
                executable="field_server_node.py",
                name="field_server",
                parameters=[{"scenario_path": LaunchConfiguration("scenario_path")}],
            ),
            Node(
                package="scalar_field_sim",
                executable="field_visualization_node.py",
                name="field_visualization",
                parameters=[
                    {
                        "scenario_path": LaunchConfiguration("scenario_path"),
                        "grid_step": LaunchConfiguration("grid_step"),
                        "z_mode": LaunchConfiguration("z_mode"),
                        "height_scale": LaunchConfiguration("height_scale"),
                    }
                ],
            ),
        ]
    )
