from ament_index_python import get_package_share_path
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration


def declare_launch_args(launch_description: LaunchDescription):
    pkg_path = get_package_share_path('scalar_field_sim')
    scenario_path = str(pkg_path / 'scenarios/simple_field_1.toml')

    action = DeclareLaunchArgument(
        'scenario_path',
        default_value=scenario_path,
    )
    launch_description.add_action(action)

    action = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
    )
    launch_description.add_action(action)


def create_field_server_node():
    return Node(
        package='scalar_field_sim',
        executable='field_server_node.py',
        name='field_server',
        parameters=[
            {
                'scenario_path': LaunchConfiguration('scenario_path'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }
        ],
        output='screen',
        emulate_tty=True,
    )


def create_field_visualization_node():
    return Node(
        package='scalar_field_sim',
        executable='field_visualization_node.py',
        name='field_visualization',
        parameters=[
            {
                'scenario_path': LaunchConfiguration('scenario_path'),
                'grid_step': 0.01,
                'z_mode': 'flat',
                'height_scale': 0.5,
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }
        ],
        output='screen',
        emulate_tty=True,
    )


def generate_launch_description():
    launch_description = LaunchDescription()
    declare_launch_args(launch_description=launch_description)

    action = GroupAction(
        [
            create_field_server_node(),
            create_field_visualization_node(),
        ]
    )
    launch_description.add_action(action)

    return launch_description