import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _include_launch(package_name, launch_file, condition):
    launch_path = os.path.join(
        get_package_share_directory(package_name),
        'launch',
        launch_file,
    )
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_path),
        condition=condition,
    )


def generate_launch_description():
    enable_camera = LaunchConfiguration('enable_camera')
    enable_vision = LaunchConfiguration('enable_vision')
    enable_livox = LaunchConfiguration('enable_livox')
    enable_lidar = LaunchConfiguration('enable_lidar')
    enable_debug_map = LaunchConfiguration('enable_debug_map')
    enable_drone = LaunchConfiguration('enable_drone')
    enable_contact = LaunchConfiguration('enable_contact')
    enable_serial_bridge = LaunchConfiguration('enable_serial_bridge')

    camera_launch = _include_launch(
        'camera', 'dual_camera.launch.py', IfCondition(enable_camera))
    livox_launch = _include_launch(
        'livox_ros2_driver', 'livox_lidar_launch.py', IfCondition(enable_livox))
    vision_launch = _include_launch(
        'tdt_vision', 'radar.launch.py', IfCondition(enable_vision))
    lidar_launch = _include_launch(
        'dynamic_cloud', 'lidar.launch.py', IfCondition(enable_lidar))
    drone_launch = _include_launch(
        'drone_detect', 'drone_detect.launch.py', IfCondition(enable_drone))
    contact_launch = _include_launch(
        'contact', 'contact_node.launch.py', IfCondition(enable_contact))
    serial_bridge_launch = _include_launch(
        'serial_bridge', 'serial_bridge.launch.py', IfCondition(enable_serial_bridge))

    debug_map_node = Node(
        package='debug_map',
        executable='debug_map',
        name='debug_map',
        output='screen',
        condition=IfCondition(enable_debug_map),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'enable_camera', default_value='true',
            description='Start camera dual_camera.launch.py.'),
        DeclareLaunchArgument(
            'enable_vision', default_value='true',
            description='Start tdt_vision radar.launch.py.'),
        DeclareLaunchArgument(
            'enable_livox', default_value='true',
            description='Start Livox driver.'),
        DeclareLaunchArgument(
            'enable_lidar', default_value='true',
            description='Start lidar recognition pipeline.'),
        DeclareLaunchArgument(
            'enable_debug_map', default_value='true',
            description='Start debug_map map fusion node.'),
        DeclareLaunchArgument(
            'enable_drone', default_value='true',
            description='Start drone_detect launch.'),
        DeclareLaunchArgument(
            'enable_contact', default_value='true',
            description='Start lower-machine contact bridge on /dev/ttyACM0.'),
        DeclareLaunchArgument(
            'enable_serial_bridge', default_value='true',
            description='Start referee serial bridge on /dev/ttyUSB0.'),

        LogInfo(msg='Before starting: sudo chmod a+rw /dev/ttyACM0 /dev/ttyUSB0'),

        camera_launch,
        livox_launch,
        contact_launch,
        serial_bridge_launch,

        TimerAction(period=2.0, actions=[
            vision_launch,
            lidar_launch,
        ]),
        TimerAction(period=3.0, actions=[
            debug_map_node,
            drone_launch,
        ]),
    ])
