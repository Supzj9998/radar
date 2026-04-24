from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="model/ONNX/best.onnx",
        description="ONNX model path used to build a TensorRT engine when needed.",
    )

    engine_path_arg = DeclareLaunchArgument(
        "engine_path",
        default_value="model/TensorRT/best.engine",
        description="TensorRT engine path for detect node.",
    )

    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="image_raw",
        description="Input image topic for detect node.",
    )

    boxes_topic_arg = DeclareLaunchArgument(
        "boxes_topic",
        default_value="detect/boxes",
        description="Boxes topic shared between detect and pnp.",
    )

    require_extrinsic_arg = DeclareLaunchArgument(
        "require_extrinsic",
        default_value="false",
        description="Whether pnp must receive extrinsic before publishing output.",
    )
    world_extrinsic_file_arg = DeclareLaunchArgument(
        "world_extrinsic_file",
        default_value="config/out_matrix.yaml",
        description="Static world extrinsic yaml path containing world_rvec/world_tvec.",
    )

    detect_node = Node(
        package="drone_detect",
        executable="detect_node",
        name="detect_node",
        output="screen",
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "engine_path": LaunchConfiguration("engine_path"),
                "image_topic": LaunchConfiguration("image_topic"),
                "output_boxes_topic": LaunchConfiguration("boxes_topic"),
            }
        ],
    )

    pnp_node = Node(
        package="drone_detect",
        executable="pnp_node",
        name="pnp_node",
        output="screen",
        parameters=[
            {
                "boxes_topic": LaunchConfiguration("boxes_topic"),
                "require_extrinsic": LaunchConfiguration("require_extrinsic"),
                "use_world_extrinsic_file": False,
                "use_static_laser_extrinsic": True,
                "world_extrinsic_file": LaunchConfiguration("world_extrinsic_file"),
                "use_extrinsic_topic": False,
                "use_tf_topic": False,
            }
        ],
    )

    return LaunchDescription(
        [
            model_path_arg,
            engine_path_arg,
            image_topic_arg,
            boxes_topic_arg,
            require_extrinsic_arg,
            world_extrinsic_file_arg,
            detect_node,
            pnp_node,
        ]
    )
