from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # 模型路径与 engine 路径分开配置：engine 不存在时 detect 节点会尝试由 ONNX 自动构建。
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

    # detect 发布的 boxes_topic 会被 pnp 订阅，是两个节点之间的核心数据通道。
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

    # YOLO/TensorRT 检测节点：输入图像，输出带框图像和 Float32MultiArray 检测框。
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

    # PnP 节点：把检测框转换为目标三维方向/距离，并可发布 AutoAIM 消息。
    pnp_node = Node(
        package="drone_detect",
        executable="pnp_node",
        name="pnp_node",
        output="screen",
        parameters=[
            {
                "boxes_topic": LaunchConfiguration("boxes_topic"),
                "require_extrinsic": LaunchConfiguration("require_extrinsic"),
                # 默认不使用世界外参文件，使用参数内 camera->laser 外参做方向补偿。
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
