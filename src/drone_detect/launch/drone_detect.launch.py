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

    trt_workspace_size_mb_arg = DeclareLaunchArgument(
        "trt_workspace_size_mb",
        default_value="1024",
        description="TensorRT builder workspace size in MB.",
    )

    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="image_raw",
        description="Input image topic for detect node.",
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
                "trt_workspace_size_mb": LaunchConfiguration("trt_workspace_size_mb"),
                "image_topic": LaunchConfiguration("image_topic"),
            }
        ],
    )

    # PnP 节点：把检测框转换为目标三维方向/距离，并可发布 AutoAIM 消息。
    pnp_node = Node(
        package="drone_detect",
        executable="pnp_node",
        name="pnp_node",
        output="screen",
    )

    return LaunchDescription(
        [
            model_path_arg,
            engine_path_arg,
            trt_workspace_size_mb_arg,
            image_topic_arg,
            detect_node,
            pnp_node,
        ]
    )
