from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # 模型路径与 engine 路径分开配置：engine 不存在时 model_detecter 节点会尝试由 ONNX 自动构建。
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="model/ONNX/model.onnx",
        description="ONNX model path used to build a TensorRT engine when needed.",
    )

    engine_path_arg = DeclareLaunchArgument(
        "engine_path",
        default_value="model/TensorRT/model.engine",
        description="TensorRT engine path for model_detecter node.",
    )

    trt_workspace_size_mb_arg = DeclareLaunchArgument(
        "trt_workspace_size_mb",
        default_value="1024",
        description="TensorRT builder workspace size in MB.",
    )

    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="image_raw",
        description="Input image topic for model_detecter node.",
    )

    enable_model_detecter_arg = DeclareLaunchArgument(
        "enable_model_detecter",
        default_value="true",
        description="Enable model_detecter_node.",
    )

    enable_drone_detecter_arg = DeclareLaunchArgument(
        "enable_drone_detecter",
        default_value="true",
        description="Enable drone_detecter_node publishing drone_detecter/boxes.",
    )

    enable_model_pnp_arg = DeclareLaunchArgument(
        "enable_model_pnp",
        default_value="true",
        description="Enable model_pnp_node publishing /autoaim/model.",
    )

    enable_drone_pnp_arg = DeclareLaunchArgument(
        "enable_drone_pnp",
        default_value="true",
        description="Enable drone_pnp_node publishing /autoaim/drone.",
    )

    enable_autoaim_manager_arg = DeclareLaunchArgument(
        "enable_autoaim_manager",
        default_value="true",
        description="Enable autoaim_manager_node publishing /autoaim/target.",
    )

    # YOLO/TensorRT 检测节点：输入图像，输出带框图像和 Float32MultiArray 检测框。
    model_detecter_node = Node(
        package="drone_detect",
        executable="model_detecter_node",
        name="model_detecter_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_model_detecter")),
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "engine_path": LaunchConfiguration("engine_path"),
                "trt_workspace_size_mb": LaunchConfiguration("trt_workspace_size_mb"),
                "image_topic": LaunchConfiguration("image_topic"),
            }
        ],
    )

    # model_pnp 节点：把检测框转换为目标三维方向/距离，并可发布 AutoAIM 消息。
    model_pnp_node = Node(
        package="drone_detect",
        executable="model_pnp_node",
        name="model_pnp_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_model_pnp")),
    )

    # 固定广角相机无人机检测节点：启用后发布筛选后的无人机YOLO框。
    drone_detecter_node = Node(
        package="drone_detect",
        executable="drone_detecter_node",
        name="drone_detecter_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_drone_detecter")),
    )

    # drone_pnp 节点：接收无人机YOLO框，输出无人机方向AutoAIM消息。
    drone_pnp_node = Node(
        package="drone_detect",
        executable="drone_pnp_node",
        name="drone_pnp_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_drone_pnp")),
    )

    # autoaim_manager 节点：在模型目标和无人机目标之间选择最终/autoaim/target。
    autoaim_manager_node = Node(
        package="drone_detect",
        executable="autoaim_manager_node",
        name="autoaim_manager_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_autoaim_manager")),
    )

    return LaunchDescription(
        [
            model_path_arg,
            engine_path_arg,
            trt_workspace_size_mb_arg,
            image_topic_arg,
            enable_model_detecter_arg,
            enable_drone_detecter_arg,
            enable_model_pnp_arg,
            enable_drone_pnp_arg,
            enable_autoaim_manager_arg,
            model_detecter_node,
            model_pnp_node,
            drone_detecter_node,
            drone_pnp_node,
            autoaim_manager_node,
        ]
    )
