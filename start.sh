#!/bin/bash
set -e

# 配置信息
ROS2_BASE_ENV="/opt/ros/jazzy/setup.bash"
WORKSPACE_DIR="/home/radar/01_RADAR/radar"
WORKSPACE_ENV="${WORKSPACE_DIR}/install/setup.bash"

# 1. 清理旧ROS2进程
echo "===== 1. 清理旧ROS2进程 ====="
pkill -f ros2 || true
pkill -f rclcpp || true
pkill -f livox || true
sleep 2

# 2. 加载ROS2环境
echo "===== 2. 加载ROS2环境 ====="
source ${ROS2_BASE_ENV}
source ${WORKSPACE_ENV}
echo "ROS2环境加载完成：${WORKSPACE_ENV}"

# 3. 串口设备提权
echo "===== 3. 串口设备提权 ====="
if [ -e /dev/ttyACM0 ]; then
    sudo chmod 777 /dev/ttyACM0
    echo "已赋权：/dev/ttyACM0"
else
    echo "未检测到串口设备：/dev/ttyACM0，跳过赋权并继续启动"
fi

if [ -e /dev/ttyUSB0 ]; then
    sudo chmod 777 /dev/ttyUSB0
    echo "已赋权：/dev/ttyUSB0"
else
    echo "错误：未检测到必要串口设备：/dev/ttyUSB0，终止启动"
    exit 1
fi

# 4. XFCE正确启动：弹终端+执行程序+改标题
echo "===== 4. 开始启动所有节点 ====="

# 相机节点
xfce4-terminal \
    --title="相机节点" \
    --hold \
    --command="bash -lc 'cd $WORKSPACE_DIR; source $ROS2_BASE_ENV; source $WORKSPACE_ENV; ros2 launch camera dual_camera.launch.py'" &
sleep 6

# 雷达主程序
xfce4-terminal \
    --title="雷达主程序" \
    --hold \
    --command="bash -lc 'cd $WORKSPACE_DIR; source $ROS2_BASE_ENV; source $WORKSPACE_ENV; ros2 launch tdt_vision radar.launch.py'" &
sleep 6

# Livox雷达驱动
xfce4-terminal \
    --title="Livox雷达驱动" \
    --hold \
    --command="bash -lc 'cd $WORKSPACE_DIR; source $ROS2_BASE_ENV; source $WORKSPACE_ENV; ros2 launch livox_ros2_driver livox_lidar_launch.py'" &
sleep 8

# 激光雷达识别
xfce4-terminal \
    --title="激光雷达识别" \
    --hold \
    --command="bash -lc 'cd $WORKSPACE_DIR; source $ROS2_BASE_ENV; source $WORKSPACE_ENV; ros2 launch dynamic_cloud lidar.launch.py'" &
sleep 5

# 地图融合
xfce4-terminal \
    --title="地图融合" \
    --hold \
    --command="bash -lc 'cd $WORKSPACE_DIR; source $ROS2_BASE_ENV; source $WORKSPACE_ENV; ros2 run debug_map debug_map'" &
sleep 5

# 无人机反制
#xfce4-terminal \
#    --title="无人机反制" \
#    --hold \
#    --command="bash -lc 'cd $WORKSPACE_DIR; source $ROS2_BASE_ENV; source $WORKSPACE_ENV; ros2 launch drone_detect drone_detect.launch.py'" &
#sleep 5

# 中继节点
#xfce4-terminal \
#    --title="中继节点" \
#    --hold \
#    --command="bash -lc 'cd $WORKSPACE_DIR; source $ROS2_BASE_ENV; source $WORKSPACE_ENV; ros2 launch contact contact_node.launch.py'" &
#sleep 5

# 启动串口通信（留在当前主终端，方便看串口日志）
cd "$WORKSPACE_DIR"
source "$ROS2_BASE_ENV"
source "$WORKSPACE_ENV"
ros2 launch serial_bridge serial_bridge.launch.py
