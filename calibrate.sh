#!/bin/bash
set -e

ROS2_BASE_ENV="/opt/ros/jazzy/setup.bash"
WORKSPACE_DIR="/home/radar/01_RADAR/radar"
WORKSPACE_ENV="${WORKSPACE_DIR}/install/setup.bash"

echo "===== 启动雷达标定程序 ====="

xfce4-terminal \
    --title="相机节点" \
    --hold \
    --command="bash -lc 'cd $WORKSPACE_DIR; source $ROS2_BASE_ENV; source $WORKSPACE_ENV; ros2 launch camera dual_camera.launch.py'" &
sleep 6

cd "$WORKSPACE_DIR"
source "$ROS2_BASE_ENV"
source "$WORKSPACE_ENV"

ros2 launch tdt_vision calibrate_radar.launch.py
