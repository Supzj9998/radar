#!/bin/bash
echo "===== 正在关闭所有ROS2进程 ====="
pkill -f ros2
pkill -f rclcpp
pkill -f livox
echo "所有ROS2进程已全部停止"
sleep 2
pkill xfce4-terminal
