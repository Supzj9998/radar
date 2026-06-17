#ifndef DRONE_DETECT_DRONE_PNP_H
#define DRONE_DETECT_DRONE_PNP_H

#include <mutex>

#include "gary_msgs/msg/auto_aim.hpp"
#include "opencv2/core.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

namespace drone::drone_pnp {

class DronePnpNode final : public rclcpp::Node {
public:
    explicit DronePnpNode(const rclcpp::NodeOptions& options);

private:
    void
    boxesCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    void
    cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    bool      solveBox(const std_msgs::msg::Float32MultiArray& msg,
                       cv::Vec3d& camera_tvec) const;
    cv::Vec3d cameraToGimbal(const cv::Vec3d& camera_tvec) const;
    void calculateAngleOffset(const cv::Vec3d& gimbal_tvec, double& yaw,
                              double& pitch) const;
    void publishAngleOffset(double yaw, double pitch,
                            const cv::Vec3d& gimbal_tvec) const;

    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
        boxes_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
                                                          camera_info_sub_;
    rclcpp::Publisher<gary_msgs::msg::AutoAIM>::SharedPtr autoaim_pub_;

    mutable std::mutex mutex_;
    cv::Mat            camera_matrix_;
    cv::Mat            dist_coeffs_;
    bool               camera_info_ready_ = false;

    // 云台坐标系原点在相机坐标系下的位置，当前假设云台在相机正上方20厘米。
    cv::Vec3d camera_to_gimbal_translation_m_{0.0, -0.20, 0.0};
    // 无人机真实外接矩形尺寸，后续测量后更新。
    double drone_target_width_m_ = 0.7;
    double drone_target_height_m_ = 1.6;
    // 当前云台角，后续接入实时云台状态。
    double current_gimbal_yaw_rad_ = 2.485;
    double current_gimbal_pitch_rad_ = 3.304;
};

}  // namespace drone::drone_pnp

#endif  // DRONE_DETECT_DRONE_PNP_H
