#include "drone_pnp.h"

#include <cmath>
#include <functional>
#include <vector>

#include "opencv2/calib3d.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace drone::drone_pnp {
namespace {
    constexpr const char* kDroneBoxesTopic = "drone_detecter/boxes";
    constexpr const char* kCameraInfoTopic = "bottom/camera_info";
    constexpr const char* kAutoaimTopic = "/autoaim/drone";
    constexpr size_t      kBoxStride = 6U;
}  // namespace

DronePnpNode::DronePnpNode(const rclcpp::NodeOptions& options)
    : Node("drone_pnp_node", options)
{
    camera_to_gimbal_translation_m_[0] =
        declare_parameter<double>("camera_to_gimbal_x_m", 0.0);
    camera_to_gimbal_translation_m_[1] =
        declare_parameter<double>("camera_to_gimbal_y_m", -0.20);
    camera_to_gimbal_translation_m_[2] =
        declare_parameter<double>("camera_to_gimbal_z_m", 0.0);
    drone_target_width_m_ =
        declare_parameter<double>("drone_target_width_m", 0.0);
    drone_target_height_m_ =
        declare_parameter<double>("drone_target_height_m", 0.0);
    current_gimbal_yaw_rad_ =
        declare_parameter<double>("current_gimbal_yaw_rad", 0.0);
    current_gimbal_pitch_rad_ =
        declare_parameter<double>("current_gimbal_pitch_rad", 0.0);

    boxes_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        kDroneBoxesTopic, rclcpp::SensorDataQoS(),
        std::bind(&DronePnpNode::boxesCallback, this,
                  std::placeholders::_1));
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        kCameraInfoTopic, rclcpp::SensorDataQoS(),
        std::bind(&DronePnpNode::cameraInfoCallback, this,
                  std::placeholders::_1));
    autoaim_pub_ =
        create_publisher<gary_msgs::msg::AutoAIM>(kAutoaimTopic, 10);
}

void DronePnpNode::boxesCallback(
    const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    cv::Vec3d camera_tvec;
    if (!solveBox(*msg, camera_tvec)) {
        return;
    }

    const cv::Vec3d gimbal_tvec = cameraToGimbal(camera_tvec);
    double          yaw = 0.0;
    double          pitch = 0.0;
    calculateAngleOffset(gimbal_tvec, yaw, pitch);
    publishAngleOffset(yaw, pitch, gimbal_tvec);
}

void DronePnpNode::cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    if (msg->k[0] <= 0.0 || msg->k[4] <= 0.0) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    camera_matrix_ = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) {
        camera_matrix_.at<double>(i / 3, i % 3) =
            msg->k[static_cast<size_t>(i)];
    }

    dist_coeffs_ =
        cv::Mat::zeros(static_cast<int>(msg->d.size()), 1, CV_64F);
    for (size_t i = 0; i < msg->d.size(); ++i) {
        dist_coeffs_.at<double>(static_cast<int>(i), 0) = msg->d[i];
    }
    camera_info_ready_ = true;
}

bool DronePnpNode::solveBox(const std_msgs::msg::Float32MultiArray& msg,
                            cv::Vec3d& camera_tvec) const
{
    if (msg.data.size() != kBoxStride) {
        return false;
    }

    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!camera_info_ready_ || camera_matrix_.empty()) {
            return false;
        }
        camera_matrix = camera_matrix_.clone();
        dist_coeffs = dist_coeffs_.clone();
    }

    const float left = msg.data[2];
    const float top = msg.data[3];
    const float right = msg.data[4];
    const float bottom = msg.data[5];
    if (right <= left || bottom <= top) {
        return false;
    }

    const float half_w = static_cast<float>(drone_target_width_m_ * 0.5);
    const float half_h = static_cast<float>(drone_target_height_m_ * 0.5);
    const std::vector<cv::Point3f> object_points{{-half_w, -half_h, 0.0F},
                                                 {half_w, -half_h, 0.0F},
                                                 {half_w, half_h, 0.0F},
                                                 {-half_w, half_h, 0.0F}};
    const std::vector<cv::Point2f> image_points{
        {left, top}, {right, top}, {right, bottom}, {left, bottom}};

    cv::Vec3d rvec;
    return cv::solvePnP(object_points, image_points, camera_matrix,
                        dist_coeffs, rvec, camera_tvec, false,
                        cv::SOLVEPNP_ITERATIVE);
}

cv::Vec3d DronePnpNode::cameraToGimbal(const cv::Vec3d& camera_tvec) const
{
    return camera_tvec - camera_to_gimbal_translation_m_;
}

void DronePnpNode::calculateAngleOffset(const cv::Vec3d& gimbal_tvec,
                                        double& yaw, double& pitch) const
{
    const double x = gimbal_tvec[0];
    const double y = gimbal_tvec[1];
    const double z = gimbal_tvec[2];
    yaw = -std::atan2(x, z);
    pitch = -std::atan2(-y, std::sqrt(x * x + z * z));
}

void DronePnpNode::publishAngleOffset(double yaw, double pitch,
                                      const cv::Vec3d& gimbal_tvec) const
{
    const double x = gimbal_tvec[0];
    const double y = gimbal_tvec[1];
    const double z = gimbal_tvec[2];

    gary_msgs::msg::AutoAIM msg;
    msg.header.stamp = get_clock()->now();
    msg.yaw = static_cast<float>(yaw + current_gimbal_yaw_rad_);
    msg.pitch = static_cast<float>(pitch + current_gimbal_pitch_rad_);
    msg.target_distance =
        static_cast<float>(std::sqrt(x * x + y * y + z * z));
    msg.target_id = 0;
    msg.vision_mode = gary_msgs::msg::AutoAIM::VISION_MODE_ARMOR;
    msg.shoot_command = gary_msgs::msg::AutoAIM::CEASE_FIRE;
    autoaim_pub_->publish(msg);
}

}  // namespace drone::drone_pnp

RCLCPP_COMPONENTS_REGISTER_NODE(drone::drone_pnp::DronePnpNode)
