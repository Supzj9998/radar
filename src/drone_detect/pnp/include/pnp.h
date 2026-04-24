#ifndef DRONE_PNP__PNP_H_
#define DRONE_PNP__PNP_H_

#include <array>
#include <mutex>
#include <string>
#include <vector>

#include "base_interface/msg/polar3f.hpp"
#include "gary_msgs/msg/auto_aim.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "opencv2/core.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "tf2_msgs/msg/tf_message.hpp"

namespace drone::pnp {

struct BoxDetection {
  int class_id;
  float score;
  float x;
  float y;
  float w;
  float h;
  bool has_corners{false};
  std::array<cv::Point2f, 4> corners{};
};

class PnpNode : public rclcpp::Node {
 public:
  explicit PnpNode(const rclcpp::NodeOptions& options);

 private:
  void boxesCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void extrinsicCallback(const geometry_msgs::msg::TransformStamped::SharedPtr msg);
  void tfCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
  bool parseBoxes(const std_msgs::msg::Float32MultiArray::SharedPtr& msg,
                  std::vector<BoxDetection>& boxes) const;
  bool solveFromBox(const BoxDetection& box, cv::Vec3d& tvec) const;
  bool loadWorldExtrinsicFromFile(const std::string& yaml_path);
  bool loadStaticLaserExtrinsic();
  bool updateExtrinsicFromTransform(const geometry_msgs::msg::TransformStamped& tf_msg);
  cv::Vec3d cameraToWorld(const cv::Vec3d& p_camera) const;
  cv::Vec3d cameraDirectionToLaser(const cv::Vec3d& p_camera) const;
  cv::Vec3d cameraToLaser(const cv::Vec3d& p_camera) const;
  base_interface::msg::Polar3f toPolarMessage(const cv::Vec3d& tvec) const;
  gary_msgs::msg::AutoAIM toAutoAimMessage(const cv::Vec3d& tvec) const;

  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr boxes_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr extrinsic_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_;
  rclcpp::Publisher<base_interface::msg::Polar3f>::SharedPtr polar_pub_;
  rclcpp::Publisher<gary_msgs::msg::AutoAIM>::SharedPtr autoaim_pub_;

  std::string boxes_topic_;
  std::string output_topic_;
  std::string autoaim_topic_;
  std::string camera_info_topic_;
  std::string extrinsic_topic_;
  std::string tf_topic_;
  std::string world_extrinsic_file_;
  std::string camera_frame_id_;
  std::string laser_frame_id_;
  int input_stride_{10};
  int target_class_id_{-1};
  bool use_camera_info_{true};
  bool use_static_laser_extrinsic_{true};
  bool use_extrinsic_topic_{false};
  bool use_tf_topic_{false};
  bool require_extrinsic_{false};
  bool use_world_extrinsic_file_{false};
  bool publish_autoaim_{true};
  bool allow_shoot_{false};
  int autoaim_target_id_{0};
  int autoaim_vision_mode_{1};
  bool output_in_degrees_{true};
  bool input_is_undistorted_{true};
  double target_width_m_{0.072};
  double target_height_m_{0.050};
  std::vector<double> laser_translation_m_{0.0, 0.0, 0.0};
  std::vector<double> laser_rpy_deg_{0.0, 0.0, 0.0};

  mutable std::mutex param_mutex_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  cv::Matx33d r_laser_camera_{cv::Matx33d::eye()};
  cv::Vec3d t_laser_camera_{0.0, 0.0, 0.0};
  cv::Matx33d r_world_camera_{cv::Matx33d::eye()};
  cv::Vec3d t_world_camera_{0.0, 0.0, 0.0};
  bool extrinsic_ready_{false};
};

}  // namespace drone::pnp

#endif  // DRONE_PNP__PNP_H_
