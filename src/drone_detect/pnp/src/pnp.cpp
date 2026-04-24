#include "pnp.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "opencv2/calib3d.hpp"
#include "opencv2/core/persistence.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace drone::pnp {
namespace {
constexpr double kRadToDeg = 57.29577951308232;

std::string normalizeFrame(std::string frame_id) {
  frame_id.erase(std::remove(frame_id.begin(), frame_id.end(), ' '), frame_id.end());
  while (!frame_id.empty() && frame_id.front() == '/') {
    frame_id.erase(frame_id.begin());
  }
  return frame_id;
}

cv::Matx33d quatToMat(double x, double y, double z, double w) {
  const double norm = std::sqrt(x * x + y * y + z * z + w * w);
  if (norm <= 1e-12) {
    return cv::Matx33d::eye();
  }
  x /= norm;
  y /= norm;
  z /= norm;
  w /= norm;

  return cv::Matx33d(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
                     2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
                     2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
                     1.0 - 2.0 * (x * x + y * y));
}

cv::Matx33d rpyDegToMat(double roll_deg, double pitch_deg, double yaw_deg) {
  const double roll = roll_deg / kRadToDeg;
  const double pitch = pitch_deg / kRadToDeg;
  const double yaw = yaw_deg / kRadToDeg;

  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);

  const cv::Matx33d rx(1.0, 0.0, 0.0,
                       0.0, cr, -sr,
                       0.0, sr, cr);
  const cv::Matx33d ry(cp, 0.0, sp,
                       0.0, 1.0, 0.0,
                       -sp, 0.0, cp);
  const cv::Matx33d rz(cy, -sy, 0.0,
                       sy, cy, 0.0,
                       0.0, 0.0, 1.0);
  return rz * ry * rx;
}
}  // namespace

PnpNode::PnpNode(const rclcpp::NodeOptions& options) : Node("pnp_node", options) {
  boxes_topic_ = this->declare_parameter<std::string>("boxes_topic", "detect/boxes");
  output_topic_ = this->declare_parameter<std::string>("output_topic", "pnp/polar");
  autoaim_topic_ = this->declare_parameter<std::string>("autoaim_topic", "/autoaim/target");
  camera_info_topic_ = this->declare_parameter<std::string>("camera_info_topic", "camera_info");
  extrinsic_topic_ = this->declare_parameter<std::string>("extrinsic_topic", "camera/extrinsic");
  tf_topic_ = this->declare_parameter<std::string>("tf_topic", "/tf");
  world_extrinsic_file_ =
      this->declare_parameter<std::string>("world_extrinsic_file", "config/out_matrix.yaml");
  camera_frame_id_ = normalizeFrame(
      this->declare_parameter<std::string>("camera_frame_id", "camera_optical_frame"));
  laser_frame_id_ =
      normalizeFrame(this->declare_parameter<std::string>("laser_frame_id", "laser_link"));

  input_stride_ = this->declare_parameter<int>("input_stride", 10);
  target_class_id_ = this->declare_parameter<int>("target_class_id", -1);
  use_camera_info_ = this->declare_parameter<bool>("use_camera_info", true);
  use_static_laser_extrinsic_ =
      this->declare_parameter<bool>("use_static_laser_extrinsic", true);
  use_extrinsic_topic_ = this->declare_parameter<bool>("use_extrinsic_topic", false);
  use_tf_topic_ = this->declare_parameter<bool>("use_tf_topic", false);
  require_extrinsic_ = this->declare_parameter<bool>("require_extrinsic", false);
  use_world_extrinsic_file_ = this->declare_parameter<bool>("use_world_extrinsic_file", false);
  publish_autoaim_ = this->declare_parameter<bool>("publish_autoaim", true);
  allow_shoot_ = this->declare_parameter<bool>("allow_shoot", false);
  autoaim_target_id_ = this->declare_parameter<int>("autoaim_target_id", 0);
  autoaim_vision_mode_ =
      this->declare_parameter<int>("autoaim_vision_mode", gary_msgs::msg::AutoAIM::VISION_MODE_ARMOR);
  output_in_degrees_ = this->declare_parameter<bool>("output_in_degrees", true);
  input_is_undistorted_ = this->declare_parameter<bool>("input_is_undistorted", true);
  target_width_m_ = this->declare_parameter<double>("target_width_m", 0.072);
  target_height_m_ = this->declare_parameter<double>("target_height_m", 0.050);
  laser_translation_m_ = this->declare_parameter<std::vector<double>>(
      "laser_translation_m", {0.0, 0.0, 0.0});
  laser_rpy_deg_ = this->declare_parameter<std::vector<double>>(
      "laser_rpy_deg", {0.0, 0.0, 0.0});

  auto camera_k = this->declare_parameter<std::vector<double>>(
      "camera_matrix", {1280.0, 0.0, 512.0, 0.0, 1280.0, 384.0, 0.0, 0.0, 1.0});
  auto dist = this->declare_parameter<std::vector<double>>("dist_coeffs", {0.0, 0.0, 0.0, 0.0, 0.0});

  if (camera_k.size() != 9) {
    RCLCPP_WARN(this->get_logger(),
                "camera_matrix size is %zu, expected 9. Fallback to identity-like defaults.",
                camera_k.size());
    camera_k = {1280.0, 0.0, 512.0, 0.0, 1280.0, 384.0, 0.0, 0.0, 1.0};
  }

  camera_matrix_ = cv::Mat(3, 3, CV_64F);
  for (int i = 0; i < 9; ++i) {
    camera_matrix_.at<double>(i / 3, i % 3) = camera_k[static_cast<size_t>(i)];
  }

  dist_coeffs_ = cv::Mat::zeros(static_cast<int>(dist.size()), 1, CV_64F);
  if (!input_is_undistorted_) {
    for (size_t i = 0; i < dist.size(); ++i) {
      dist_coeffs_.at<double>(static_cast<int>(i), 0) = dist[i];
    }
  }

  polar_pub_ = this->create_publisher<base_interface::msg::Polar3f>(output_topic_, 10);
  if (publish_autoaim_) {
    autoaim_pub_ = this->create_publisher<gary_msgs::msg::AutoAIM>(autoaim_topic_, 10);
  }
  boxes_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
      boxes_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PnpNode::boxesCallback, this, std::placeholders::_1));
  if (use_camera_info_) {
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, rclcpp::SensorDataQoS(),
        std::bind(&PnpNode::cameraInfoCallback, this, std::placeholders::_1));
  }
  if (use_extrinsic_topic_) {
    extrinsic_sub_ = this->create_subscription<geometry_msgs::msg::TransformStamped>(
        extrinsic_topic_, 10, std::bind(&PnpNode::extrinsicCallback, this, std::placeholders::_1));
  }
  if (use_tf_topic_) {
    tf_sub_ = this->create_subscription<tf2_msgs::msg::TFMessage>(
        tf_topic_, 10, std::bind(&PnpNode::tfCallback, this, std::placeholders::_1));
  }
  if (use_static_laser_extrinsic_) {
    if (loadStaticLaserExtrinsic()) {
      RCLCPP_INFO(this->get_logger(), "Loaded static camera->laser extrinsic from parameters.");
    } else {
      RCLCPP_ERROR(this->get_logger(),
                   "Invalid static camera->laser extrinsic parameters, falling back to identity.");
      std::lock_guard<std::mutex> lock(param_mutex_);
      r_laser_camera_ = cv::Matx33d::eye();
      t_laser_camera_ = cv::Vec3d(0.0, 0.0, 0.0);
      extrinsic_ready_ = true;
    }
  }
  if (use_world_extrinsic_file_) {
    if (loadWorldExtrinsicFromFile(world_extrinsic_file_)) {
      RCLCPP_INFO(this->get_logger(), "Loaded static world extrinsic: %s",
                  world_extrinsic_file_.c_str());
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to load static world extrinsic: %s",
                   world_extrinsic_file_.c_str());
    }
  }

  RCLCPP_INFO(this->get_logger(),
              "PnP node ready. boxes=%s output=%s use_world_extrinsic_file=%s "
              "use_static_laser_extrinsic=%s use_extrinsic_topic=%s use_tf_topic=%s "
              "require_extrinsic=%s input_is_undistorted=%s",
              boxes_topic_.c_str(), output_topic_.c_str(),
              use_world_extrinsic_file_ ? "true" : "false",
              use_static_laser_extrinsic_ ? "true" : "false",
              use_extrinsic_topic_ ? "true" : "false", use_tf_topic_ ? "true" : "false",
              require_extrinsic_ ? "true" : "false",
              input_is_undistorted_ ? "true" : "false");
}

void PnpNode::boxesCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
  std::vector<BoxDetection> boxes;
  if (!parseBoxes(msg, boxes) || boxes.empty()) {
    return;
  }

  const BoxDetection* best = nullptr;
  float best_score = -std::numeric_limits<float>::infinity();
  for (const auto& box : boxes) {
    if (target_class_id_ >= 0 && box.class_id != target_class_id_) {
      continue;
    }
    if (box.score > best_score) {
      best_score = box.score;
      best = &box;
    }
  }

  if (best == nullptr) {
    return;
  }

  cv::Vec3d tvec;
  if (!solveFromBox(*best, tvec)) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "solvePnP failed for current detection.");
    return;
  }

  cv::Vec3d output_vec = tvec;
  if (require_extrinsic_) {
    std::lock_guard<std::mutex> lock(param_mutex_);
    if (!extrinsic_ready_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Waiting for extrinsic from configured source.");
      return;
    }
  }
  if (extrinsic_ready_) {
    output_vec = use_world_extrinsic_file_ ? cameraToWorld(tvec) : cameraDirectionToLaser(tvec);
  }

  polar_pub_->publish(toPolarMessage(output_vec));
  if (publish_autoaim_ && autoaim_pub_) {
    autoaim_pub_->publish(toAutoAimMessage(output_vec));
  }
}

void PnpNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(param_mutex_);
  camera_matrix_ = cv::Mat(3, 3, CV_64F);
  for (int i = 0; i < 9; ++i) {
    camera_matrix_.at<double>(i / 3, i % 3) = msg->k[static_cast<size_t>(i)];
  }

  dist_coeffs_ = cv::Mat::zeros(static_cast<int>(msg->d.size()), 1, CV_64F);
  if (!input_is_undistorted_) {
    for (size_t i = 0; i < msg->d.size(); ++i) {
      dist_coeffs_.at<double>(static_cast<int>(i), 0) = msg->d[i];
    }
  }
}

void PnpNode::extrinsicCallback(const geometry_msgs::msg::TransformStamped::SharedPtr msg) {
  if (updateExtrinsicFromTransform(*msg)) {
    RCLCPP_INFO_ONCE(this->get_logger(), "Received extrinsic from topic '%s'.",
                     extrinsic_topic_.c_str());
  }
}

void PnpNode::tfCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
  for (const auto& tf_msg : msg->transforms) {
    if (updateExtrinsicFromTransform(tf_msg)) {
      RCLCPP_INFO_ONCE(this->get_logger(), "Received extrinsic from tf topic '%s'.",
                       tf_topic_.c_str());
      return;
    }
  }
}

bool PnpNode::parseBoxes(const std_msgs::msg::Float32MultiArray::SharedPtr& msg,
                         std::vector<BoxDetection>& boxes) const {
  const size_t kStride = static_cast<size_t>(input_stride_);
  if (kStride != 6U && kStride != 10U) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Unsupported input_stride=%zu, expected 6 or 10.", kStride);
    return false;
  }
  if (msg->data.empty()) {
    return false;
  }
  if (msg->data.size() % kStride != 0U) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Invalid boxes message length: %zu (must be divisible by 6).",
                         msg->data.size());
    return false;
  }

  boxes.clear();
  boxes.reserve(msg->data.size() / kStride);
  for (size_t i = 0; i < msg->data.size(); i += kStride) {
    BoxDetection box{};
    box.class_id = static_cast<int>(msg->data[i]);
    box.score = msg->data[i + 1];
    if (kStride == 10U) {
      box.has_corners = true;
      box.corners = {
          cv::Point2f(msg->data[i + 2], msg->data[i + 3]),
          cv::Point2f(msg->data[i + 4], msg->data[i + 5]),
          cv::Point2f(msg->data[i + 6], msg->data[i + 7]),
          cv::Point2f(msg->data[i + 8], msg->data[i + 9]),
      };
    } else {
      box.x = msg->data[i + 2];
      box.y = msg->data[i + 3];
      box.w = msg->data[i + 4];
      box.h = msg->data[i + 5];
      if (box.w <= 1.0F || box.h <= 1.0F) {
        continue;
      }
    }
    boxes.push_back(box);
  }
  return true;
}

bool PnpNode::solveFromBox(const BoxDetection& box, cv::Vec3d& tvec) const {
  const double half_w = target_width_m_ * 0.5;
  const double half_h = target_height_m_ * 0.5;

  std::vector<cv::Point3f> object_points{
      cv::Point3f(-half_w, -half_h, 0.0),
      cv::Point3f(half_w, -half_h, 0.0),
      cv::Point3f(half_w, half_h, 0.0),
      cv::Point3f(-half_w, half_h, 0.0),
  };

  std::vector<cv::Point2f> image_points;
  image_points.reserve(4);
  if (box.has_corners) {
    image_points.push_back(box.corners[0]);
    image_points.push_back(box.corners[1]);
    image_points.push_back(box.corners[2]);
    image_points.push_back(box.corners[3]);
  } else {
    const float x = box.x;
    const float y = box.y;
    const float w = box.w;
    const float h = box.h;
    image_points.push_back(cv::Point2f(x, y));
    image_points.push_back(cv::Point2f(x + w, y));
    image_points.push_back(cv::Point2f(x + w, y + h));
    image_points.push_back(cv::Point2f(x, y + h));
  }

  cv::Vec3d rvec;
  cv::Mat camera_matrix;
  cv::Mat dist_coeffs;
  {
    std::lock_guard<std::mutex> lock(param_mutex_);
    camera_matrix = camera_matrix_.clone();
    dist_coeffs = dist_coeffs_.clone();
  }

  return cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec, false,
                      cv::SOLVEPNP_ITERATIVE);
}

bool PnpNode::updateExtrinsicFromTransform(const geometry_msgs::msg::TransformStamped& tf_msg) {
  const std::string parent = normalizeFrame(tf_msg.header.frame_id);
  const std::string child = normalizeFrame(tf_msg.child_frame_id);

  if (!((parent == laser_frame_id_ && child == camera_frame_id_) ||
        (parent == camera_frame_id_ && child == laser_frame_id_))) {
    return false;
  }

  const auto& t = tf_msg.transform.translation;
  const auto& q = tf_msg.transform.rotation;
  const cv::Matx33d r_parent_child = quatToMat(q.x, q.y, q.z, q.w);
  const cv::Vec3d t_parent_child(t.x, t.y, t.z);

  std::lock_guard<std::mutex> lock(param_mutex_);
  if (parent == camera_frame_id_ && child == laser_frame_id_) {
    // TF is camera->laser. Convert point with p_l = R_l_c * p_c + t_l_c.
    r_laser_camera_ = r_parent_child;
    t_laser_camera_ = t_parent_child;
  } else {
    // TF is laser->camera. Invert to camera->laser for PnP output conversion.
    r_laser_camera_ = r_parent_child.t();
    t_laser_camera_ = -(r_laser_camera_ * t_parent_child);
  }
  extrinsic_ready_ = true;
  return true;
}

bool PnpNode::loadStaticLaserExtrinsic() {
  if (laser_translation_m_.size() != 3U || laser_rpy_deg_.size() != 3U) {
    return false;
  }

  std::lock_guard<std::mutex> lock(param_mutex_);
  r_laser_camera_ = rpyDegToMat(laser_rpy_deg_[0], laser_rpy_deg_[1], laser_rpy_deg_[2]);
  t_laser_camera_ =
      cv::Vec3d(laser_translation_m_[0], laser_translation_m_[1], laser_translation_m_[2]);
  extrinsic_ready_ = true;
  return true;
}

bool PnpNode::loadWorldExtrinsicFromFile(const std::string& yaml_path) {
  cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }

  cv::Mat world_rvec;
  cv::Mat world_tvec;
  fs["world_rvec"] >> world_rvec;
  fs["world_tvec"] >> world_tvec;
  fs.release();

  if (world_rvec.empty() || world_tvec.empty()) {
    return false;
  }

  cv::Mat r_cw;
  cv::Rodrigues(world_rvec, r_cw);  // Xc = Rcw * Xw + tcw
  cv::Mat t_cw = world_tvec.reshape(1, 3);

  cv::Mat r_wc = r_cw.t();          // Xw = Rwc * Xc + twc
  cv::Mat t_wc = -r_wc * t_cw;      // twc = -Rwc * tcw

  std::lock_guard<std::mutex> lock(param_mutex_);
  r_world_camera_ = cv::Matx33d(
      r_wc.at<double>(0, 0), r_wc.at<double>(0, 1), r_wc.at<double>(0, 2),
      r_wc.at<double>(1, 0), r_wc.at<double>(1, 1), r_wc.at<double>(1, 2),
      r_wc.at<double>(2, 0), r_wc.at<double>(2, 1), r_wc.at<double>(2, 2));
  t_world_camera_ = cv::Vec3d(t_wc.at<double>(0, 0), t_wc.at<double>(1, 0), t_wc.at<double>(2, 0));
  extrinsic_ready_ = true;
  return true;
}

cv::Vec3d PnpNode::cameraToWorld(const cv::Vec3d& p_camera) const {
  std::lock_guard<std::mutex> lock(param_mutex_);
  return r_world_camera_ * p_camera + t_world_camera_;
}

cv::Vec3d PnpNode::cameraDirectionToLaser(const cv::Vec3d& p_camera) const {
  std::lock_guard<std::mutex> lock(param_mutex_);
  // For camera-center aiming, only the camera-to-actuator rotation matters.
  // Translational offset would instead model a laser/emitter origin, which is not the desired behavior here.
  return r_laser_camera_ * p_camera;
}

cv::Vec3d PnpNode::cameraToLaser(const cv::Vec3d& p_camera) const {
  std::lock_guard<std::mutex> lock(param_mutex_);
  return r_laser_camera_ * p_camera + t_laser_camera_;
}

base_interface::msg::Polar3f PnpNode::toPolarMessage(const cv::Vec3d& tvec) const {
  const double x = tvec[0];
  const double y = tvec[1];
  const double z = tvec[2];

  const double yaw = std::atan2(x, z);
  const double pitch = std::atan2(-y, std::sqrt(x * x + z * z));
  const double distance = std::sqrt(x * x + y * y + z * z);

  base_interface::msg::Polar3f msg;
  if (output_in_degrees_) {
    msg.yaw = static_cast<float>(yaw * kRadToDeg);
    msg.pitch = static_cast<float>(pitch * kRadToDeg);
  } else {
    msg.yaw = static_cast<float>(yaw);
    msg.pitch = static_cast<float>(pitch);
  }
  msg.distance = static_cast<float>(distance);
  return msg;
}

gary_msgs::msg::AutoAIM PnpNode::toAutoAimMessage(const cv::Vec3d& tvec) const {
  const double x = tvec[0];
  const double y = tvec[1];
  const double z = tvec[2];
  const double yaw = std::atan2(x, z);
  const double pitch = std::atan2(-y, std::sqrt(x * x + z * z));
  const double distance = std::sqrt(x * x + y * y + z * z);

  gary_msgs::msg::AutoAIM msg;
  msg.header.stamp = this->get_clock()->now();
  msg.yaw = static_cast<float>(yaw);
  msg.pitch = static_cast<float>(pitch);
  msg.target_id = static_cast<uint8_t>(std::clamp(autoaim_target_id_, 0, 7));
  msg.target_distance = static_cast<float>(distance);
  msg.vision_mode = static_cast<uint8_t>(std::clamp(autoaim_vision_mode_, 1, 4));
  msg.shoot_command = allow_shoot_ ? gary_msgs::msg::AutoAIM::ALLOW_SHOOT
                                   : gary_msgs::msg::AutoAIM::CEASE_FIRE;
  return msg;
}

}  // namespace drone::pnp

RCLCPP_COMPONENTS_REGISTER_NODE(drone::pnp::PnpNode)
