#include "drone_detecter.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>

#include "cv_bridge/cv_bridge.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace drone::drone_detecter {

DroneDetecterNode::DroneDetecterNode(const rclcpp::NodeOptions& options)
    : Node("drone_detecter_node", options)
{
    const auto engine_path =
        declare_parameter<std::string>("engine_path",
                                       "model/TensorRT/drone.engine");
    const auto model_path =
        declare_parameter<std::string>("model_path", "model/ONNX/drone.onnx");
    const auto workspace_size_mb =
        declare_parameter<int>("trt_workspace_size_mb", 1024);

    if (!engine_path.empty() && !model_path.empty()) {
        const auto yolo_type = ::yolo::Type::V11;
        RCLCPP_INFO(get_logger(), "Using YOLO decoder: %s",
                    ::yolo::type_name(yolo_type));
        yolo_ = ::yolo::load_or_build(
            engine_path, model_path, yolo_type,
            static_cast<float>(
                declare_parameter<double>("conf_threshold", 0.25)),
            static_cast<float>(
                declare_parameter<double>("nms_threshold", 0.45)),
            static_cast<size_t>(workspace_size_mb) * 1024ULL * 1024ULL);
        if (!yolo_) {
            throw std::runtime_error("Failed to load drone yolo engine");
        }
    }
    else {
        RCLCPP_WARN(get_logger(),
                    "drone_detecter yolo model paths are empty, inference is disabled");
    }

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "camera_image", rclcpp::SensorDataQoS(),
        std::bind(&DroneDetecterNode::imageCallback, this,
                  std::placeholders::_1));
    boxes_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
        "drone_detecter/boxes", 10);
}

void DroneDetecterNode::imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    if (!yolo_) {
        return;
    }

    // ros2消息转opencv矩阵
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr =
            cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                              "cv_bridge conversion failed: %s", e.what());
        return;
    }

    tdt_radar::Image image(cv_ptr->image.data, cv_ptr->image.cols,
                           cv_ptr->image.rows);
    // yolo推理
    const auto detections = yolo_->forward(image);
    // 对所有框筛选颜色和置信度
    const auto filtered_detections =
        filterDetectionsByColor(cv_ptr->image, detections);
    if (filtered_detections.empty()) {
        return;
    }
    publishDetections(filtered_detections);
}

yolo::BoxArray DroneDetecterNode::filterDetectionsByColor(
    const cv::Mat& image, const yolo::BoxArray& detections) const
{
    auto clampBox = [&image](const yolo::Box& box) {
        const int left = std::clamp(static_cast<int>(box.left), 0, image.cols);
        const int top = std::clamp(static_cast<int>(box.top), 0, image.rows);
        const int right =
            std::clamp(static_cast<int>(box.right), 0, image.cols);
        const int bottom =
            std::clamp(static_cast<int>(box.bottom), 0, image.rows);
        return cv::Rect(left, top, right - left, bottom - top);
    };

    bool      has_red = false;
    bool      has_blue = false;
    yolo::Box best_red;
    yolo::Box best_blue;

    for (const auto& box : detections) {
        if (box.confidence < 0.5F) {
            continue;
        }

        const cv::Rect roi_rect = clampBox(box);
        if (roi_rect.width <= 0 || roi_rect.height <= 0) {
            continue;
        }

        const cv::Scalar mean_color = cv::mean(image(roi_rect));
        const bool       is_blue = mean_color[0] > mean_color[2];
        if (is_blue) {
            if (!has_blue || box.confidence > best_blue.confidence) {
                best_blue = box;
                has_blue = true;
            }
        }
        else {
            if (!has_red || box.confidence > best_red.confidence) {
                best_red = box;
                has_red = true;
            }
        }
    }

    yolo::BoxArray filtered;
    if (detect_blue_target_) {
        if (has_blue) {
            filtered.push_back(best_blue);
        }
    }
    else if (has_red) {
        filtered.push_back(best_red);
    }
    return filtered;
}

void DroneDetecterNode::publishDetections(
    const yolo::BoxArray& detections) const
{
    std_msgs::msg::Float32MultiArray msg;
    msg.data.reserve(detections.size() * 6U);
    for (const auto& box : detections) {
        msg.data.insert(msg.data.end(),
                        {static_cast<float>(box.class_label),
                         box.confidence, box.left, box.top, box.right,
                         box.bottom});
    }
    boxes_pub_->publish(msg);
}

}  // namespace drone::drone_detecter

RCLCPP_COMPONENTS_REGISTER_NODE(drone::drone_detecter::DroneDetecterNode)
