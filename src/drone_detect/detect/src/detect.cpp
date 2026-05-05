#include "detect.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "cv_bridge/cv_bridge.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace drone::detect {

Detect::Detect(const rclcpp::NodeOptions& options)
    : Node("detect_node", options)
{
    // 读取参数和模型路径
    const auto engine_path = declare_parameter<std::string>(
        "engine_path", "model/TensorRT/best.engine");
    const auto model_path = declare_parameter<std::string>(
        "model_path", "model/ONNX/best.onnx");
    // 读取TensorRT构建engine使用的workspace大小
    const auto workspace_size_mb =
        declare_parameter<int>("trt_workspace_size_mb", 1024);

    // 指定yolo类型
    const auto yolo_type = ::yolo::Type::V11;
    RCLCPP_INFO(get_logger(), "Using YOLO decoder: %s",
                ::yolo::type_name(yolo_type));
    // 加载YOLO/TensorRT推理引擎
    yolo = ::yolo::load_or_build(
        engine_path, model_path, yolo_type,
        static_cast<float>(
            declare_parameter<double>("conf_threshold", 0.25)),
        static_cast<float>(
            declare_parameter<double>("nms_threshold", 0.45)),
        static_cast<size_t>(workspace_size_mb) * 1024ULL * 1024ULL);
    if (!yolo) {
        throw std::runtime_error("Failed to load yolo engine");
    }

    // yolo检测后的图像发布者
    image_pub = create_publisher<sensor_msgs::msg::Image>(
        declare_parameter<std::string>("output_image_topic",
                                       "detect/image_with_boxes"),
        10);
    // yolo检测框发布者
    boxes_pub = create_publisher<std_msgs::msg::Float32MultiArray>(
        declare_parameter<std::string>("output_boxes_topic",
                                       "detect/boxes"),
        10);
    // 原始图像订阅者
    image_sub = create_subscription<sensor_msgs::msg::Image>(
        declare_parameter<std::string>("image_topic", "image_raw"),
        rclcpp::SensorDataQoS(),
        std::bind(&Detect::callback, this, std::placeholders::_1));

    if (show_debug_image) {
        cv::namedWindow("yolo_debug", cv::WINDOW_NORMAL);
    }
}

Detect::~Detect()
{
    if (show_debug_image) {
        cv::destroyWindow("yolo_debug");
    }
}

void Detect::callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    // 把ros图像消息转换成Opencv图像
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

    // 构造推理输入
    tdt_radar::Image image(cv_ptr->image.data, cv_ptr->image.cols,
                           cv_ptr->image.rows);
    // yolo推理
    const auto detections = yolo->forward(image);
    // 处理检测结果
    const auto processed_detection =
        processDetections(detections, cv_ptr->image);
    // 发布检测框
    publishDetections(processed_detection);
    // 在图像上画框
    drawDetections(cv_ptr->image, detections);
    // OpenCV调试显示：每帧只刷新一次，避免多目标时重复阻塞。
    if (show_debug_image) {
        cv::imshow("yolo_debug", cv_ptr->image);
        auto key = cv::waitKey(1);
        if (key == 'q' || key == 'Q') {
            show_debug_image = false;
            cv::destroyWindow("yolo_debug");
        }
    }
    // 发布画框后的图像
    image_pub->publish(
        *cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8,
                            cv_ptr->image)
             .toImageMsg());
}

std::array<float, 9> Detect::processDetections(
    const yolo::BoxArray& detections, cv::Mat& image)
{
    std::array<float, 9> result{0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                0.0F, 0.0F, 0.0F, -1.0F};

    if (detections.empty()) {
        return result;
    }

    // 找出置信度大于0.6且最高的检测框
    const auto best = std::max_element(
        detections.begin(), detections.end(),
        [](const yolo::Box& lhs, const yolo::Box& rhs) {
            return lhs.confidence < rhs.confidence;
        });
    if (best->confidence <= 0.6F) {
        return result;
    }

    // 取最高置信度框在图像中的ROI区域
    const int left = std::clamp(static_cast<int>(best->left), 0, image.cols);
    const int top = std::clamp(static_cast<int>(best->top), 0, image.rows);
    const int right =
        std::clamp(static_cast<int>(best->right), 0, image.cols);
    const int bottom =
        std::clamp(static_cast<int>(best->bottom), 0, image.rows);
    const cv::Rect roi_rect(left, top, right - left, bottom - top);
    if (roi_rect.width <= 0 || roi_rect.height <= 0) {
        return result;
    }
    cv::Mat roi = image(roi_rect);

    // 对ROI做灰度化和二值化
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255,
                  cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 在二值图上查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return result;
    }

    contours.erase(
        std::remove_if(contours.begin(), contours.end(),
                       [](const std::vector<cv::Point>& contour) {
                           return contour.size() < 5U;
                       }),
        contours.end());
    if (contours.empty()) {
        return result;
    }

    // 取面积最大的8个轮廓，轮廓不足8个时保留全部
    std::sort(contours.begin(), contours.end(),
              [](const std::vector<cv::Point>& lhs,
                 const std::vector<cv::Point>& rhs) {
                  return cv::contourArea(lhs) > cv::contourArea(rhs);
              });
    if (contours.size() > 8U) {
        contours.resize(8U);
    }

    // 按轮廓中点x坐标最近原则配对
    std::vector<float> contour_center_x;
    contour_center_x.reserve(contours.size());
    for (const auto& contour : contours) {
        const cv::Rect rect = cv::boundingRect(contour);
        contour_center_x.push_back(
            static_cast<float>(rect.x) + static_cast<float>(rect.width) * 0.5F);
    }

    std::vector<size_t> nearest_index(contours.size(), contours.size());
    std::vector<float> nearest_x_diff(
        contours.size(), std::numeric_limits<float>::max());
    for (size_t i = 0; i < contours.size(); ++i) {
        for (size_t j = 0; j < contours.size(); ++j) {
            if (i == j) {
                continue;
            }
            const float x_diff =
                std::abs(contour_center_x[i] - contour_center_x[j]);
            if (x_diff < nearest_x_diff[i]) {
                nearest_x_diff[i] = x_diff;
                nearest_index[i] = j;
            }
        }
    }

    std::vector<bool> used(contours.size(), false);
    std::vector<std::pair<size_t, size_t>> contour_pairs;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (used[i] || nearest_index[i] == contours.size()) {
            continue;
        }

        const size_t nearest = nearest_index[i];
        if (used[nearest]) {
            used[i] = true;
            continue;
        }

        // 如果离A最近的B，已经有更近的C可以与B配对，则删除A。
        if (nearest_index[nearest] != i &&
            nearest_x_diff[nearest] < nearest_x_diff[i]) {
            used[i] = true;
            continue;
        }

        used[i] = true;
        used[nearest] = true;
        contour_pairs.emplace_back(i, nearest);
    }

    float pair_result = -1.0F;
    if (contour_pairs.size() == 3U) {
        pair_result = 3.0F;
    }
    else if (contour_pairs.size() == 4U) {
        pair_result = 4.0F;
    }
    if (pair_result < 0.0F) {
        return result;
    }

    struct PairInfo {
        std::array<cv::Point2f, 2> centers;
        float                      average_x{0.0F};
    };

    auto contourCenter = [](const std::vector<cv::Point>& contour) {
        const cv::Rect rect = cv::boundingRect(contour);
        return cv::Point2f(
            static_cast<float>(rect.x) + static_cast<float>(rect.width) * 0.5F,
            static_cast<float>(rect.y) +
                static_cast<float>(rect.height) * 0.5F);
    };

    std::vector<PairInfo> pair_infos;
    pair_infos.reserve(contour_pairs.size());
    for (const auto& pair : contour_pairs) {
        PairInfo info;
        info.centers[0] = contourCenter(contours[pair.first]);
        info.centers[1] = contourCenter(contours[pair.second]);
        info.average_x = (info.centers[0].x + info.centers[1].x) * 0.5F;
        pair_infos.push_back(info);
    }

    size_t left_pair = 0U;
    size_t right_pair = 1U;
    float  max_average_x_diff = -1.0F;
    for (size_t i = 0; i < pair_infos.size(); ++i) {
        for (size_t j = i + 1U; j < pair_infos.size(); ++j) {
            const float average_x_diff =
                std::abs(pair_infos[i].average_x - pair_infos[j].average_x);
            if (average_x_diff > max_average_x_diff) {
                max_average_x_diff = average_x_diff;
                left_pair = i;
                right_pair = j;
            }
        }
    }

    if (pair_infos[left_pair].average_x > pair_infos[right_pair].average_x) {
        std::swap(left_pair, right_pair);
    }
    auto left_points = pair_infos[left_pair].centers;
    auto right_points = pair_infos[right_pair].centers;
    std::sort(left_points.begin(), left_points.end(),
              [](const cv::Point2f& lhs, const cv::Point2f& rhs) {
                  return lhs.y < rhs.y;
              });
    std::sort(right_points.begin(), right_points.end(),
              [](const cv::Point2f& lhs, const cv::Point2f& rhs) {
                  return lhs.y < rhs.y;
              });

    const cv::Point2f roi_offset(static_cast<float>(roi_rect.x),
                                 static_cast<float>(roi_rect.y));
    std::array<cv::Point2f, 4> pnp_points{
        left_points[0] + roi_offset, right_points[0] + roi_offset,
        right_points[1] + roi_offset, left_points[1] + roi_offset};

    result[0] = pnp_points[0].x;
    result[1] = pnp_points[0].y;
    result[2] = pnp_points[1].x;
    result[3] = pnp_points[1].y;
    result[4] = pnp_points[2].x;
    result[5] = pnp_points[2].y;
    result[6] = pnp_points[3].x;
    result[7] = pnp_points[3].y;
    result[8] = pair_result;
    return result;
}

void Detect::publishDetections(const std::array<float, 9>& detection) const
{
    // 创建ros2消息
    std_msgs::msg::Float32MultiArray msg;
    // 提前预留容量
    msg.data.reserve(detection.size());
    // 传统视觉处理后的pnp点和配对结果转换ros2消息
    msg.data.insert(msg.data.end(), detection.begin(), detection.end());

    // 发布消息
    boxes_pub->publish(msg);
}

void Detect::drawDetections(cv::Mat&              image,
                            const yolo::BoxArray& detections) const
{
    // 遍历每一个检测框
    for (const auto& box : detections) {
        // 将浮点坐标转换成画图所需的整数坐标
        const cv::Point left_top(static_cast<int>(box.left),
                                 static_cast<int>(box.top));
        const cv::Point right_bottom(static_cast<int>(box.right),
                                     static_cast<int>(box.bottom));
        // 画绿色矩形
        cv::rectangle(image, left_top, right_bottom, cv::Scalar(0, 255, 0),
                      2);
        const std::string label = cv::format("%.2f", box.confidence);
        cv::putText(image, label, left_top, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);
    }
}

}  // namespace drone::detect

RCLCPP_COMPONENTS_REGISTER_NODE(drone::detect::Detect)
