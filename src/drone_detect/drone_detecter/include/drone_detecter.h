#ifndef DRONE_DETECT_DRONE_DETECTER_H
#define DRONE_DETECT_DRONE_DETECTER_H

#include <memory>

#include "BaseInfer.hpp"
#include "opencv2/core.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "yolos.hpp"

namespace drone::drone_detecter {

class DroneDetecterNode final : public rclcpp::Node {
public:
    explicit DroneDetecterNode(const rclcpp::NodeOptions& options);

private:
    // 图像回调
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    // 颜色判断
    yolo::BoxArray
    filterDetectionsByColor(const cv::Mat&        image,
                            const yolo::BoxArray& detections) const;
    // 发布筛选后的无人机yolo框
    void publishDetections(const yolo::BoxArray& detections) const;

    // yolo推理器成员
    std::shared_ptr<tdt_radar::Infer<yolo::BoxArray>> yolo_;
    // 图像订阅器
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    // 筛选后的无人机检测框发布器
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr
        boxes_pub_;

    // true表示当前需要蓝色目标，false表示当前需要红色目标。
    bool detect_blue_target_ = true;
};

}  // namespace drone::drone_detecter

#endif  // DRONE_DETECT_DRONE_DETECTER_H
