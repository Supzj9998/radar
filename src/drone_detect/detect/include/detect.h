#ifndef DRONE_DETECT_DETECT_H
#define DRONE_DETECT_DETECT_H

#include <array>
#include <memory>
#include <string>
#include "BaseInfer.hpp"
#include "opencv2/core.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "yolos.hpp"

namespace drone::detect {

class Detect final : public rclcpp::Node {
public:
    // 构造函数
    explicit Detect(const rclcpp::NodeOptions& options);
    ~Detect() override;

private:
    // 接收图像后的回调
    void callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    // 给yolo的检测结果画框
    void drawDetections(cv::Mat&              image,
                        const yolo::BoxArray& detections) const;

    std::array<float, 9> processDetections(
        const yolo::BoxArray& detections, cv::Mat& image);
    // 将yolo的结果发布
    void publishDetections(const std::array<float, 9>& detection) const;

    // yolo推理器
    std::shared_ptr<tdt_radar::Infer<yolo::BoxArray>> yolo;
    // 图像订阅者
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
    // 图像发布者
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub;
    // 发布yolo检测结果数组
    // 每个目标占10个元素，类别，置信度，四个角点的坐标(left,top,right,top,right,bottom,left,bottom)
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr
        boxes_pub;

    // debug开关
    bool show_debug_image = true;
};
}  // namespace drone::detect

#endif
