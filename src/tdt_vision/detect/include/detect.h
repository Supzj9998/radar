#ifndef RADAR_DETECT_H
#define RADAR_DETECT_H

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include "classify.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "NvidiaInterface.hpp"
#include "opencv2/opencv.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "vision_interface/msg/detect_result.hpp"
#include "yolos.hpp"
#include "BaseInfer.hpp"
#include <fstream>
namespace tdt_radar {

class Detect final : public rclcpp::Node {
public:
    explicit Detect(const rclcpp::NodeOptions& options);
    void callback(const std::shared_ptr<sensor_msgs::msg::Image> msg);
    // 原始图像订阅者
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
    // 压缩图像订阅者
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr
        compressed_image_sub;

private:
    // 整车检测模型
    std::shared_ptr<Infer<yolo::BoxArray>>     yolo;
    // 装甲板检测模型
    std::shared_ptr<Infer<yolo::BoxArray>>     armor_yolo;
    // 类别编号
    std::shared_ptr<Infer<int>> classifier;
    // 发布器
    rclcpp::Publisher<vision_interface::msg::DetectResult>::SharedPtr pub;

    // 是否正在处理rosbag
    bool        if_rosbag = false;
    // 敌方颜色
    int         EnemyColor;
    int         debug;
    // 模型路径
    std::string yolo_path;
    std::string armor_path;
    std::string classify_path;
};
class Car {
public:
    // 车的矩形区域
    cv::Rect       car_rect;
    // yolo整车类
    yolo::Box      car;
    // yolo装甲板类
    yolo::BoxArray armors;
    // 车辆中心
    cv::Point2f    center;
    // ？？？
    cv::Rect       center_rect;
    // 编号
    int            number = 0;
    // 颜色
    int            color = 1;
};
}  // namespace tdt_radar

#endif
