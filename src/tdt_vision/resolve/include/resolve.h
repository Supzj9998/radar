#ifndef RADAR_RESOLVE_H
#define RADAR_RESOLVE_H

#include <memory>
#include <geometry_msgs/msg/detail/point__struct.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_interface/msg/match_info.hpp>
#include "cv_bridge/cv_bridge.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "opencv2/opencv.hpp"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "radar_utils.h"
#include "vision_interface/msg/detect_result.hpp"
#include "vision_interface/msg/radar2_sentry.hpp"
namespace tdt_radar {
//创建节点类Resolve final表示此类不能再被继承
class Resolve final : public rclcpp::Node {
public:
    //构造函数 初始化节点
    explicit Resolve(const rclcpp::NodeOptions& options);
    //将收到的二维坐标转换为地图坐标并发布成点云
    void callback(const std::shared_ptr<geometry_msgs::msg::Vector3> msg);
    //处理视觉识别模块的检测结果 把视觉检测到的“蓝方”和“红方”坐标转换到地图坐标，并绘制在小地图上，同时生成点云并发布
    void DetectCallback(
        const vision_interface::msg::DetectResult::SharedPtr msg);
    //回调函数处理比赛信息
    void MatchInfoCallback(
        const vision_interface::msg::MatchInfo::SharedPtr msg);
    //声明订阅者
    //接收三维点
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr point_sub;
    //接收视觉检测结果
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        detect_sub;
    //接收比赛状态信息
    rclcpp::Subscription<vision_interface::msg::MatchInfo>::SharedPtr
            match_info_sub;
    //声明parser类指针
    parser* parser_;
    //声明变量代表敌方颜色
    int     EnemyColor = 1;
    //声明cv:Mat类小地图
    cv::Mat minimap;
    //标志点信息
    int     markers[6];
    //当前比赛的时间
    int16_t match_time;
    //16个机器人的血量
    uint8_t robot_hp[16];

private:
    //声明发布者
    //发布雷达点云信息
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub;
    //发布雷达融合后的检测结果，给其他模块
    rclcpp::Publisher<vision_interface::msg::DetectResult>::SharedPtr
        pub_radar;
};

//代表地图上某辆车的位置和编号
class map_car {
public:
    float x;
    float y;
    int   id;
};
}  // namespace tdt_radar

#endif
