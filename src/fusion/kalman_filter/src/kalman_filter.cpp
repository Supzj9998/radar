#include "kalman_filter.h"
#include <pcl_conversions/pcl_conversions.h>
#include "filter_plus.h"

namespace tdt_radar {

KalmanFilter::KalmanFilter(const rclcpp::NodeOptions& node_options)
    : rclcpp::Node("kalman_filter_node", node_options)
{
    // 订阅欧式聚类点云信息，进回调
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_cluster", 10,
        std::bind(&KalmanFilter::callback, this, std::placeholders::_1));
    // 发布卡尔曼滤波后的点云
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_kalman", 10);
    // 发布给哨兵/其他模块的数据
    radar_pub_ =
        this->create_publisher<vision_interface::msg::Radar2Sentry>(
            "/radar2sentry", 10);
    // 发布检测结果
    radar_detect_pub_ =
        this->create_publisher<vision_interface::msg::DetectResult>(
            "/kalman_detect", 10);
    // 订阅视觉识别结果，进检测回调
    sub_detect_ =
        this->create_subscription<vision_interface::msg::DetectResult>(
            "/resolve_result", rclcpp::SensorDataQoS(),
            std::bind(&KalmanFilter::detect_callback, this,
                      std::placeholders::_1));
    // 订阅激光雷达预警信息
    sub_lidar_ =
        this->create_subscription<vision_interface::msg::RadarWarn>(
            "/lidar_detect", 10,
            std::bind(&KalmanFilter::lidar_callback, this,
                      std::placeholders::_1));
    // 订阅匹配信息，进match回调
    sub_match_ =
        this->create_subscription<vision_interface::msg::MatchInfo>(
            "/match_info", 10,
            std::bind(&KalmanFilter::match_callback, this,
                      std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Kalman_filter_Node has been started.");
}

void KalmanFilter::match_callback(
    const vision_interface::msg::MatchInfo::SharedPtr msg)
{
    // 将match消息保存到当前节点的成员变量 match_info
    this->match_info = *msg;
    RCLCPP_INFO(this->get_logger(), "Match_info_callback");
}

void KalmanFilter::detect_callback(
    const vision_interface::msg::DetectResult::SharedPtr msg)
{
    // 取时间戳
    rclcpp::Time time = msg->header.stamp;
    // 遍历 6 个目标槽位
    for (int i = 0; i < 6; i++) {
        // 创建一个红方点
        pcl::PointXY red_point;
        red_point.x = msg->red_x[i];
        red_point.y = msg->red_y[i];
        // 过滤无效红点
        if (red_point.x == 0 || red_point.y == 0)
            continue;
        // 遍历所有卡尔曼跟踪器并做匹配（视觉检测到一个点时，并不知道它一定属于哪个 kf
        // 让每个 kf 都试一次匹配最终只有时间和空间都接近的那个 kf 才会把这条识别记录收下。
        for (auto& kf : KFs) {
            kf.camera_match(time, red_point, 2, i);
        }

        // 创建一个蓝方点
        pcl::PointXY blue_point;
        blue_point.x = msg->blue_x[i];
        blue_point.y = msg->blue_y[i];
         // 过滤无效蓝点
        if (blue_point.x == 0 || blue_point.y == 0)
            continue;
        // 遍历所有卡尔曼跟踪器并做匹配
        for (auto& kf : KFs) {
            kf.camera_match(time, blue_point, 0, i);
        }
    }
}

// 收到一条 RadarWarn 消息后，把这条消息保存到类成员 lidar_detect
void KalmanFilter::lidar_callback(
    const vision_interface::msg::RadarWarn::SharedPtr msg)
{
    this->lidar_detect = *msg;
}


void KalmanFilter::callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // 取时间戳
    rclcpp::Time time = msg->header.stamp;
    // 记录当前程序运行时刻
    auto         now_time = std::chrono::steady_clock::now();
    // 创建三维点云对象 cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    // 创建二维点云对象 cloud_xy
    pcl::PointCloud<pcl::PointXY>::Ptr cloud_xy(
        new pcl::PointCloud<pcl::PointXY>);
    // ROS 点云转 PCL 点云
    pcl::fromROSMsg(*msg, *cloud);
    // 把三维点云投影成二维点云
    for (auto point : cloud->points) {
        pcl::PointXY point_xy;
        point_xy.x = point.x;
        point_xy.y = point.y;
        cloud_xy->points.push_back(point_xy);
    }
    // 判断二维点云是否为空
    if (cloud_xy->points.size() == 0)
        return;
    // 让所有KFs先做一次预测
    for (auto& kf : KFs) {
        kf.update_predict_point();
        kf.has_updated = false;
    }

    // 把当前帧里的每一个雷达点，和已有的所有卡尔曼跟踪器做匹配；匹配不上就新建轨迹，匹配上就更新轨迹；如果匹配上多个，就选最近的那个更新。
    // 遍历当前帧所有观测点
    for (auto point : cloud_xy->points) {
        // 容器，存储哪些卡尔曼滤波器和这个点匹配
        std::vector<int> match_kf_indexs;
        // 遍历所有已有轨迹，逐个尝试匹配
        for (int i = 0; i < this->KFs.size(); i++) {
            if (KFs[i].match(point)) {
                match_kf_indexs.push_back(i);
            }
        }
        // 没有任何轨迹匹配上，新建一个卡尔曼跟踪器
        if (match_kf_indexs.size() == 0) {
            Kalman_filter_plus kf(point, time);
            KFs.push_back(kf);
        } 
        // 刚好只有一个轨迹匹配上，拿这个点去更新
        else if (match_kf_indexs.size() == 1) {
            KFs[match_kf_indexs[0]].update(point, time);
        } 
        // 有多个轨迹都能匹配这个点
        else {
            // 先设一个很大的最小距离
            float min_distance = 1000000;
            int   min_index = 0;
            // 遍历所有候选轨迹
            for (auto index : match_kf_indexs) {
                // 计算“预测位置到观测点”的距离
                float distance =
                    KFs[index].Distance(KFs[index].predict_point, point);
                // 找距离最小者
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = index;
                }
            }
            // 用最近轨迹更新
            KFs[min_index].update(point, time);
        }
    }
    // 创建一个新的彩色点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    // 遍历 KFs
    for (int i = KFs.size() - 1; i >= 0; i--) {
        // 判断轨迹是否超时
        if ((KFs[i].last_time) > 1.5) {
            KFs.erase(KFs.begin() + i);
        }
        // 没超时的轨迹，转换成一个点 
        else {
            // 把轨迹预测位置赋给点坐标
            pcl::PointXYZRGB point;
            point.x = KFs[i].predict_point.x;
            point.y = KFs[i].predict_point.y;
            point.z = 1.5;
            // 获取这个轨迹的颜色类别
            int color = KFs[i].get_color();
            // 根据颜色类别给点上色
            switch (color) {
            case 0: 
                point.b = 255;
                break;

            case 2:
                point.r = 255;
                break;

            default:
                point.r = KFs[i].color[0];
                point.g = KFs[i].color[1];
                point.b = KFs[i].color[2];
                break;
            }
            // 把点加入输出点云
            cloud_filtered->points.push_back(point);
        }
    }
    // 设置坐标系
    cloud_filtered->header.frame_id = "rm_frame";
    // 创建 ROS 点云消息
    sensor_msgs::msg::PointCloud2 output;
    // PCL 转 ROS 消息
    pcl::toROSMsg(*cloud_filtered, output);
    // 设置发布消息头
    output.header.frame_id = "rm_frame";
    output.header.stamp = msg->header.stamp;
    // 发布点云
    pub_->publish(output);
    // 统计本次回调耗时
    auto  end_time = std::chrono::steady_clock::now();
    float dur_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time - now_time)
                         .count();
    // 创建一个检测结果消息
    vision_interface::msg::DetectResult detect_msg;
    // 遍历所有轨迹
    for (auto kf : KFs) {
        // 没有识别历史的轨迹跳过
        if (kf.detect_history.size() == 0)
            continue;
        // 蓝色目标
        if (kf.get_color() == 0) {
            int number = kf.get_number();
            detect_msg.blue_x[number] = kf.predict_point.x;
            detect_msg.blue_y[number] = kf.predict_point.y;
        }
        // 红色目标
        if (kf.get_color() == 2) {
            int number = kf.get_number();
            detect_msg.red_x[number] = kf.predict_point.x;
            detect_msg.red_y[number] = kf.predict_point.y;
        }
    }
    // 根据己方颜色做坐标翻转
    if (match_info.self_color == 0) {
        for (int i = 0; i < 6; i++) {
            if (detect_msg.blue_x[i] != 0 && detect_msg.blue_y[i] != 0) {
                detect_msg.blue_x[i] = 28 - detect_msg.blue_x[i];
                detect_msg.blue_y[i] = 15 - detect_msg.blue_y[i];
            }
            if (detect_msg.red_x[i] != 0 && detect_msg.red_y[i] != 0) {
                detect_msg.red_x[i] = 28 - detect_msg.red_x[i];
                detect_msg.red_y[i] = 15 - detect_msg.red_y[i];
            }
        }
    }
    // 发布 DetectResult
    radar_detect_pub_->publish(detect_msg);

    // 整理给哨兵的消息
    vision_interface::msg::Radar2Sentry radar_msg;
    if (match_info.self_color == 0) {
        for (int i = 0; i < 6; i++) {
            radar_msg.radar_enemy_x[i] = detect_msg.red_x[i];
            radar_msg.radar_enemy_y[i] = detect_msg.red_y[i];
        }

        radar_pub_->publish(radar_msg);
    }
}
}  // namespace tdt_radar
RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::KalmanFilter)
