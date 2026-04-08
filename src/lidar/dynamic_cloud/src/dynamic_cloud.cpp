#include "dynamic_cloud.h"
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/time.hpp>
namespace tdt_radar {
DynamicCloud::DynamicCloud(const rclcpp::NodeOptions& node_options)
    : rclcpp::Node("dynamic_cloud_node", node_options),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_)
{
    RCLCPP_INFO(this->get_logger(), "Dynamic_cloud Node start");
    // 创建点云指针
    auto temp_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>);
    // 读取PCD地图文件
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("config/RM2025.pcd",
                                            *temp_cloud) == -1) {
        PCL_ERROR("Couldn't read file map.pcd \n");
    }
    // 创建体素滤波器
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    // 降采样
    sor.setInputCloud(temp_cloud);
    sor.setLeafSize(0.1f, 0.1f, 0.1f);
    auto result = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>);
    sor.filter(*result);
    // 保存降采样结果
    map_cloud = result;
    // 建立 KD 树
    kd_Tree.setInputCloud(map_cloud);

    // 订阅雷达发布的点云，进回调
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar", 10,
        std::bind(&DynamicCloud::callback, this, std::placeholders::_1));
    // 动态点云发布器
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_dynamic", 10);
    // 其他点云发布器
    other_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_other", 10);
    // 检测结果发布器
    detect_pub_ = this->create_publisher<vision_interface::msg::RadarWarn>(
        "/lidar_detect", 10);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Dynamic Cloud Launch!");
}

// 提取动态点云
void DynamicCloud::GetDynamicCloud(
    // 输入点云
    pcl::PointCloud<pcl::PointXYZ>& input_cloud,
    // 输出点云（动态点云）
    pcl::PointCloud<pcl::PointXYZ>& output_cloud, float threshold,
    int thread_num)
{
    // KD 树搜索时只找最近的 1 个点。
    int                                         K = 1;
    // 创建线程容器
    std::vector<std::thread>                    threads;
    // 给每个线程准备一个局部点云
    std::vector<pcl::PointCloud<pcl::PointXYZ>> clouds(thread_num);
    // 记录开始时间
    auto start = std::chrono::system_clock::now();
    // 输入点云的大小
    int  cloud_size = input_cloud.points.size();
    // 计算每个线程负责多少点
    int  step = cloud_size / thread_num;
    // 创建多个线程
    for (int i = 0; i < thread_num; i++) {
        threads.push_back(std::thread([i, step, cloud_size, &clouds,
                                       &input_cloud, this, threshold, K]() {
            for (int j = i * step; j < (i + 1) * step; j++) {
                std::vector<int>   pointIdxNKNSearch(K);
                std::vector<float> pointNKNSquaredDistance(K);
                if (kd_Tree.nearestKSearch(input_cloud.points[j], K,
                                           pointIdxNKNSearch,
                                           pointNKNSquaredDistance) > 0) {
                    // 用距离阈值判断是否为动态点
                    if (pointNKNSquaredDistance[0] > threshold) {
                        clouds[i].points.push_back(input_cloud.points[j]);
                    }
                }
            }
        }));
    }
    // 等待线程结束
    for (auto& t : threads) {
        t.join();
    }
    // 合并各线程结果
    for (auto& cloud : clouds) {
        output_cloud += cloud;
    }
    // 记录结束时间
    auto end = std::chrono::system_clock::now();
}

// 坐标变换
void TransformCloud(
    // 输入点云
    pcl::PointCloud<pcl::PointXYZ>&      input_cloud,
    // 输出点云
    pcl::PointCloud<pcl::PointXYZ>&      output_cloud,
    // TF变换
    geometry_msgs::msg::TransformStamped transform_stamped,
    // 线程数
    int                                  thread_num)
{
    // 创建线程函数
    std::vector<std::thread>                    threads;
    // 给每个线程准备一个局部结果点云
    std::vector<pcl::PointCloud<pcl::PointXYZ>> clouds(thread_num);
    // 记录开始时间
    auto start = std::chrono::system_clock::now();
    // 获取输入点云大小
    int  cloud_size = input_cloud.points.size();
    // 计算每个线程处理多少点
    int  step = cloud_size / thread_num;

    // 单位仿射变换
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z;
    Eigen::Quaternionf rotation(transform_stamped.transform.rotation.w,
                                transform_stamped.transform.rotation.x,
                                transform_stamped.transform.rotation.y,
                                transform_stamped.transform.rotation.z);
    transform.rotate(rotation);

    // 多线程处理 对每个点做坐标变换
    for (int i = 0; i < thread_num; i++) {
        threads.push_back(std::thread(
            [i, step, cloud_size, &clouds, &input_cloud, transform]() {
                for (int j = i * step; j < (i + 1) * step && j < cloud_size;
                     j++) {
                    pcl::PointXYZ   point = input_cloud.points[j];
                    Eigen::Vector3f point_vec(point.x, point.y, point.z);
                    Eigen::Vector3f point_out = transform * point_vec;

                    pcl::PointXYZ transformed_point;
                    transformed_point.x = point_out.x();
                    transformed_point.y = point_out.y();
                    transformed_point.z = point_out.z();

                    clouds[i].points.push_back(transformed_point);
                }
            }));
    }

    for (auto& t : threads) {
        t.join();
    }
    for (auto& cloud : clouds) {
        output_cloud += cloud;
    }
    // 计算耗时
    auto end = std::chrono::system_clock::now();
    RCLCPP_INFO(
        rclcpp::get_logger("rclcpp"), "transform time: %f",
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count() /
            1000.0);
}

// 主回调
void DynamicCloud::callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // 创建消息
    vision_interface::msg::RadarWarn lidar_detect;

    // 下方四个lambda判断一个点是否落在某个三维区域

    auto dart_cloud_filter = [](pcl::PointXYZ& point) {
        return (point.x > 28 - 0.5889 - 0.1885 && point.x < 28 - 0.5889) &&
               (point.y > 3.925 && point.y < 4.525) &&
               (point.z > 2.4722 - 0.859 + 0.1 && point.z < 2.4722);
    };  // 飞镖检测
    auto fly_safe_filter = [](pcl::PointXYZ& point) {
        return (point.x > 28 - 2.775 && point.x < 27.5) &&
               (point.y > 0.2 && point.y < 2.2) &&
               (point.z > 1.7 && point.z < 3);
    };  // 飞机起飞

    auto fly_warn_filter = [](pcl::PointXYZ& point) {
        return (point.x > 19.83 && point.x < 28 - 2.7) &&
               (point.y > 0.2 && point.y < 1.356 + 2.4 + 0.8) &&
               (point.z > 1.7 && point.z < 3);
    };  // 飞机飞到半场

    auto fly_alarm_filter = [](pcl::PointXYZ& point) {
        return (point.x > 13 && point.x < 20.5) &&
               (point.y > 0.2 && point.y < 1.356 + 2.4 + 0.8) &&
               (point.z > 1.7 && point.z < 3);
    };  // 飞机飞到中场
    // 把ros点云消息转换成pcl点云
    auto receive_cloud = pcl::PointCloud<pcl::PointXYZ>();
    pcl::fromROSMsg(*msg, receive_cloud);

    // 记录时间
    std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();
    // 准备 TF 变量
    geometry_msgs::msg::TransformStamped transform_stamped;
    // 记录时间
    auto ta = std::chrono::steady_clock::now();
    // 查询TF是否存在
    try {
        transform_stamped = tf_buffer_.lookupTransform(
            "rm_frame", msg->header.frame_id, tf2::TimePointZero);
    }
    catch (tf2::TransformException& ex) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Transform error: %s",
                     ex.what());
        return;
    }
    // 定义变换后的点云
    pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
    // 准备仿射变换
    auto                           transform = Eigen::Affine3f::Identity();
    transform.translation() << transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z;
    Eigen::Quaternionf rotation(transform_stamped.transform.rotation.w,
                                transform_stamped.transform.rotation.x,
                                transform_stamped.transform.rotation.y,
                                transform_stamped.transform.rotation.z);
    transform.rotate(rotation);
    // 仿射变换
    pcl::transformPointCloud(receive_cloud, transformed_cloud, transform);
    // 过滤结果点云
    pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
    pcl::PointCloud<pcl::PointXYZ> other_filtered_cloud;
    // 遍历变换后的所有点
    for (size_t i = 0; i < transformed_cloud.size(); i++) {
        auto& point = transformed_cloud.points[i];
        // 过滤不符合条件的点 只保留场地内，地面附近，有效区域
        if (point.x < 3 || point.x > 28 || point.y < 0 || point.y > 15 ||
            point.z < 0 || point.z > 1.4 ||
            (point.y > 0 && point.y < 5 && point.x > 25) ||
            ((21.5 - 2.9 / sqrt(2)) < (point.x + point.y) &&
             (point.x + point.y) < (21.5 + 2.9 / sqrt(2)) &&
             (-6.5 - 0.9 / sqrt(2)) < (point.y - point.x) &&
             (point.y - point.x) < (-6.5 + 0.9 / sqrt(2)))) {
            // 特殊高空区域保留到 other_filtered_cloud
            if ((point.x > 28 - 0.5889 - 0.1885 && point.x < 28 - 0.5889) &&
                    (point.y > 3.925 && point.y < 4.525) &&
                    (point.z > 2.4722 - 0.859 + 0.1 && point.z < 2.4722) ||
                (point.x > 13 && point.x < 27.5) &&
                    (point.y > 0.2 && point.y < 1.356 + 2.4 + 0.8) &&
                    (point.z > 1.7 && point.z < 3)) {
                other_filtered_cloud.push_back(point);
            }
            continue;
        }
        filtered_cloud.push_back(point);
    }
    // 提取当前帧动态点云
    pcl::PointCloud<pcl::PointXYZ> dynamic_pointcloud;
    GetDynamicCloud(filtered_cloud, dynamic_pointcloud, 0.1, 12);
    // 把当前帧结果加入历史缓存
    if (accumulate_count < accumulate_time) {
        accumulated_clouds_.push_back(dynamic_pointcloud.makeShared());
        other_accumulated_clouds_.push_back(
            other_filtered_cloud.makeShared());
        accumulate_count++;
    } else {
        accumulated_clouds_.erase(accumulated_clouds_.begin());
        accumulated_clouds_.push_back(dynamic_pointcloud.makeShared());
        other_accumulated_clouds_.erase(other_accumulated_clouds_.begin());
        other_accumulated_clouds_.push_back(
            other_filtered_cloud.makeShared());
    }
    // 合并主动态点云缓存
    pcl::PointCloud<pcl::PointXYZ> accumulated_cloud;
    for (auto it = accumulated_clouds_.begin();
         it != accumulated_clouds_.end(); ++it) {
        accumulated_cloud += **it;
    }
    // 合并特殊高空区域点云缓存
    pcl::PointCloud<pcl::PointXYZ> other_accumulated_cloud;
    for (auto it = other_accumulated_clouds_.begin();
         it != other_accumulated_clouds_.end(); ++it) {
        other_accumulated_cloud += **it;
    }
    // 更新时间点
    ta = std::chrono::steady_clock::now();
    // ROS 点云消息对象
    sensor_msgs::msg::PointCloud2 output;
    // 设置坐标系
    accumulated_cloud.header.frame_id = "rm_frame";
    other_accumulated_cloud.header.frame_id = "rm_frame";
    // PCL 点云转换成 ROS 消息
    pcl::toROSMsg(accumulated_cloud, output);
    // ROS 消息设置 frame_id
    output.header.frame_id = "rm_frame";
    // 时间戳
    output.header.stamp = msg->header.stamp;
    // 发布
    pub_->publish(output);

    // 把高空特殊区域点云转成 ROS 消息
    pcl::toROSMsg(other_accumulated_cloud, output);
    // 发布
    other_pub_->publish(output);
    // 筛选飞镖检测区域内的点
    pcl::PointCloud<pcl::PointXYZ> dart_cloud;
    for (size_t i = 0; i < other_accumulated_cloud.size(); i++) {
        auto& point = other_accumulated_cloud.points[i];
        if (dart_cloud_filter(point)) {
            dart_cloud.push_back(point);
        }
    }

    // 如果在飞镖检测区域里累计到的点数超过 5 个，就认为“检测到了飞镖相关目标”，然后输出警告日志  lidar_detect.dart_state = 1
    if (dart_cloud.size() > 5) {
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "Find dart cloud!");
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "Dart cloud size: %d",
                    dart_cloud.size());
        lidar_detect.dart_state = 1;
    }

    // 飞机状态判定准备
    pcl::PointCloud<pcl::PointXYZ> fly_safe_cloud;
    pcl::PointCloud<pcl::PointXYZ> fly_warn_cloud;
    pcl::PointCloud<pcl::PointXYZ> fly_alarm_cloud;

    // 统计飞机在“起飞区 / 预警区 / 报警区”的点数
    for (size_t i = 0; i < other_accumulated_cloud.size(); i++) {
        auto& point = other_accumulated_cloud.points[i];
        if (fly_safe_filter(point)) {
            fly_safe_cloud.push_back(point);
        }
        if (fly_warn_filter(point)) {
            fly_warn_cloud.push_back(point);
        }
        if (fly_alarm_filter(point)) {
            fly_alarm_cloud.push_back(point);
        }
    }
    // 点数超过 40，就设置飞机状态
    if (fly_safe_cloud.size() > 40) {
        lidar_detect.fly_state = 1;
    }

    if (fly_warn_cloud.size() > 40) {
        lidar_detect.fly_state = 2;
    }

    if (fly_alarm_cloud.size() > 40) {
        lidar_detect.fly_state = 3;
    }

    // 打印飞机状态
    switch (lidar_detect.fly_state) {
    case 0:
        break;
    case 1:
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"),
                    "Safe fly object detected!");
        break;
    case 2:
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"),
                    "Warn fly object detected!");
        break;
    case 3:
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"),
                     "Alarm fly object detected!");
        break;
    default:
        break;
    }

    // 发布检测结果，本帧雷达处理得到的最终告警结果发布出去
    detect_pub_->publish(lidar_detect);
    // 记录回调结束时间
    std::chrono::steady_clock::time_point t2 =
        std::chrono::steady_clock::now();
    // 打印总耗时
    RCLCPP_INFO(
        rclcpp::get_logger("rclcpp"), "Dynamic callback time: %f",
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                .count() /
            1000.0);
}
}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::DynamicCloud)
