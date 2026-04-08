#include "cluster.h"
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>
#include <rclcpp/duration.hpp>
namespace tdt_radar {

Cluster::Cluster(const rclcpp::NodeOptions& node_options)
    : Node("cluster_node", node_options)
{
    RCLCPP_INFO(this->get_logger(), "cluster_node start");
    // 订阅动态点话题 进回调
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_dynamic", 10,
        std::bind(&Cluster::callback, this, std::placeholders::_1));
    // 发布器
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_cluster", 10);
}

void Cluster::callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // 记录开始时间
    std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();
    // 准备容器
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    // ROS 点云转 PCL 点云
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty()) {
        return;
    }
    // 创建 KD-Tree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    // 设置 KD-Tree 输入点云
    tree->setInputCloud(cloud);
    // 获取当前系统时间
    auto time = std::chrono::system_clock::now();

    // 创建欧式聚类对象
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    // 设置聚类距离阈值
    ec.setClusterTolerance(0.25);
    // 设置最小聚类点数
    ec.setMinClusterSize(5);
    // 设置最大聚类点数
    ec.setMaxClusterSize(1000);
    // 设置搜索方法（KD-Tree)
    ec.setSearchMethod(tree);
    // 设置输入点云
    ec.setInputCloud(cloud);
    // 定义聚类结果容器
    std::vector<pcl::PointIndices> cluster_indices;
    // 执行聚类
    ec.extract(cluster_indices);
    // 创建输出点云
    pcl::PointCloud<pcl::PointXYZ>* out_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    // 遍历每个聚类
    for (auto it = cluster_indices.begin(); it != cluster_indices.end();
         ++it) {
            // 为当前聚类创建点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(
            new pcl::PointCloud<pcl::PointXYZ>);
            // 把当前聚类中的点拷贝出来
        for (auto pit = it->indices.begin(); pit != it->indices.end();
             ++pit) {
            cloud_cluster->points.push_back(cloud->points[*pit]);
        }
        // 设置当前聚类点云属性
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // 求聚类中心点
        pcl::PointXYZ move_point;
        // 累加聚类内所有点
        for (auto point : cloud_cluster->points) {
            move_point.x += point.x;
            move_point.y += point.y;
            move_point.z += point.z;
        }
        // 求平均值
        move_point.x /= cloud_cluster->points.size();
        move_point.y /= cloud_cluster->points.size();
        move_point.z /= cloud_cluster->points.size();
        // 把中心点加入输出点云
        out_cloud->points.push_back(move_point);
    }
    // 转回 ROS 消息并发布
    sensor_msgs::msg::PointCloud2 output;
    // PCL 点云转 ROS 点云
    pcl::toROSMsg(*out_cloud, output);
    // 设置坐标系
    output.header.frame_id = "rm_frame";
    // 设置时间戳
    output.header.stamp = msg->header.stamp;
    // 发布结果
    pub_->publish(output);
    // 记录结束时间
    std::chrono::steady_clock::time_point t2 =
        std::chrono::steady_clock::now();
    // 打印处理耗时
    RCLCPP_INFO(
        this->get_logger(), "Cluster callback time: %f",
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                .count() /
            1000.0);
}
}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Cluster)