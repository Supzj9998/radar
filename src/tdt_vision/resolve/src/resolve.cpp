#include "resolve.h"

#define TDT_INFO(msg) std::cout << msg << std::endl

namespace tdt_radar {
//调用父类构造函数创建"radar_resolve_node"节点
Resolve::Resolve(const rclcpp::NodeOptions& node_options)
    : Node("radar_resolve_node", node_options)
{
    //创建parser_对象
    parser_ = new parser();
    //读取"config/RM2025.png"加载小地图图像到minimap
    minimap = cv::imread("config/RM2025.png");
    //订阅camera_point2D话题的消息作为callback的第一个参数 用于接收来自摄像头或雷达的单个2D点坐标
    point_sub = this->create_subscription<geometry_msgs::msg::Vector3>(
        "camera_point2D", rclcpp::SensorDataQoS(),
        std::bind(&Resolve::callback, this, std::placeholders::_1));
    //发布"camera_point3D"话题 用于把2D坐标变换后的3D点云发出去
    pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "camera_point3D", rclcpp::SensorDataQoS());
    //订阅"match_info"话题的消息作为MatchInfoCallback的第一个参数 用于接收比赛信息
    match_info_sub =
        this->create_subscription<vision_interface::msg::MatchInfo>(
            "match_info", rclcpp::SensorDataQoS(),
            std::bind(&Resolve::MatchInfoCallback, this,
                      std::placeholders::_1));
    //订阅"detect_result"话题作为DetectCallback的第一个参数 用于接收视觉识别模块的结果
    detect_sub =
        this->create_subscription<vision_interface::msg::DetectResult>(
            "detect_result", rclcpp::SensorDataQoS(),
            std::bind(&Resolve::DetectCallback, this,
                      std::placeholders::_1));
    //发布"resolve_result"话题 用于发布雷达/视觉融合之后的最终坐标数据
    pub_radar = this->create_publisher<vision_interface::msg::DetectResult>(
        "resolve_result", rclcpp::SensorDataQoS());
    //输出初始化信息
    TDT_INFO("Load radar resolve node success!");
}
//回调函数处理比赛信息
void Resolve::MatchInfoCallback(
    const vision_interface::msg::MatchInfo::SharedPtr msg)
{
    //用markers接收6个标志点信息
    for (int i = 0; i < 6; i++) {
        markers[i] = msg->marks[i];
    }
    //robot_hp接收16个机器人血量
    for (int i = 0; i < 16; i++) {
        robot_hp[i] = msg->robot_hp[i];
    }
    //用match_time接收比赛时间
    match_time = msg->match_time;
}

//将收到的二维坐标转换为地图坐标并发布成点云
void Resolve::callback(const geometry_msgs::msg::Vector3::SharedPtr msg)
{
    //把ROS消息转换成opencv二维点
    cv::Point2f point;
    point.x = msg->x;
    point.y = msg->y;
    //计算该点在2D地图中的坐标(x,y)
    auto center_point = parser_->parse(point);

    //创建一个PCL类型的点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ send_point;
    //将二维点的x,y坐标传给三维点
    send_point.x = center_point.x;
    send_point.y = 15 - center_point.y;
    std::cout << center_point << std::endl;
    //三维点的z坐标固定是1
    send_point.z = 1;
    //将三维点传入点云
    cloud->points.push_back(send_point);
    //创建ROS消息
    sensor_msgs::msg::PointCloud2 output;
    //将点云转换成ROS消息
    pcl::toROSMsg(*cloud, output);
    //设置坐标系名称
    output.header.frame_id = "rm_frame";
    //将output消息发布到camera_point3D话题
    pub->publish(output);
}

//把视觉检测到的“蓝方”和“红方”坐标转换到地图坐标，并绘制在小地图上，同时生成点云并发布
void Resolve::DetectCallback(
    const vision_interface::msg::DetectResult::SharedPtr msg)
{
    //声明一个小地图的克隆
    cv::Mat                                 Map_clone = minimap.clone();
    //声明一辆车
    std::vector<map_car>                    cars;
    //创建消息对象
    vision_interface::msg::DetectResult     send_data;
    //创建点云对象
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZRGBA>);
    //遍历6个目标对每个目标分别处理蓝方和红方的数据
    for (int i = 0; i < 6; i++) {
        //创建二维点接收消息中的蓝色点
        cv::Point2f blue_point;
        blue_point.x = msg->blue_x[i];
        blue_point.y = msg->blue_y[i];
        //判断此点是否有效
        if (blue_point.x * blue_point.y) {
            //计算此点在2D地图上的坐标(x,y)
            auto center_point = parser_->parse(blue_point);
            //坐标修正
            center_point.y = 15 + center_point.y;
            //将此点存入ROS消息中
            send_data.blue_x[i] = center_point.x;
            send_data.blue_y[i] = center_point.y;
            //将此点传入 生成一个蓝色的点并传入点云
            pcl::PointXYZRGBA send_point;
            send_point.x = center_point.x;
            send_point.y = center_point.y;
            send_point.z = 1;
            send_point.r = 0;
            send_point.g = 0;
            send_point.b = 255;
            send_point.a = 255;
            cloud->points.push_back(send_point);
            //在克隆的小地图上绘制出这个点
            cv::circle(
                Map_clone,
                cv::Point((Map_clone.cols * center_point.x) / 28,
                          Map_clone.rows * (15 - center_point.y) / 15),
                20, cv::Scalar(200, 0, 0), -1);
            //在点旁写下编号
            cv::putText(
                Map_clone, std::to_string(i + 1),
                cv::Point((Map_clone.cols * center_point.x) / 28 - 10,
                          Map_clone.rows * (15 - center_point.y) / 15 + 10),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        }
        //红色同理
        cv::Point2f red_point;
        red_point.x = msg->red_x[i];
        red_point.y = msg->red_y[i];
        if (red_point.x * red_point.y) {
            auto center_point = parser_->parse(red_point);
            center_point.y = 15 + center_point.y;
            send_data.red_x[i] = center_point.x;
            send_data.red_y[i] = center_point.y;
            pcl::PointXYZRGBA send_point;
            send_point.x = center_point.x;
            send_point.y = center_point.y;
            send_point.z = 1;
            send_point.r = 255;
            send_point.g = 0;
            send_point.b = 0;
            send_point.a = 255;
            cloud->points.push_back(send_point);
            cv::circle(
                Map_clone,
                cv::Point((Map_clone.cols * center_point.x) / 28,
                          Map_clone.rows * (15 - center_point.y) / 15),
                20, cv::Scalar(0, 0, 200), -1);
            cv::putText(
                Map_clone, std::to_string(i + 1),
                cv::Point((Map_clone.cols * center_point.x) / 28 - 10,
                          Map_clone.rows * (15 - center_point.y) / 15 + 10),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        }
    }
    //将接收到消息的时间传入要发布的消息
    send_data.header.stamp = msg->header.stamp;
    //发布消息
    pub_radar->publish(send_data);
    //创建一个ROS2点云消息
    auto cloud_msg = sensor_msgs::msg::PointCloud2();
    //将PCL点云转换成ROS消息
    pcl::toROSMsg(*cloud, cloud_msg);
    //设置消息头 坐标轴+时间
    cloud_msg.header.frame_id = "rm_frame";
    cloud_msg.header.stamp = msg->header.stamp;
    //发布点云消息
    pub->publish(cloud_msg);
}
}  // namespace tdt_radar
RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Resolve)