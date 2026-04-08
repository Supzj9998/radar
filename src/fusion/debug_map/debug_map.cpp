#include <fstream>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_interface/msg/detect_result.hpp>
#include <vision_interface/msg/match_info.hpp>
#include <vision_interface/msg/radar2_sentry.hpp>
#include <vision_interface/msg/radar_warn.hpp>
namespace tdt_radar {
class DebugMap : public rclcpp::Node {
public:
    explicit DebugMap(const rclcpp::NodeOptions& options)
        : Node("debug_map", options)
    {
        // 订阅kalman的结果，进回调
        detect_result_sub =
            this->create_subscription<vision_interface::msg::DetectResult>(
                "/kalman_detect", 10,
                std::bind(&DebugMap::callback, this,
                          std::placeholders::_1));
        // 订阅resolve的结果，进相机回调
        camera_detect_sub =
            this->create_subscription<vision_interface::msg::DetectResult>(
                "/resolve_result", rclcpp::SensorDataQoS(),
                std::bind(&DebugMap::camera_callback, this,
                          std::placeholders::_1));
        // 读取地图图片
        map = cv::imread("config/RM2025.png");
        // 订阅比赛信息话题，进save_match_info
        match_info_sub =
            this->create_subscription<vision_interface::msg::MatchInfo>(
                "/match_info", 10,
                std::bind(&DebugMap::save_match_info, this,
                          std::placeholders::_1));
        // 订阅预警
        radar_warn_pub =
            this->create_publisher<vision_interface::msg::RadarWarn>(
                "/hero_state", 10);
        // 创建map发布者
        debug_map_pub =
            this->create_publisher<sensor_msgs::msg::Image>("/map_2d", 10);
        // 创建哨兵发布者
        radar2sentry_pub =
            this->create_publisher<vision_interface::msg::Radar2Sentry>(
                "/Radar2Sentry", rclcpp::SensorDataQoS());
        // 缩放地图图片
        cv::resize(map, map, cv::Size(28 * 25, 15 * 25));
    }
    // 保存比赛信息
    void save_match_info(
        const std::shared_ptr<vision_interface::msg::MatchInfo> msg)
    {
        this->match_info = *msg;
        if (msg->self_color == 1)
            match_info.self_color = 2;
    }

    void show_map()
    {
        // 获取当前时间
        auto   now_time = std::chrono::system_clock::now();
        // 把当前时间转成一个 double 秒数
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now_time.time_since_epoch())
                          .count() /
                      1000.0;
        // 复制地图
        auto clone_map = map.clone();
        // 循环处理 6 个目标槽位
        for (int i = 0; i < 6; i++) {
            // 生成显示编号
            int number = i + 1;
            if (number == 6)
                number++;
            // 判断是否绘制蓝方目标
            if (blue_point[i].x * blue_point[i].y &&
                time - blue_update[i] < 0.5) {
                // 把蓝方地图坐标映射成图像像素坐标
                cv::Point2f point = cv::Point2f(
                    clone_map.cols * blue_point[i].x / 28,
                    clone_map.rows * (15 - blue_point[i].y) / 15);
                // 画蓝色圆点
                cv::circle(clone_map, point, 10, cv::Scalar(200, 0, 0), -1);
                // 在蓝点上写编号
                cv::putText(clone_map, std::to_string(number),
                            cv::Point(point.x - 6, point.y + 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 255, 255));
            }
            // 红方部分
            if (red_point[i].x * red_point[i].y &&
                time - red_update[i] < 0.5) {
                cv::Point2f point = cv::Point2f(
                    clone_map.cols * red_point[i].x / 28,
                    clone_map.rows * (15 - red_point[i].y) / 15);
                cv::circle(clone_map, point, 10, cv::Scalar(0, 0, 200), -1);
                cv::putText(clone_map, std::to_string(number),
                            cv::Point(point.x - 6, point.y + 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 255, 255));
            }
        }
        cv::imshow("map", clone_map);
        cv::waitKey(1);
    }
    void camera_callback(
        const std::shared_ptr<vision_interface::msg::DetectResult> msg)
    {
        // 获取当前时间
        auto   now = std::chrono::system_clock::now();
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch())
                          .count() /
                      1000.0;
        // 遍历六个目标
        for (int i = 0; i < 6; i++) {
            // 处理蓝方目标
            if (msg->blue_x[i] * msg->blue_y[i]) {
                // 蓝方目标的时间判断
                if (time - blue_time[i] > 5) {
                    // 更新蓝方坐标
                    blue_point[i] =
                        cv::Point2f(msg->blue_x[i], msg->blue_y[i]);
                    // 根据己方颜色做坐标翻转
                    if (!match_info.self_color) {
                        blue_point[i] = cv::Point2f(28 - msg->blue_x[i],
                                                    15 - msg->blue_y[i]);
                    }
                    // 记录蓝方更新时间
                    blue_update[i] = time;
                }
            }
            // 处理红方目标
            if (msg->red_x[i] * msg->red_y[i]) {
                if (time - red_time[i] > 5) {
                    red_point[i] =
                        cv::Point2f(msg->red_x[i], msg->red_y[i]);
                    if (!match_info.self_color) {
                        red_point[i] = cv::Point2f(28 - msg->red_x[i],
                                                   15 - msg->red_y[i]);
                    }
                    red_update[i] = time;
                }
            }
        }
        // 显示地图
        show_map();
    }

    void
    callback(const std::shared_ptr<vision_interface::msg::DetectResult> msg)
    {
        // 获取当前时间
        auto   now = std::chrono::system_clock::now();
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch())
                          .count() /
                      1000.0;
        // 遍历六个目标
        for (int i = 0; i < 6; i++) {
            // 更新蓝色方
            if (msg->blue_x[i] * msg->blue_y[i]) {
                blue_point[i] = cv::Point2f(msg->blue_x[i], msg->blue_y[i]);
                blue_time[i] = time;
                blue_update[i] = time;
            }
            // 更新红色方
            if (msg->red_x[i] * msg->red_y[i]) {
                red_point[i] = cv::Point2f(msg->red_x[i], msg->red_y[i]);
                red_time[i] = time;
                red_update[i] = time;
            }
        }
        // 显示地图
        show_map();
        // 创建预警信息
        vision_interface::msg::RadarWarn radar_warn;
        // 英雄状态预警
        if (hero_count1 > 10) {
            radar_warn.hero_state = 1;
            radar_warn_pub->publish(radar_warn);
        } else if (hero_count2 > 10) {
            radar_warn.hero_state = 2;
            radar_warn_pub->publish(radar_warn);
        }
        // 判断敌方英雄所在区域
        if (match_info.self_color == 0) {
            if (red_point[0].x > (28 - 8.668)) {
                hero_count1++;
                hero_count2--;
            } else if (red_point[0].x < (28 - 20.3) &&
                       red_point[0].x > (28 - 25.075) &&
                       red_point[0].y < 15 && red_point[0].y > 10.3) {
                hero_count1--;
                hero_count2++;
            } else {
                hero_count1--;
                hero_count2--;
            }
        }
        if (match_info.self_color == 2) {
            if (blue_point[0].x < 8.668) {
                hero_count1++;
                hero_count2--;
            } else if (blue_point[0].x > 20.3 && blue_point[0].x < 25.075 &&
                       blue_point[0].y > 0 &&
                       blue_point[0].y < (15 - 10.3)) {
                hero_count1--;
                hero_count2++;
            } else {
                hero_count1--;
                hero_count2--;
            }
        }

        // 发给哨兵的信息
        vision_interface::msg::Radar2Sentry radar2sentry;
        if (match_info.self_color == 0) {
            for (int i = 0; i < 6; i++) {
                if (!relax[i]) {
                    if (match_info.marks[i] >= 117) {
                        relax[i] = true;
                        relax_time[i] = time;
                    } else {
                        if (time - red_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = red_point[i].x;
                            radar2sentry.radar_enemy_y[i] = red_point[i].y;
                        }
                    }
                } else {
                    if (match_info.marks[i] < 105) {
                        relax[i] = false;
                        if (time - red_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = red_point[i].x;
                            radar2sentry.radar_enemy_y[i] = red_point[i].y;
                        }
                    } else if (time - relax_time[i] > 0.35) {
                        relax_time[i] = time;
                        if (time - red_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = red_point[i].x;
                            radar2sentry.radar_enemy_y[i] = red_point[i].y;
                        }
                    }
                }
            }
        }
        if (match_info.self_color == 2) {
            for (int i = 0; i < 6; i++) {
                if (!relax[i]) {
                    if (match_info.marks[i] >= 117) {
                        relax[i] = true;
                        relax_time[i] = time;
                    } else {
                        if (time - blue_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = blue_point[i].x;
                            radar2sentry.radar_enemy_y[i] = blue_point[i].y;
                        }
                    }
                } else {
                    if (match_info.marks[i] < 105) {
                        relax[i] = false;
                        if (time - blue_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = blue_point[i].x;
                            radar2sentry.radar_enemy_y[i] = blue_point[i].y;
                        }
                    } else if (time - relax_time[i] > 0.35) {
                        relax_time[i] = time;
                        if (time - blue_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = blue_point[i].x;
                            radar2sentry.radar_enemy_y[i] = blue_point[i].y;
                        }
                    }
                }
            }
        }
        radar2sentry_pub->publish(radar2sentry);
    }
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        detect_result_sub;
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
                                                          camera_detect_sub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_map_pub;
    rclcpp::Publisher<vision_interface::msg::RadarWarn>::SharedPtr
        radar_warn_pub;
    rclcpp::Publisher<vision_interface::msg::Radar2Sentry>::SharedPtr
        radar2sentry_pub;
    rclcpp::Subscription<vision_interface::msg::MatchInfo>::SharedPtr
        match_info_sub;

    double blue_time[6];
    double red_time[6];

    bool   relax[6];
    double relax_time[6];

    double blue_update[6];
    double red_update[6];

    int hero_count1;
    int hero_count2;

    cv::Point2f blue_point[6];
    cv::Point2f red_point[6];

    vision_interface::msg::MatchInfo match_info;
    cv::Mat                          map;
    int                              count = 0;
};
}  // namespace tdt_radar

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node_options = rclcpp::NodeOptions();
    rclcpp::spin(std::make_shared<tdt_radar::DebugMap>(node_options));
    rclcpp::shutdown();
    return 0;
}
