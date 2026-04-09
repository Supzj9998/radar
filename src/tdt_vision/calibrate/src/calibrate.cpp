#include "calibrate.h"
#include <string>

namespace tdt_radar {

Calibrate::Calibrate(const rclcpp::NodeOptions& options)
    : Node("radar_calibrate_node", options)
{
    std::cout << "Calibrate start" << std::endl;
    // 开大窗口
    cv::namedWindow("calibrate", cv::WINDOW_AUTOSIZE);
    cv::resizeWindow("calibrate", 1440, 900);
    cv::moveWindow("calibrate", 1920 - 1440, 1080 - 900);
    // 开ROI小窗口
    cv::namedWindow("ROI", cv::WINDOW_AUTOSIZE);
    cv::resizeWindow("ROI", 400, 400);
    cv::moveWindow("ROI", 0, 0);
    // 注册鼠标回调函数
    cv::setMouseCallback("calibrate", mousecallback, 0);
    // 读取相机参数文件
    cv::FileStorage fs;
    fs.open("./config/camera_params.yaml", cv::FileStorage::READ);
    fs["camera_matrix"] >> camera_matrix;
    fs["dist_coeffs"] >> dist_coeffs;
    fs.release();

    // 初始化真实世界中的标定点
    real_points.push_back(self_FORTRESS);
    real_points.push_back(self_Tower);
    real_points.push_back(enemy_Base);
    real_points.push_back(enemy_Tower);
    real_points.push_back(enemy_High);
    // 创建解析器对象
    parser_ = new parser();

    RCLCPP_INFO(this->get_logger(), "\n"
                                    "  ______    ____  ______\n"
                                    " /_  __/   / __ \\/_  __/\n"
                                    "  / /_____/ / / / / /   \n"
                                    " / /_____/ /_/ / / /    \n"
                                    "/_/     /_____/ /_/     ");

    // 订阅普通图像话题，进回调
    image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "camera_image", rclcpp::SensorDataQoS(),
        std::bind(&Calibrate::callback, this, std::placeholders::_1));
    // 订阅压缩图像话题，进回调
    compressed_image_sub =
        this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "compressed_image", rclcpp::SensorDataQoS(),
            std::bind(&Calibrate::compressed_callback, this,
                      std::placeholders::_1));
    std::cout << "Calibrate end" << std::endl;
}

void Calibrate::callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    // ros转opencv
    auto    img = cv_bridge::toCvCopy(msg, "bgr8")->image;
    cv::Mat calib_img;
    cv::resize(img, calib_img, cv::Size(1536, 1125));
    cvimage_ = calib_img;
    // 判断是否标定
    if (is_calibrating) {
        // 写字
        cv::putText(img, std::to_string(pick_points.size()),
                    cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 3,
                    cv::Scalar(0, 0, 255), 2);
        cv::putText(img, "Press 'n' to add good point", cv::Point(50, 400),
                    cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 2);
        if (pick_points.size() == real_points.size()) {
            solve();
            parser_->Change_Matrix();
        }
    } else {
        parser_->draw_ui(img);
        cv::putText(img, "Press Enter to Calibrate !!!", cv::Point(50, 200),
                    cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 2);
    }
    auto temp = img.clone();
    cv::resize(img, img, cv::Size(1536, 1125));
    cv::imshow("calibrate", img);
    auto key = cv::waitKey(10);
    switch (key) {
    case 13:
        is_calibrating = true;
        break;
    default:
        break;
    }
}

void Calibrate::compressed_callback(
    const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{
    // 解码压缩图像
    auto    img = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    cv::Mat calib_img;
    // 生成标定用缩放图
    cv::resize(img, calib_img, cv::Size(1536, 1125));
    cvimage_ = calib_img;
    if (is_calibrating) {
        // 显示当前已经选了几个点
        cv::putText(img, std::to_string(pick_points.size()),
                    cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 3,
                    cv::Scalar(0, 0, 255), 2);
        // 判断点数是否够了
        if (pick_points.size() == real_points.size()) {
            solve();
            parser_->Change_Matrix();
        }
    } else {
        parser_->draw_ui(img);
        cv::putText(img, "Press Enter to Calibrate !!!", cv::Point(50, 200),
                    cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 2);
    }
    auto temp = img.clone();
    cv::resize(img, img, cv::Size(1536, 1125));
    cv::imshow("calibrate", img);
    auto key = cv::waitKey(10);
    switch (key) {
    case 13:
        is_calibrating = true;
        break;
    default:
        break;
    }
}

void mousecallback(int event, int x, int y, int flags, void* userdata)
{
    // 临时变量保存键盘按键
    int temp_key = 0;

    switch (event) {
    // 左键点击
    case cv::EVENT_LBUTTONDOWN:
        if (is_calibrating) {
            do {
                temp_key = cv::waitKey(10);
                // 判断是否标定
                switch (temp_key) {
                case 'w':
                    y -= 1;
                    break;
                case 'a':
                    x -= 1;
                    break;
                case 's':
                    y += 1;
                    break;
                case 'd':
                    x += 1;
                    break;
                }
                // 防止 ROI 超出边界
                x = std::max(50, std::min(x, cvimage_.cols - 50));
                y = std::max(50, std::min(y, cvimage_.rows - 50));
                // 取出鼠标附近的 ROI
                cv::Mat roi = cvimage_(cv::Rect(x - 50, y - 50, 100, 100));
                cv::Mat dst;
                cv::resize(roi, dst, cv::Size(400, 400));
                // 在放大图上画十字准星
                cv::line(dst, cv::Point(200, 100), cv::Point(200, 300),
                         cv::Scalar(0, 0, 255), 1);
                cv::line(dst, cv::Point(100, 200), cv::Point(300, 200),
                         cv::Scalar(0, 0, 255), 1);
                cv::imshow("ROI", dst);

            } 
            // 标定
            while (temp_key != 'n');

            x *= 1.3333333333 * 2;
            y *= 1.3333333333 * 2;
            // 打印当前点
            std::cout << "x:" << x << " y:" << y << std::endl;
            pick_points.push_back(cv::Point2f(x, y));
        }
        break;

    case cv::EVENT_MOUSEMOVE:
        if (x > cvimage_.cols - 50 || y > cvimage_.rows - 50 || x < 50 ||
            y < 50)
            break;
        cv::Mat roi = cvimage_(cv::Rect(x - 50, y - 50, 100, 100));
        cv::Mat dst;
        cv::resize(roi, dst, cv::Size(400, 400));
        cv::line(dst, cv::Point(200, 100), cv::Point(200, 300),
                 cv::Scalar(0, 0, 255), 1);
        cv::line(dst, cv::Point(100, 200), cv::Point(300, 200),
                 cv::Scalar(0, 0, 255), 1);
        cv::imshow("ROI", dst);
        break;
    }
}

void Calibrate::solve()
{
    cv::solvePnP(real_points, pick_points, camera_matrix, dist_coeffs, rvec,
                 tvec, 0, cv::SOLVEPNP_EPNP);
    std::cout << "rvec:" << rvec << std::endl;
    std::cout << "tvec:" << tvec << std::endl;
    cv::FileStorage fs;
    fs.open("./config/out_matrix.yaml", cv::FileStorage::WRITE);
    fs << "world_rvec" << rvec;
    fs << "world_tvec" << tvec;
    fs.release();
    pick_points.clear();
    is_calibrating = false;
}
}  // namespace tdt_radar

RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Calibrate);
