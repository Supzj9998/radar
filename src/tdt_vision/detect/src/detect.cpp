#include "detect.h"

#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/utilities.hpp>
#define TDT_INFO(msg) std::cout << msg << std::endl
#define MAX_CARS 12
#define MAX_ARMORS 20
namespace tdt_radar {
// 图片数量
unsigned int count_img = 0;

bool isNonEmptyFile(const std::string& path)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    return fs::is_regular_file(path, ec) && fs::file_size(path, ec) > 0;
}

// 判断颜色
int getColor(cv::Mat& img)
{
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    cv::Mat    blueMinusRed = channels[0] - channels[2];
    cv::Mat    redMinusBlue = channels[2] - channels[0];
    cv::Scalar avgBlueMinusRed = cv::mean(blueMinusRed);
    cv::Scalar avgRedMinusBlue = cv::mean(redMinusBlue);
    cv::Scalar avgGreen = cv::mean(channels[1]);
    if (avgBlueMinusRed[0] > avgRedMinusBlue[0]) {
        // 偏蓝返回0
        return 0;
    } else {
        // 偏红返回2
        return 2;
    }
}

// 判断小矩形是否完全被包含在大矩形
bool isRectInside(const cv::Rect& small, const cv::Rect& big)
{
    bool topLeftInside = big.contains(small.tl());
    bool topRightInside =
        big.contains(cv::Point(small.x + small.width, small.y));
    bool bottomLeftInside =
        big.contains(cv::Point(small.x, small.y + small.height));
    bool bottomRightInside = big.contains(
        cv::Point(small.x + small.width, small.y + small.height));
    return (topLeftInside && topRightInside && bottomLeftInside &&
            bottomRightInside);
}

// 判断一个yolo检测框是否完全在另一个yolo检测框中
bool isBoxInside(const yolo::Box& small, const yolo::Box& big)
{
    cv::Rect small_rect(small.left, small.top, small.right - small.left,
                        small.bottom - small.top);
    cv::Rect big_rect(big.left, big.top, big.right - big.left,
                      big.bottom - big.top);
    return isRectInside(small_rect, big_rect);
}

// 安全裁剪roi函数
cv::Rect getSafeRect(cv::Mat& image, cv::Rect& rect)
{
    cv::Rect save_rect;
    save_rect.x = std::max(0, rect.x);
    save_rect.y = std::max(0, rect.y);
    save_rect.width = std::min(image.cols - save_rect.x, rect.width);
    save_rect.height = std::min(image.rows - save_rect.y, rect.height);
    return save_rect;
}

// 初始化detect节点名："radar_detect_node"
Detect::Detect(const rclcpp::NodeOptions& node_options)
    : Node("radar_detect_node", node_options)
{
    cv::namedWindow("detect", cv::WINDOW_NORMAL);

    // 检查 CUDA / NVIDIA 环境
    std::cout << "Checking CUDA with nvidia-smi...\n";
    if (system("nvidia-smi") == 0) {
        RCLCPP_INFO(this->get_logger(), "CUDA is available.");
    } else {
        RCLCPP_ERROR(this->get_logger(), "CUDA is not available. Exiting.");
        rclcpp::shutdown();
    }
    // 读取模型路径
    cv::FileStorage fs;
    fs.open("./config/detect_params.yaml", cv::FileStorage::READ);
    fs["yolo_path"] >> yolo_path;
    fs["armor_path"] >> armor_path;
    fs["classify_path"] >> classify_path;
    fs.release();

    // 检查 yolo engine
    if (!isNonEmptyFile(yolo_path)) {
        system("python3 src/utils/onnx2trt.py "
               "--onnx=model/ONNX/RM2025.onnx "
               "--saveEngine=model/TensorRT/yolo.engine "
               "--minBatch 1 "
               "--optBatch 1 "
               "--maxBatch 2 "
               "--Shape=1280x1280 "
               "--input_name=images");
    } else {
        TDT_INFO("Load yolo engine!");
    }
    // 检查 armor_yolo engine
    if (!isNonEmptyFile(armor_path)) {
        system("python3 src/utils/onnx2trt.py "
               "--onnx=model/ONNX/armor_yolo.onnx "
               "--saveEngine=model/TensorRT/armor_yolo.engine "
               "--minBatch 1 "
               "--optBatch 5 "
               "--maxBatch 12 "
               "--Shape=192x192 "
               "--input_name=images");
    } else {
        TDT_INFO("Load armor_yolo engine!");
    }
    // 检查 classify engine
    if (!isNonEmptyFile(classify_path)) {
        system("python3 src/utils/onnx2trt.py "
               "--onnx=model/ONNX/classify.onnx "
               "--saveEngine=model/TensorRT/classify.engine "
               "--minBatch 1 "
               "--optBatch 10 "
               "--maxBatch 20 "
               "--Shape=224x224 "
               "--input_name=input");
    } else {
        TDT_INFO("Load classify engine!");
    }
    // 打印路径
    std::cout << "yolo_path:" << yolo_path << "\n";
    std::cout << "armor_path:" << armor_path << "\n";
    std::cout << "classify_path:" << classify_path << "\n";

    // 加载分类器
    this->classifier =
        classify::load(classify_path, classify::Type::densenet121);
    if (!this->classifier) {
        throw std::runtime_error("Failed to load classify engine: " +
                                 classify_path);
    }
    TDT_INFO("Load classify engine success!");

    // 加载装甲板检测模型
    this->armor_yolo = yolo::load(armor_path, yolo::Type::V8, 0.4f, 0.45f);
    if (!this->armor_yolo) {
        throw std::runtime_error("Failed to load armor yolo engine: " +
                                 armor_path);
    }
    TDT_INFO("Load armor_yolo engine success!");
    // 加载装甲板检测模型
    this->yolo = yolo::load(yolo_path, yolo::Type::V8, 0.65f, 0.45f);
    if (!this->yolo) {
        throw std::runtime_error("Failed to load yolo engine: " + yolo_path);
    }
    TDT_INFO("Load yolo engine success!");
    // 创建图像订阅者，调用回调
    image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "camera_image", rclcpp::SensorDataQoS(),
        std::bind(&Detect::callback, this, std::placeholders::_1));
    // 创建图像发布者
    pub = this->create_publisher<vision_interface::msg::DetectResult>(
        "detect_result", rclcpp::SensorDataQoS());
    RCLCPP_INFO(this->get_logger(), "Detect node has been started.");
}

void Detect::callback(const std::shared_ptr<sensor_msgs::msg::Image> msg)
{
    if (!yolo || !armor_yolo || !classifier) {
        RCLCPP_ERROR(this->get_logger(),
                     "Inference engines are not initialized.");
        return;
    }
    // 记录开始时间
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    std::cout << "Detecting..." << std::endl;
    // ros图像转opencv图像
    auto             img = cv_bridge::toCvShare(msg, "bgr8")->image;
    // opencv图像转推理输入格式
    tdt_radar::Image image(img.data, img.cols, img.rows);

    // yolo整车检测
    auto result = yolo->forward(image);
    // 初筛，防止车辆过多/过少
    if (result.size() == 0) {
        RCLCPP_INFO(this->get_logger(), "No Car!");
        return;
    } else if (result.size() > MAX_CARS) {
        RCLCPP_INFO(this->get_logger(), "Too Many Car!");
        return;
    }

    std::vector<tdt_radar::Image> images;
    std::vector<cv::Mat>          car_imgs;
    std::vector<Car>              cars;
    // 将yolo的输出转换成car类型
    for (auto& box : result) {
        if (box.class_label == 0 || box.class_label == 1) {
            Car car;
            car.car = box;
            cars.push_back(car);
        }
    }

    // 裁剪出每辆车的roi
    for (auto& car : cars) {
        auto     temp_rect = cv::Rect(car.car.left, car.car.top,
                                      car.car.right - car.car.left,
                                      car.car.bottom - car.car.top);
        cv::Rect temp_car_rect = getSafeRect(img, temp_rect);
        auto     car_img = img(temp_car_rect);
        car_imgs.push_back(car_img.clone());
        car.car_rect = temp_car_rect;
    }

    // 将每辆车的roi转换成推理输入格式
    for (auto& car_img : car_imgs) {
        auto image =
            tdt_radar::Image(car_img.data, car_img.cols, car_img.rows);
        images.push_back(image);
    }

    // yolo装甲板检测
    auto armor_boxes = armor_yolo->forwards(images);
    // 将每辆车对应装甲板存回car
    bool has_armor = false;
    for (int i = 0; i < armor_boxes.size(); i++) {
        if (armor_boxes[i].size() == 0) {
            continue;
        } else {
            cars[i].armors = armor_boxes[i];
            has_armor = true;
        }
    }  // 将armor_boxes存储到cars中
    if (!has_armor) {
        RCLCPP_INFO(this->get_logger(), "No Armor!");
        return;
    }

    // 保存装甲板图片
    // for(int i=0;i<cars.size();i++){
    //   if(cars[i].armors.size()==0){continue;}
    //   for(auto &armor:cars[i].armors){
    //     cv::Rect rect_img(
    //       armor.left+cars[i].car.left,
    //       armor.top+cars[i].car.top,
    //       armor.right-armor.left,
    //       armor.bottom-armor.top);
    //     // cv::rectangle(img,rect_img,cv::Scalar(255,255,255),2);
    //     cv::imwrite("/home/tdt/Label/armor_detect/"+std::to_string(count_img++)+".jpg",img(rect_img));
    //     }
    // }//这段是用来打印或者保存armor的图片

    // 分类阶段数据容器
    std::vector<cv::Mat>         armor_imgs;
    std::vector<classify::Image> armor_images;

    // 从每辆车的途中裁剪出装甲板roi
    for (auto& car : cars) {
        if (car.armors.size() == 0) {
            continue;
        }
        for (auto& armor : car.armors) {
            cv::Rect rect_img_1(
                armor.left + car.car.left, armor.top + car.car.top,
                armor.right - armor.left, armor.bottom - armor.top);
            cv::Rect rect_img = getSafeRect(img, rect_img_1);
            auto     armor_img = img(rect_img);
            armor_imgs.push_back(armor_img.clone());
        }
    }
    // 将装甲板roi转成分类器输入格式
    for (auto& armor_img : armor_imgs) {
        auto image =
            classify::Image(armor_img.data, armor_img.cols, armor_img.rows);
        armor_images.push_back(image);
    }
    // 分类
    auto armor_result = classifier->forwards(armor_images);

    // 保存用来训练分类
    //  for(int i=0;i<armor_imgs.size();i++){
    //    cv::imwrite("/home/mozihe/Label/fenqu_armor/"+std::to_string(armor_result[i])+"/"+std::to_string(count_img++)+".jpg",armor_imgs[i]);
    //  }

    // 把分类结果写回每个函数框
    for (auto& car : cars) {
        if (car.armors.size() == 0) {
            continue;
        }
        for (auto& armor : car.armors) {
            armor.class_label = armor_result[0];
            armor_result.erase(armor_result.begin());
            if (debug) {
                cv::putText(img, std::to_string(armor.class_label),
                            cv::Point(armor.left + car.car.left,
                                      armor.top + car.car.top),
                            cv::FONT_HERSHEY_SIMPLEX, 1,
                            cv::Scalar(255, 255, 255), 2);
            }
        }
    }
    // 创建要发布的ros消息
    vision_interface::msg::DetectResult detect_result;
    // 选择每辆车置信度最高的装甲板
    for (auto& car : cars) {
        if (car.armors.size() == 0) {
            continue;
        }
        cv::Rect max_rect;
        float    max_confidence = 0;
        for (auto& armor : car.armors) {
            if (armor.class_label != 0 && armor.class_label != 5 &&
                armor.confidence > max_confidence) {
                max_rect = cv::Rect(
                    armor.left + car.car.left, armor.top + car.car.top,
                    armor.right - armor.left, armor.bottom - armor.top);
                max_confidence = armor.confidence;
                car.number = armor.class_label;
            }
        }
        // 如没有装甲板画白框跳过
        if (max_confidence == 0) {
            if (debug)
                cv::rectangle(img, car.car_rect, cv::Scalar(255, 255, 255),
                              2);
            continue;
        }
        // 判断最优装甲板的颜色并计算中心点
        auto safe_rect = getSafeRect(img, max_rect);
        auto max_mat = img(safe_rect);

        car.color = getColor(max_mat);
        car.center = cv::Point2f(max_rect.x + max_rect.width / 2,
                                 max_rect.y + max_rect.height / 2);
        // 写入蓝车装甲板中心点
        if (car.color == 0) {
            detect_result.blue_x[car.number - 1] = car.center.x;
            detect_result.blue_y[car.number - 1] = car.center.y;
            if (car.center.x * car.center.y == 0 && car.number != 0) {
                RCLCPP_ERROR(this->get_logger(),
                             "Error: blue car center is 0");
            }
            if (debug) {
                cv::rectangle(img, car.car_rect, cv::Scalar(255, 0, 0), 2);
                cv::putText(img, std::to_string(car.car.confidence),
                            cv::Point(car.car.left, car.car.top),
                            cv::FONT_HERSHEY_SIMPLEX, 1,
                            cv::Scalar(255, 255, 255), 2);
            }
        }
        // 写入红车装甲板中心点
        if (car.color == 2) {
            detect_result.red_x[car.number - 1] = car.center.x;
            detect_result.red_y[car.number - 1] = car.center.y;
            if (car.center.x * car.center.y == 0 && car.number != 0) {
                RCLCPP_ERROR(this->get_logger(),
                             "Error: red car center is 0");
            }
            if (debug) {
                cv::rectangle(img, car.car_rect, cv::Scalar(0, 0, 255), 2);
                cv::putText(img, std::to_string(car.car.confidence),
                            cv::Point(car.car.left, car.car.top),
                            cv::FONT_HERSHEY_SIMPLEX, 1,
                            cv::Scalar(255, 255, 255), 2);
            }
        }
        // 如果颜色不为红蓝画白框
        if (car.color == 1) {
            if (debug)
                cv::rectangle(img, car.car_rect, cv::Scalar(255, 255, 255),
                              2);
        }
    }
    // 填时间戳
    detect_result.header.stamp = msg->header.stamp;
    // 发布结果
    pub->publish(detect_result);
    // 统计并打印总耗时
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                  begin);
    std::cout << "Detect Time: " << time_used.count() * 1000 << "ms"
              << std::endl;
    // 显示调试图像
    cv::Mat final_img;
    cv::resize(img, final_img, cv::Size(1536, 1125));
    cv::imshow("detect", final_img);
    // 按“r”切换debug模式
    auto key = cv::waitKey(1);
    if (key == 'r') {
        debug = !debug;
    }
}
}  // namespace tdt_radar
RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Detect)
