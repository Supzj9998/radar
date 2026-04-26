#ifndef DRONE_DETECT__DETECT_H_
#define DRONE_DETECT__DETECT_H_

#include <memory>
#include <string>
#include <cstdint>
#include <vector>

#include "NvInfer.h"
#include "cv_bridge/cv_bridge.hpp"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

namespace drone::detect {

// 单个目标检测结果。box 使用原始图像坐标系下的左上角和宽高。
struct Detection {
  cv::Rect box;
  int class_id;
  float score;
};

// TensorRT 输出张量对应的设备端/主机端缓冲区。
// host_buffer 设为 mutable，是为了允许 const 推理接口复用已分配的页锁定内存。
struct OutputBinding {
  std::string name;
  void* device_buffer{nullptr};
  mutable void* host_buffer{nullptr};
  std::size_t buffer_bytes{0};
};

// TensorRT 对象使用自定义 deleter 交给 unique_ptr 管理，避免构造失败时泄漏资源。
struct TrtDestroy {
  template <typename T>
  void operator()(T* ptr) const {
    if (ptr != nullptr) {
      delete ptr;
    }
  }
};

class DetectNode : public rclcpp::Node {
 public:
  explicit DetectNode(const rclcpp::NodeOptions& options);
  ~DetectNode() override;

 private:
  // ROS 图像入口：完成 BGR 转换、推理、可视化以及检测框发布。
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

  // 确保 engine_path_ 指向可用 TensorRT engine；不存在时尝试由 ONNX 自动构建。
  bool ensureEngineAvailable();

  // 反序列化 TensorRT engine，并为输入输出张量分配 CUDA/device 与 pinned host 缓冲。
  bool loadEngine();

  // 执行完整推理流程：预处理、H2D、enqueue、D2H、YOLO 解码、全局 NMS。
  std::vector<Detection> infer(const cv::Mat& image) const;

  // 将 OpenCV BGR 图像转换为 TensorRT 输入使用的 NCHW float blob。
  cv::Mat preprocess(const cv::Mat& image) const;

  // 兼容常见 YOLO 输出布局，将网络输出还原为原图坐标系检测框。
  std::vector<Detection> decodeYoloOutput(const std::vector<float>& data,
                                          const std::vector<int64_t>& shape,
                                          const cv::Size& image_size) const;

  // 多输出张量场景下再做一次全局 NMS，避免不同输出头产生重复框。
  std::vector<Detection> applyGlobalNms(const std::vector<Detection>& detections) const;

  // 将标注后的 OpenCV 图像重新封装为 ROS Image 消息，并沿用输入 header。
  sensor_msgs::msg::Image::SharedPtr makeImageMsg(const cv::Mat& image,
                                                  const std_msgs::msg::Header& header) const;
  void drawDetections(cv::Mat& image, const std::vector<Detection>& detections) const;
  void showDebugWindow(const cv::Mat& image);

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr boxes_pub_;

  std::unique_ptr<nvinfer1::IRuntime, TrtDestroy> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy> context_;

  // 输入张量缓冲区：host 端为 pinned memory，便于异步拷贝到 GPU。
  void* input_buffer_device_{nullptr};
  mutable void* input_buffer_host_{nullptr};
  std::size_t input_buffer_bytes_{0};

  // 允许一个 engine 暴露多个输出张量，每个输出单独维护绑定信息。
  std::vector<OutputBinding> output_bindings_;
  void* cuda_stream_{nullptr};
  bool model_ready_{false};

  // 运行参数由 launch 或 ROS 参数服务器配置。
  std::string image_topic_;
  std::string output_image_topic_;
  std::string output_boxes_topic_;
  std::string model_path_;
  std::string engine_path_;
  std::string input_tensor_name_{"images"};
  int input_width_{640};
  int input_height_{640};
  float conf_threshold_{0.25F};
  float nms_threshold_{0.45F};
  bool yolo_has_objectness_{true};
  std::string debug_window_name_{"detect_debug"};
  bool debug_window_created_{false};
};

}  // namespace drone::detect

#endif  // DRONE_DETECT__DETECT_H_
