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

struct Detection {
  cv::Rect box;
  int class_id;
  float score;
};

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
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
  bool ensureEngineAvailable();
  bool loadEngine();
  std::vector<Detection> infer(const cv::Mat& image) const;
  cv::Mat preprocess(const cv::Mat& image) const;
  std::vector<Detection> decodeYoloOutput(const std::vector<float>& data,
                                          const std::vector<int64_t>& shape,
                                          const cv::Size& image_size) const;
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
  void* input_buffer_device_{nullptr};
  void* output_buffer_device_{nullptr};
  mutable void* input_buffer_host_{nullptr};
  mutable void* output_buffer_host_{nullptr};
  std::size_t input_buffer_bytes_{0};
  std::size_t output_buffer_bytes_{0};
  void* cuda_stream_{nullptr};
  bool model_ready_{false};

  std::string image_topic_;
  std::string output_image_topic_;
  std::string output_boxes_topic_;
  std::string model_path_;
  std::string engine_path_;
  std::string input_tensor_name_{"images"};
  std::string output_tensor_name_;
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
