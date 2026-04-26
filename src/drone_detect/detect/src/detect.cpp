#include "detect.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuda_runtime_api.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/image_encodings.hpp"

// 调试窗口开关：注释掉该宏即可关闭 OpenCV 可视化窗口，适合无显示环境部署。
#define DRONE_DETECT_ENABLE_DEBUG_WINDOW

namespace drone::detect {
namespace {

// TensorRT 日志只输出 ERROR 及以上，避免每帧推理时刷屏。
class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kERROR) {
      std::fprintf(stderr, "[TensorRT] %s\n", msg);
    }
  }
};

TrtLogger g_trt_logger;

// 计算 TensorRT Dims 中所有维度的乘积，用于推导张量元素数量。
std::size_t volume(const nvinfer1::Dims& dims) {
  std::size_t total = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    total *= static_cast<std::size_t>(dims.d[i]);
  }
  return total;
}

// 将 TensorRT Dims 转成普通 vector，便于后续按 YOLO 常见布局解析。
std::vector<int64_t> dimsToVector(const nvinfer1::Dims& dims) {
  std::vector<int64_t> out;
  out.reserve(static_cast<std::size_t>(dims.nbDims));
  for (int i = 0; i < dims.nbDims; ++i) {
    out.push_back(static_cast<int64_t>(dims.d[i]));
  }
  return out;
}

}  // namespace

DetectNode::DetectNode(const rclcpp::NodeOptions& options) : Node("detect_node", options) {
  // 所有参数都提供默认值，launch 文件可以覆盖模型路径、话题名和阈值。
  image_topic_ = this->declare_parameter<std::string>("image_topic", "image_raw");
  output_image_topic_ =
      this->declare_parameter<std::string>("output_image_topic", "detect/image_with_boxes");
  output_boxes_topic_ =
      this->declare_parameter<std::string>("output_boxes_topic", "detect/boxes");
  model_path_ = this->declare_parameter<std::string>("model_path", "model/ONNX/best.onnx");
  engine_path_ =
      this->declare_parameter<std::string>("engine_path", "model/TensorRT/best.engine");
  input_tensor_name_ = this->declare_parameter<std::string>("input_tensor_name", "images");
  input_width_ = this->declare_parameter<int>("input_width", 640);
  input_height_ = this->declare_parameter<int>("input_height", 640);
  conf_threshold_ = this->declare_parameter<double>("conf_threshold", 0.25);
  nms_threshold_ = this->declare_parameter<double>("nms_threshold", 0.45);
  yolo_has_objectness_ = this->declare_parameter<bool>("yolo_has_objectness", true);

  image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(output_image_topic_, 10);
  boxes_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(output_boxes_topic_, 10);

  // 图像使用 SensorDataQoS，降低实时图像流在队列中堆积的概率。
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&DetectNode::imageCallback, this, std::placeholders::_1));

  if (model_path_.empty()) {
    RCLCPP_WARN(this->get_logger(),
                "Parameter 'model_path' is empty. Detection will be skipped until a valid ONNX "
                "path is provided.");
    return;
  }

  if (!ensureEngineAvailable()) {
    RCLCPP_ERROR(this->get_logger(), "TensorRT engine is unavailable.");
    return;
  }

  if (!loadEngine()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load TensorRT engine: %s", engine_path_.c_str());
    return;
  }

  model_ready_ = true;
  RCLCPP_INFO(this->get_logger(), "Loaded TensorRT engine from: %s", engine_path_.c_str());
}

DetectNode::~DetectNode() {
  // 释放顺序保持 CUDA 流、输入缓冲、输出缓冲、窗口资源逐项清理。
  if (cuda_stream_ != nullptr) {
    cudaStreamDestroy(static_cast<cudaStream_t>(cuda_stream_));
    cuda_stream_ = nullptr;
  }
  if (input_buffer_device_ != nullptr) {
    cudaFree(input_buffer_device_);
    input_buffer_device_ = nullptr;
  }
  if (input_buffer_host_ != nullptr) {
    cudaFreeHost(input_buffer_host_);
    input_buffer_host_ = nullptr;
  }
  for (auto& binding : output_bindings_) {
    if (binding.device_buffer != nullptr) {
      cudaFree(binding.device_buffer);
      binding.device_buffer = nullptr;
    }
    if (binding.host_buffer != nullptr) {
      cudaFreeHost(binding.host_buffer);
      binding.host_buffer = nullptr;
    }
  }
  if (debug_window_created_) {
    cv::destroyWindow(debug_window_name_);
    debug_window_created_ = false;
  }
}

bool DetectNode::ensureEngineAvailable() {
  const std::filesystem::path engine_file_path(engine_path_);
  // engine 输出目录不存在时自动创建，方便首次部署直接由 ONNX 构建。
  if (engine_file_path.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(engine_file_path.parent_path(), ec);
  }

  std::error_code ec;
  if (std::filesystem::exists(engine_file_path, ec) &&
      std::filesystem::is_regular_file(engine_file_path, ec) &&
      std::filesystem::file_size(engine_file_path, ec) > 0) {
    return true;
  }

  const std::string shape = std::to_string(input_height_) + "x" + std::to_string(input_width_);
  // 先按固定 batch=1 的动态 shape 构建；如果模型不接受 shape 参数，再回退到静态构建。
  const std::string dynamic_command =
      "/usr/src/tensorrt/bin/trtexec --onnx=" + model_path_ + " --saveEngine=" + engine_path_ +
      " --minShapes=" + input_tensor_name_ + ":1x3x" + shape + " --optShapes=" +
      input_tensor_name_ + ":1x3x" + shape + " --maxShapes=" + input_tensor_name_ + ":1x3x" +
      shape + " --skipInference";
  const std::string static_command = "/usr/src/tensorrt/bin/trtexec --onnx=" + model_path_ +
                                     " --saveEngine=" + engine_path_ + " --skipInference";

  RCLCPP_WARN(this->get_logger(), "TensorRT engine not found. Building engine with: %s",
              dynamic_command.c_str());
  int ret = std::system(dynamic_command.c_str());
  if (ret != 0) {
    RCLCPP_WARN(this->get_logger(),
                "Dynamic-shape TensorRT build failed, retrying static-shape build: %s",
                static_command.c_str());
    ret = std::system(static_command.c_str());
    if (ret != 0) {
      std::filesystem::remove(engine_file_path, ec);
      RCLCPP_ERROR(this->get_logger(), "TensorRT engine build failed with exit code %d", ret);
      return false;
    }
  }

  return std::filesystem::exists(engine_file_path, ec) &&
         std::filesystem::is_regular_file(engine_file_path, ec) &&
         std::filesystem::file_size(engine_file_path, ec) > 0;
}

bool DetectNode::loadEngine() {
  // TensorRT engine 是二进制序列化文件，需要完整读入内存后反序列化。
  std::ifstream file(engine_path_, std::ios::binary);
  if (!file) {
    return false;
  }
  file.seekg(0, std::ios::end);
  const std::size_t size = static_cast<std::size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  std::vector<char> serialized_engine(size);
  file.read(serialized_engine.data(), static_cast<std::streamsize>(size));

  runtime_.reset(nvinfer1::createInferRuntime(g_trt_logger));
  if (!runtime_) {
    return false;
  }
  engine_.reset(runtime_->deserializeCudaEngine(serialized_engine.data(), size));
  if (!engine_) {
    return false;
  }
  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    return false;
  }

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    return false;
  }
  cuda_stream_ = stream;

  // 自动枚举所有输出张量，避免强依赖具体 YOLO 导出时的输出名称。
  const int tensor_count = engine_->getNbIOTensors();
  output_bindings_.clear();
  for (int i = 0; i < tensor_count; ++i) {
    const char* tensor_name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kOUTPUT) {
      output_bindings_.push_back(OutputBinding{tensor_name});
    }
  }
  if (output_bindings_.empty()) {
    return false;
  }

  if (!context_->setInputShape(input_tensor_name_.c_str(),
                               nvinfer1::Dims4(1, 3, input_height_, input_width_))) {
    return false;
  }

  // 输入输出缓冲区在 load 阶段一次性分配，推理时只复用内存减少实时开销。
  const auto input_dims = context_->getTensorShape(input_tensor_name_.c_str());
  input_buffer_bytes_ = volume(input_dims) * sizeof(float);
  if (cudaMalloc(&input_buffer_device_, input_buffer_bytes_) != cudaSuccess ||
      cudaMallocHost(&input_buffer_host_, input_buffer_bytes_) != cudaSuccess) {
    return false;
  }

  for (auto& binding : output_bindings_) {
    const auto output_dims = context_->getTensorShape(binding.name.c_str());
    binding.buffer_bytes = volume(output_dims) * sizeof(float);
    if (cudaMalloc(&binding.device_buffer, binding.buffer_bytes) != cudaSuccess ||
        cudaMallocHost(&binding.host_buffer, binding.buffer_bytes) != cudaSuccess) {
      return false;
    }
  }

  return true;
}

void DetectNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                          "cv_bridge conversion failed: %s", e.what());
    return;
  }

  cv::Mat annotated = cv_ptr->image.clone();
  std::vector<Detection> detections;

  // 推理异常通常说明 engine、CUDA 或输入 shape 有问题，置为不可用以避免持续报错。
  if (model_ready_) {
    try {
      detections = infer(cv_ptr->image);
      drawDetections(annotated, detections);
    } catch (const std::exception& e) {
      model_ready_ = false;
      RCLCPP_ERROR(this->get_logger(), "TensorRT inference failed: %s", e.what());
    }
  }

  // 发布给 PnP 的格式为每个目标 10 个 float:
  // class_id, score, x1, y1, x2, y2, x3, y3, x4, y4。
  std_msgs::msg::Float32MultiArray boxes_msg;
  boxes_msg.data.reserve(detections.size() * 10U);
  for (const auto& det : detections) {
    const float x1 = static_cast<float>(det.box.x);
    const float y1 = static_cast<float>(det.box.y);
    const float x2 = static_cast<float>(det.box.x + det.box.width);
    const float y2 = static_cast<float>(det.box.y);
    const float x3 = static_cast<float>(det.box.x + det.box.width);
    const float y3 = static_cast<float>(det.box.y + det.box.height);
    const float x4 = static_cast<float>(det.box.x);
    const float y4 = static_cast<float>(det.box.y + det.box.height);
    boxes_msg.data.push_back(static_cast<float>(det.class_id));
    boxes_msg.data.push_back(det.score);
    boxes_msg.data.push_back(x1);
    boxes_msg.data.push_back(y1);
    boxes_msg.data.push_back(x2);
    boxes_msg.data.push_back(y2);
    boxes_msg.data.push_back(x3);
    boxes_msg.data.push_back(y3);
    boxes_msg.data.push_back(x4);
    boxes_msg.data.push_back(y4);
  }
  boxes_pub_->publish(boxes_msg);

  image_pub_->publish(*makeImageMsg(annotated, msg->header));
  showDebugWindow(annotated);
}

std::vector<Detection> DetectNode::infer(const cv::Mat& image) const {
  // blobFromImage 输出 NCHW float blob，与 TensorRT 输入 shape 保持一致。
  cv::Mat blob = preprocess(image);
  const std::size_t tensor_size = static_cast<std::size_t>(blob.total()) * sizeof(float);
  std::memcpy(input_buffer_host_, blob.ptr<float>(), tensor_size);

  auto stream = static_cast<cudaStream_t>(cuda_stream_);
  // H2D、推理、D2H 共用同一 CUDA stream，最后同步保证 host 端输出可读。
  if (cudaMemcpyAsync(input_buffer_device_, input_buffer_host_, input_buffer_bytes_,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    throw std::runtime_error("Failed to copy input to device.");
  }

  if (!context_->setTensorAddress(input_tensor_name_.c_str(), input_buffer_device_)) {
    throw std::runtime_error("Failed to bind TensorRT tensor addresses.");
  }
  // TensorRT 10 的 enqueueV3 需要显式绑定每个输入/输出张量地址。
  for (const auto& binding : output_bindings_) {
    if (!context_->setTensorAddress(binding.name.c_str(), binding.device_buffer)) {
      throw std::runtime_error("Failed to bind TensorRT output tensor addresses.");
    }
  }

  if (!context_->enqueueV3(stream)) {
    throw std::runtime_error("TensorRT enqueueV3 failed.");
  }

  for (const auto& binding : output_bindings_) {
    if (cudaMemcpyAsync(binding.host_buffer, binding.device_buffer, binding.buffer_bytes,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
      throw std::runtime_error("Failed to copy output to host.");
    }
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("Failed to synchronize CUDA stream.");
  }

  std::vector<Detection> detections;
  for (const auto& binding : output_bindings_) {
    // 每个输出张量先独立解码，最后统一 NMS。
    const auto output_dims = context_->getTensorShape(binding.name.c_str());
    const auto output_shape = dimsToVector(output_dims);
    const auto* output_data = static_cast<const float*>(binding.host_buffer);
    std::vector<float> output(output_data, output_data + volume(output_dims));
    auto partial = decodeYoloOutput(output, output_shape, image.size());
    detections.insert(detections.end(), partial.begin(), partial.end());
  }

  return applyGlobalNms(detections);
}

cv::Mat DetectNode::preprocess(const cv::Mat& image) const {
  cv::Mat blob;
  // swapRB=true 将 OpenCV BGR 转为常见 YOLO 训练使用的 RGB，并归一化到 [0,1]。
  cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(input_width_, input_height_),
                         cv::Scalar(), true, false);
  return blob;
}

std::vector<Detection> DetectNode::decodeYoloOutput(const std::vector<float>& data,
                                                    const std::vector<int64_t>& shape,
                                                    const cv::Size& image_size) const {
  if (data.empty() || shape.empty()) {
    return {};
  }

  int64_t num_preds = 0;
  int64_t attrs = 0;
  // 不同 YOLO 导出路径可能给出 [N, attrs]、[attrs, N] 或带 batch/HW 的输出。
  enum class TensorLayout {
    kPredsAttrs,
    kAttrsPreds,
    kOnePredsAttrs,
    kAttrsPredsOne,
    kAttrsHW,
    kHWAttrs
  };
  TensorLayout layout = TensorLayout::kPredsAttrs;

  if (shape.size() == 2) {
    num_preds = shape[0];
    attrs = shape[1];
  } else if (shape.size() == 3) {
    if (shape[0] != 1) {
      return {};
    }
    if (shape[1] <= 256 && shape[2] > shape[1]) {
      attrs = shape[1];
      num_preds = shape[2];
      layout = TensorLayout::kAttrsPreds;
    } else {
      num_preds = shape[1];
      attrs = shape[2];
    }
  } else if (shape.size() == 4) {
    if (shape[0] != 1) {
      return {};
    }
    if (shape[1] > 1 && shape[1] <= 256 && shape[2] > 1 && shape[3] > 1) {
      attrs = shape[1];
      num_preds = shape[2] * shape[3];
      layout = TensorLayout::kAttrsHW;
    } else if (shape[3] > 1 && shape[3] <= 256 && shape[1] > 1 && shape[2] > 1) {
      attrs = shape[3];
      num_preds = shape[1] * shape[2];
      layout = TensorLayout::kHWAttrs;
    } else if (shape[1] == 1) {
      num_preds = shape[2];
      attrs = shape[3];
      layout = TensorLayout::kOnePredsAttrs;
    } else if (shape[3] == 1) {
      attrs = shape[1];
      num_preds = shape[2];
      layout = TensorLayout::kAttrsPredsOne;
    } else {
      return {};
    }
  } else {
    return {};
  }

  if (num_preds <= 0 || attrs < 5) {
    return {};
  }

  // 统一访问不同布局中的第 pred_idx 个预测、第 attr_idx 个属性。
  auto value_at = [&](int64_t pred_idx, int64_t attr_idx) -> float {
    switch (layout) {
      case TensorLayout::kPredsAttrs:
        return data[static_cast<std::size_t>(pred_idx * attrs + attr_idx)];
      case TensorLayout::kAttrsPreds:
        return data[static_cast<std::size_t>(attr_idx * num_preds + pred_idx)];
      case TensorLayout::kOnePredsAttrs:
        return data[static_cast<std::size_t>(pred_idx * attrs + attr_idx)];
      case TensorLayout::kAttrsPredsOne:
        return data[static_cast<std::size_t>(attr_idx * num_preds + pred_idx)];
      case TensorLayout::kAttrsHW:
        return data[static_cast<std::size_t>(attr_idx * num_preds + pred_idx)];
      case TensorLayout::kHWAttrs:
        return data[static_cast<std::size_t>(pred_idx * attrs + attr_idx)];
    }
    return 0.0F;
  };

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  class_ids.reserve(static_cast<std::size_t>(num_preds));
  confidences.reserve(static_cast<std::size_t>(num_preds));
  boxes.reserve(static_cast<std::size_t>(num_preds));

  const float scale_x = static_cast<float>(image_size.width) / static_cast<float>(input_width_);
  const float scale_y = static_cast<float>(image_size.height) / static_cast<float>(input_height_);

  for (int64_t i = 0; i < num_preds; ++i) {
    // 默认 YOLO 输出前四项为中心点 cx/cy 与宽高 w/h，坐标相对于网络输入尺寸。
    const float cx = value_at(i, 0);
    const float cy = value_at(i, 1);
    const float w = value_at(i, 2);
    const float h = value_at(i, 3);

    int class_id = 0;
    float confidence = 0.0F;

    if (attrs == 5) {
      confidence = value_at(i, 4);
    } else {
      // 部分模型输出 objectness * class_score，部分模型没有 objectness，需要参数控制。
      const float objectness = yolo_has_objectness_ ? value_at(i, 4) : 1.0F;
      const int cls_start = yolo_has_objectness_ ? 5 : 4;
      if (cls_start >= attrs) {
        continue;
      }
      float best_cls = -std::numeric_limits<float>::infinity();
      for (int64_t c = cls_start; c < attrs; ++c) {
        const float cls_score = value_at(i, c);
        if (cls_score > best_cls) {
          best_cls = cls_score;
          class_id = static_cast<int>(c - cls_start);
        }
      }
      confidence = objectness * best_cls;
    }

    if (confidence < conf_threshold_) {
      continue;
    }

    // 将网络输入尺寸上的框缩放回原图，并裁剪到图像边界内。
    int left = static_cast<int>((cx - 0.5F * w) * scale_x);
    int top = static_cast<int>((cy - 0.5F * h) * scale_y);
    int width = static_cast<int>(w * scale_x);
    int height = static_cast<int>(h * scale_y);

    left = std::clamp(left, 0, image_size.width - 1);
    top = std::clamp(top, 0, image_size.height - 1);
    width = std::clamp(width, 1, image_size.width - left);
    height = std::clamp(height, 1, image_size.height - top);

    boxes.emplace_back(left, top, width, height);
    confidences.push_back(confidence);
    class_ids.push_back(class_id);
  }

  std::vector<int> keep_indices;
  // 单个输出张量内部先做 NMS，减少后续跨输出合并的候选数量。
  cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, keep_indices);

  std::vector<Detection> detections;
  detections.reserve(keep_indices.size());
  for (const int idx : keep_indices) {
    detections.push_back(Detection{boxes[idx], class_ids[idx], confidences[idx]});
  }
  return detections;
}

std::vector<Detection> DetectNode::applyGlobalNms(const std::vector<Detection>& detections) const {
  if (detections.empty()) {
    return {};
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  boxes.reserve(detections.size());
  confidences.reserve(detections.size());
  for (const auto& det : detections) {
    boxes.push_back(det.box);
    confidences.push_back(det.score);
  }

  std::vector<int> keep_indices;
  cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, keep_indices);

  std::vector<Detection> kept;
  kept.reserve(keep_indices.size());
  for (const int idx : keep_indices) {
    kept.push_back(detections[static_cast<std::size_t>(idx)]);
  }
  return kept;
}

sensor_msgs::msg::Image::SharedPtr DetectNode::makeImageMsg(const cv::Mat& image,
                                                            const std_msgs::msg::Header& header) const {
  return cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, image).toImageMsg();
}

void DetectNode::drawDetections(cv::Mat& image, const std::vector<Detection>& detections) const {
  for (const auto& det : detections) {
    // 绿色框和文字只用于输出图像/调试窗口，不影响下游 PnP 数据。
    cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
    const std::string label =
        "id:" + std::to_string(det.class_id) + " conf:" + std::to_string(det.score);
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int top = std::max(det.box.y, text_size.height + 4);
    cv::rectangle(image,
                  cv::Point(det.box.x, top - text_size.height - 4),
                  cv::Point(det.box.x + text_size.width, top + baseline - 4),
                  cv::Scalar(0, 255, 0), cv::FILLED);
    cv::putText(image, label, cv::Point(det.box.x, top - 4), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1);
  }
}

void DetectNode::showDebugWindow(const cv::Mat& image) {
#ifndef DRONE_DETECT_ENABLE_DEBUG_WINDOW
  (void)image;
  return;
#else
  if (!debug_window_created_) {
    cv::namedWindow(debug_window_name_, cv::WINDOW_NORMAL);
    debug_window_created_ = true;
  }
  cv::imshow(debug_window_name_, image);
  cv::waitKey(1);
#endif
}

}  // namespace drone::detect

RCLCPP_COMPONENTS_REGISTER_NODE(drone::detect::DetectNode)
