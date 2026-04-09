#include <filesystem>
#include <stdexcept>
#include <thread>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/detail/image__struct.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vision_interface/msg/match_info.hpp>

using namespace std::chrono_literals;

class RosbagPlayer : public rclcpp::Node {
public:
    RosbagPlayer(const rclcpp::NodeOptions& options)
        : Node("rosbag_player_node", options)
    {
        this->declare_parameter<std::string>("rosbag_file", "");
        this->get_parameter("rosbag_file", rosbag_file);

        pointcloud_publisher_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>(
                "/livox/lidar", 10);
        image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "camera_image", rclcpp::SensorDataQoS());
        match_info_publisher_ =
            this->create_publisher<vision_interface::msg::MatchInfo>(
                "/match_info", 10);
        open_bag();

        processing_thread_ =
            std::make_shared<std::thread>(&RosbagPlayer::play_bag, this);
    }

    ~RosbagPlayer()
    {
        if (processing_thread_ && processing_thread_->joinable()) {
            processing_thread_->join();
        }
    }

private:
    void open_bag()
    {
        namespace fs = std::filesystem;

        if (rosbag_file.empty()) {
            throw std::invalid_argument("Parameter 'rosbag_file' is empty.");
        }

        const fs::path bag_path(rosbag_file);
        if (!fs::exists(bag_path)) {
            throw std::invalid_argument(
                "Rosbag path does not exist: " + rosbag_file);
        }

        rosbag2_storage::StorageOptions storage_options;
        storage_options.uri = rosbag_file;

        if (fs::is_regular_file(bag_path)) {
            const auto extension = bag_path.extension().string();
            if (extension == ".db3") {
                storage_options.storage_id = "sqlite3";
            }
            else if (extension == ".mcap") {
                storage_options.storage_id = "mcap";
            }
            else {
                throw std::invalid_argument(
                    "Unsupported rosbag file extension for path: " +
                    rosbag_file);
            }
        }
        else if (!fs::is_directory(bag_path)) {
            throw std::invalid_argument(
                "Rosbag path must be a directory, .db3 file, or .mcap file: " +
                rosbag_file);
        }

        try {
            reader_.open(storage_options);
            RCLCPP_INFO(this->get_logger(), "Opened rosbag: %s",
                        rosbag_file.c_str());
        }
        catch (const std::exception& e) {
            throw std::runtime_error(
                "Failed to open rosbag '" + rosbag_file + "': " + e.what());
        }
    }

    void play_bag()
    {
        while (rclcpp::ok()) {
            if (!reader_.has_next())
                open_bag();

            auto start_time = std::chrono::high_resolution_clock::now();
            auto bag_message = reader_.read_next();
            auto ros_time = rclcpp::Clock().now();

            if (bag_message->topic_name == "/livox/lidar") {
                auto pointcloud_msg =
                    std::make_shared<sensor_msgs::msg::PointCloud2>();
                rclcpp::Serialization<sensor_msgs::msg::PointCloud2>
                                          serialization;
                rclcpp::SerializedMessage serialized_msg(
                    *bag_message->serialized_data);
                serialization.deserialize_message(&serialized_msg,
                                                  pointcloud_msg.get());
                pointcloud_msg->header.stamp = ros_time;
                pointcloud_publisher_->publish(*pointcloud_msg);
            }
            else if (bag_message->topic_name == "/compressed_image") {
                auto image_msg =
                    std::make_shared<sensor_msgs::msg::CompressedImage>();
                rclcpp::Serialization<sensor_msgs::msg::CompressedImage>
                                          serialization;
                rclcpp::SerializedMessage serialized_msg(
                    *bag_message->serialized_data);
                serialization.deserialize_message(&serialized_msg,
                                                  image_msg.get());
                auto img = cv::imdecode(image_msg->data, cv::IMREAD_COLOR);
                auto msg =
                    cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img)
                        .toImageMsg();
                msg->header.stamp = ros_time;
                image_publisher_->publish(*msg);
            }
            else if (bag_message->topic_name == "/match_info") {
                auto match_info_msg =
                    std::make_shared<vision_interface::msg::MatchInfo>();
                rclcpp::Serialization<vision_interface::msg::MatchInfo>
                                          serialization;
                rclcpp::SerializedMessage serialized_msg(
                    *bag_message->serialized_data);
                serialization.deserialize_message(&serialized_msg,
                                                  match_info_msg.get());
                match_info_publisher_->publish(*match_info_msg);
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time)
                    .count();
            if ((duration < 100) && (duration > 1)) {
                std::this_thread::sleep_for(
                    100ms - std::chrono::milliseconds(duration));
            }
        }
        RCLCPP_INFO(this->get_logger(), "No more messages in the bag.");
        rclcpp::shutdown();
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        pointcloud_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    rclcpp::Publisher<vision_interface::msg::MatchInfo>::SharedPtr
                                 match_info_publisher_;
    rosbag2_cpp::Reader          reader_;
    std::shared_ptr<std::thread> processing_thread_;
    std::string                  rosbag_file;
};

RCLCPP_COMPONENTS_REGISTER_NODE(RosbagPlayer)
