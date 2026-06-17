#ifndef DRONE_DETECT_AUTOAIM_MANAGER_H
#define DRONE_DETECT_AUTOAIM_MANAGER_H

#include "gary_msgs/msg/auto_aim.hpp"
#include "rclcpp/rclcpp.hpp"

namespace drone::autoaim_manager {

class AutoaimManagerNode final : public rclcpp::Node {
public:
    explicit AutoaimManagerNode(const rclcpp::NodeOptions& options);

private:
    void droneCallback(const gary_msgs::msg::AutoAIM::SharedPtr msg);
    void modelCallback(const gary_msgs::msg::AutoAIM::SharedPtr msg);
    void statusCallback(const gary_msgs::msg::AutoAIM::SharedPtr msg);
    void publishSelectedTarget();

    rclcpp::Subscription<gary_msgs::msg::AutoAIM>::SharedPtr drone_sub_;
    rclcpp::Subscription<gary_msgs::msg::AutoAIM>::SharedPtr model_sub_;
    rclcpp::Subscription<gary_msgs::msg::AutoAIM>::SharedPtr status_sub_;
    rclcpp::Publisher<gary_msgs::msg::AutoAIM>::SharedPtr target_pub_;

    gary_msgs::msg::AutoAIM latest_drone_;
    gary_msgs::msg::AutoAIM latest_model_;
    gary_msgs::msg::AutoAIM latest_status_;
    rclcpp::Time latest_drone_time_;
    rclcpp::Time latest_model_time_;
    rclcpp::Time latest_status_time_;
    bool has_drone_ = false;
    bool has_model_ = false;
    bool has_status_ = false;
};

}  // namespace drone::autoaim_manager

#endif  // DRONE_DETECT_AUTOAIM_MANAGER_H
