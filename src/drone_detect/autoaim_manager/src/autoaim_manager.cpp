#include "autoaim_manager.h"

#include <functional>

#include "rclcpp_components/register_node_macro.hpp"

namespace drone::autoaim_manager {
namespace {
    constexpr const char* kDroneTopic = "/autoaim/drone";
    constexpr const char* kModelTopic = "/autoaim/model";
    constexpr const char* kStatusTopic = "/autoaim/status";
    constexpr const char* kTargetTopic = "/autoaim/target";
    constexpr double kMessageTimeoutSec = 0.2;
}  // namespace

AutoaimManagerNode::AutoaimManagerNode(const rclcpp::NodeOptions& options)
    : Node("autoaim_manager_node", options)
{
    drone_sub_ = create_subscription<gary_msgs::msg::AutoAIM>(
        kDroneTopic, rclcpp::SensorDataQoS(),
        std::bind(&AutoaimManagerNode::droneCallback, this,
                  std::placeholders::_1));
    model_sub_ = create_subscription<gary_msgs::msg::AutoAIM>(
        kModelTopic, rclcpp::SensorDataQoS(),
        std::bind(&AutoaimManagerNode::modelCallback, this,
                  std::placeholders::_1));
    status_sub_ = create_subscription<gary_msgs::msg::AutoAIM>(
        kStatusTopic, rclcpp::SensorDataQoS(),
        std::bind(&AutoaimManagerNode::statusCallback, this,
                  std::placeholders::_1));
    target_pub_ =
        create_publisher<gary_msgs::msg::AutoAIM>(kTargetTopic, 10);
}

void AutoaimManagerNode::droneCallback(
    const gary_msgs::msg::AutoAIM::SharedPtr msg)
{
    latest_drone_ = *msg;
    latest_drone_time_ = now();
    has_drone_ = true;
    publishSelectedTarget();
}

void AutoaimManagerNode::modelCallback(
    const gary_msgs::msg::AutoAIM::SharedPtr msg)
{
    latest_model_ = *msg;
    latest_model_time_ = now();
    has_model_ = true;
    publishSelectedTarget();
}

void AutoaimManagerNode::statusCallback(
    const gary_msgs::msg::AutoAIM::SharedPtr msg)
{
    latest_status_ = *msg;
    latest_status_time_ = now();
    has_status_ = true;
    publishSelectedTarget();
}

void AutoaimManagerNode::publishSelectedTarget()
{
    const auto current_time = now();
    const auto message_timeout = rclcpp::Duration::from_seconds(kMessageTimeoutSec);
    const bool model_fresh = has_model_ && current_time - latest_model_time_ <= message_timeout;
    const bool drone_fresh = has_drone_ && current_time - latest_drone_time_ <= message_timeout;
    const bool status_fresh = has_status_ && current_time - latest_status_time_ <= message_timeout;

    if (model_fresh) {
        target_pub_->publish(latest_model_);
    }
    else if (drone_fresh) {
        target_pub_->publish(latest_drone_);
    }
    else if (status_fresh) {
        target_pub_->publish(latest_status_);
    }
}

}  // namespace drone::autoaim_manager

RCLCPP_COMPONENTS_REGISTER_NODE(drone::autoaim_manager::AutoaimManagerNode)
