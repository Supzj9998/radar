cmake_minimum_required(VERSION 3.8)
project(kalman_filter)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
set(CMAKE_POLICY_DEFAULT_CMP0144 NEW)
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)
set(ENV{FLANN_ROOT} "")
set(ENV{PCL_ROOT} "")
unset(FLANN_ROOT)
unset(FLANN_ROOT CACHE)
unset(PCL_ROOT)
unset(PCL_ROOT CACHE)

find_package (ament_cmake_auto REQUIRED) 
ament_auto_find_build_dependencies ()

include_directories(include)

set(KALMAN_FILTER "kalman_filter")
file(GLOB_RECURSE KALMAN_FILTER_SRCS
    "src/*.cpp"
)

ament_auto_add_library(${KALMAN_FILTER} SHARED
    ${KALMAN_FILTER_SRCS}
)
target_link_libraries(${KALMAN_FILTER} ${PCL_LIBRARIES})

rclcpp_components_register_node(${KALMAN_FILTER}
    PLUGIN "tdt_radar::KalmanFilter"
    EXECUTABLE ${KALMAN_FILTER}_node
)



ament_auto_package(USE_SCOPED_HEADER_INSTALL_DIR)
