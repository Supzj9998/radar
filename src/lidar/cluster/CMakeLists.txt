cmake_minimum_required(VERSION 3.8)
project(cluster)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1)
set(CMAKE_BUILD_TYPE "RELEASE")
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

set(CLUSTER_NODE "cluster")
file(GLOB_RECURSE CLUSTER_NODE_SRCS "src/*.cpp")

ament_auto_add_library(${CLUSTER_NODE} SHARED
    ${CLUSTER_NODE_SRCS}
)

target_link_libraries(${CLUSTER_NODE} ${PCL_LIBRARIES})

rclcpp_components_register_node(${CLUSTER_NODE}
    PLUGIN "tdt_radar::Cluster"
    EXECUTABLE ${CLUSTER_NODE}_node
)
ament_auto_package(USE_SCOPED_HEADER_INSTALL_DIR)
