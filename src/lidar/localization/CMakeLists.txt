cmake_minimum_required(VERSION 3.8)
project(localization)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1)
set(CMAKE_BUILD_TYPE "RELEASE") 
#gicp配准需要release编译
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

set(LOCALIZATION_NODE "localization")
file(GLOB_RECURSE LOCALIZATION_NODE_SRCS "src/*.cpp")

ament_auto_add_library(${LOCALIZATION_NODE} SHARED
    ${LOCALIZATION_NODE_SRCS}
)

target_link_libraries(${LOCALIZATION_NODE} ${PCL_LIBRARIES})

rclcpp_components_register_node(${LOCALIZATION_NODE}
    PLUGIN "tdt_radar::Localization"
    EXECUTABLE ${LOCALIZATION_NODE}_node
)
target_link_libraries(${LOCALIZATION_NODE} ${PCL_LIBRARIES})

ament_auto_package(USE_SCOPED_HEADER_INSTALL_DIR)
