cmake_minimum_required(VERSION 3.8)
project(drone_detect)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
set(CMAKE_POLICY_DEFAULT_CMP0144 NEW)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS /usr/include/x86_64-linux-gnu /usr/include
)
find_library(TENSORRT_LIBRARY
  NAMES nvinfer
  PATHS /usr/lib/x86_64-linux-gnu /usr/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT not found. Expected NvInfer.h and libnvinfer on this machine.")
endif()

include_directories(
  "detect/include"
  "pnp/include"
  ${TENSORRT_INCLUDE_DIR}
)

set(DRONE_DETECT "drone_detect")
file(GLOB_RECURSE DRONE_DETECT_SRCS "detect/src/*.cpp")

ament_auto_add_library(${DRONE_DETECT} SHARED
  ${DRONE_DETECT_SRCS}
)

target_link_libraries(${DRONE_DETECT}
  ${TENSORRT_LIBRARY}
  CUDA::cudart
)

rclcpp_components_register_node(${DRONE_DETECT}
  PLUGIN "drone::detect::DetectNode"
  EXECUTABLE detect_node
)

set(DRONE_PNP "drone_pnp")
file(GLOB_RECURSE DRONE_PNP_SRCS "pnp/src/*.cpp")

ament_auto_add_library(${DRONE_PNP} SHARED
  ${DRONE_PNP_SRCS}
)

rclcpp_components_register_node(${DRONE_PNP}
  PLUGIN "drone::pnp::PnpNode"
  EXECUTABLE pnp_node
)

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE
  launch
)
