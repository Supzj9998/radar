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
find_package(OpenCV REQUIRED)

if(EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
elseif(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES
    OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL ""
    OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "52")
  if(DEFINED ENV{CUDAARCHS} AND NOT "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}" CACHE STRING "CUDA architectures" FORCE)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)
  endif()
endif()

if("${CMAKE_CUDA_ARCHITECTURES}" MATCHES "^[0-9]+$"
    AND NOT CMAKE_CUDA_FLAGS MATCHES "-arch=sm_${CMAKE_CUDA_ARCHITECTURES}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${CMAKE_CUDA_ARCHITECTURES}")
endif()

enable_language(CUDA)
include(${PROJECT_SOURCE_DIR}/cmake/modules/libnvinferConfig.cmake)

include_directories(
  "autoaim_manager/include"
  "drone_detecter/include"
  "drone_pnp/include"
  "model_detecter/include"
  "infer/include"
  "model_pnp/include"
)

set(DRONE_INFER "drone_infer")
file(GLOB_RECURSE DRONE_INFER_SRCS "infer/src/*.cpp" "infer/src/*.cu")

add_library(${DRONE_INFER} SHARED
  ${DRONE_INFER_SRCS}
)
target_include_directories(${DRONE_INFER} PUBLIC
  ${PROJECT_SOURCE_DIR}/infer/include
  ${CUDAToolkit_INCLUDE_DIRS}
  ${nvinfer_INCLUDE_DIRS}
)
target_link_libraries(${DRONE_INFER}
  ${nvinfer_LIBRARY}
  CUDA::cudart
  nvonnxparser
  ${OpenCV_LIBS}
)
target_compile_options(${DRONE_INFER} PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-U_FORTIFY_SOURCE>
  $<$<COMPILE_LANGUAGE:CUDA>:-D_FORTIFY_SOURCE=0>
)
install(TARGETS ${DRONE_INFER}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# model_detecter_node 通过本包 infer 目录复用 TensorRT 推理封装。
set(DRONE_DETECT "drone_detect")
file(GLOB_RECURSE DRONE_DETECT_SRCS "model_detecter/src/*.cpp")

ament_auto_add_library(${DRONE_DETECT} SHARED
  ${DRONE_DETECT_SRCS}
)

target_link_libraries(${DRONE_DETECT} ${DRONE_INFER})

rclcpp_components_register_node(${DRONE_DETECT}
  PLUGIN "drone::detect::Detect"
  EXECUTABLE model_detecter_node
)

# drone_detecter_node 用于后续接入另一个相机的目标引导检测。
set(DRONE_DRONE_DETECTER "drone_drone_detecter")
file(GLOB_RECURSE DRONE_DRONE_DETECTER_SRCS "drone_detecter/src/*.cpp")

ament_auto_add_library(${DRONE_DRONE_DETECTER} SHARED
  ${DRONE_DRONE_DETECTER_SRCS}
)

target_link_libraries(${DRONE_DRONE_DETECTER} ${DRONE_INFER})

rclcpp_components_register_node(${DRONE_DRONE_DETECTER}
  PLUGIN "drone::drone_detecter::DroneDetecterNode"
  EXECUTABLE drone_detecter_node
)

# drone_pnp_node 接收无人机检测框，后续用于无人机粗略PnP。
set(DRONE_DRONE_PNP "drone_drone_pnp")
file(GLOB_RECURSE DRONE_DRONE_PNP_SRCS "drone_pnp/src/*.cpp")

ament_auto_add_library(${DRONE_DRONE_PNP} SHARED
  ${DRONE_DRONE_PNP_SRCS}
)

rclcpp_components_register_node(${DRONE_DRONE_PNP}
  PLUGIN "drone::drone_pnp::DronePnpNode"
  EXECUTABLE drone_pnp_node
)

# model_pnp_node 只依赖 ROS/OpenCV 依赖，用于几何解算和消息转换。
set(DRONE_MODEL_PNP "drone_model_pnp")
file(GLOB_RECURSE DRONE_MODEL_PNP_SRCS "model_pnp/src/*.cpp")

ament_auto_add_library(${DRONE_MODEL_PNP} SHARED
  ${DRONE_MODEL_PNP_SRCS}
)

rclcpp_components_register_node(${DRONE_MODEL_PNP}
  PLUGIN "drone::model_pnp::ModelPnpNode"
  EXECUTABLE model_pnp_node
)

# autoaim_manager_node 管理无人机和模型两路AutoAIM目标来源。
set(DRONE_AUTOAIM_MANAGER "drone_autoaim_manager")
file(GLOB_RECURSE DRONE_AUTOAIM_MANAGER_SRCS "autoaim_manager/src/*.cpp")

ament_auto_add_library(${DRONE_AUTOAIM_MANAGER} SHARED
  ${DRONE_AUTOAIM_MANAGER_SRCS}
)

rclcpp_components_register_node(${DRONE_AUTOAIM_MANAGER}
  PLUGIN "drone::autoaim_manager::AutoaimManagerNode"
  EXECUTABLE autoaim_manager_node
)

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE
  launch
)
