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
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
  endif()
endif()

if("${CMAKE_CUDA_ARCHITECTURES}" MATCHES "^[0-9]+$"
    AND NOT CMAKE_CUDA_FLAGS MATCHES "-arch=sm_${CMAKE_CUDA_ARCHITECTURES}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${CMAKE_CUDA_ARCHITECTURES}")
endif()

enable_language(CUDA)
include(${PROJECT_SOURCE_DIR}/cmake/modules/libnvinferConfig.cmake)

include_directories(
  "detect/include"
  "infer/include"
  "pnp/include"
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

install(
  TARGETS ${DRONE_INFER}
  LIBRARY DESTINATION lib
)

# detect_node 通过本包 infer 目录复用 TensorRT 推理封装。
set(DRONE_DETECT "drone_detect")
file(GLOB_RECURSE DRONE_DETECT_SRCS "detect/src/*.cpp")

ament_auto_add_library(${DRONE_DETECT} SHARED
  ${DRONE_DETECT_SRCS}
)

target_link_libraries(${DRONE_DETECT} ${DRONE_INFER})

rclcpp_components_register_node(${DRONE_DETECT}
  PLUGIN "drone::detect::Detect"
  EXECUTABLE detect_node
)

# pnp_node 只依赖 ROS/OpenCV 依赖，用于几何解算和消息转换。
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
