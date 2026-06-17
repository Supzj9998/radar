cmake_minimum_required(VERSION 3.8)
project(tdt_vision)
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

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(  
  "detect/include"
  "resolve/include"
  "calibrate/include"
)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_BUILD_TYPE Debug)

if(NOT DEFINED CMAKE_CUDA_COMPILER)
  if(DEFINED ENV{CUDACXX} AND NOT "$ENV{CUDACXX}" STREQUAL "")
    set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "CUDA compiler" FORCE)
  elseif(EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
  elseif(EXISTS "/usr/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
  if(DEFINED ENV{CUDAARCHS} AND NOT "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}" CACHE STRING "CUDA architectures" FORCE)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
  endif()
endif()

add_subdirectory(utils)
include_directories(${PROJECT_SOURCE_DIR}/utils/include)

set(RADAR_DETECT "radar_detect")
file(GLOB_RECURSE RADAR_DETECT_SRCS "detect/src/*.cpp")
ament_auto_add_library(${RADAR_DETECT} SHARED
    ${RADAR_DETECT_SRCS}
)
find_package(CUDAToolkit REQUIRED)
include(${PROJECT_SOURCE_DIR}/cmake/modules/libnvinferConfig.cmake)
message("Using CUDA toolkit from: ${CUDAToolkit_BIN_DIR}, TensorRT Version: ${nvinfer_VERSION}")
add_definitions(-D use_tensorrt)
enable_language(CUDA)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${nvinfer_INCLUDE_DIRS})

add_subdirectory(infer)
include_directories(${PROJECT_SOURCE_DIR}/infer/include)

target_link_libraries(${RADAR_DETECT}
    utils
    infer 
    ${nvinfer_LIBRARY} 
    CUDA::cudart
    nvonnxparser 
    fmt 
    ${PCL_LIBRARIES}
    )
rclcpp_components_register_node(${RADAR_DETECT}
    PLUGIN "tdt_radar::Detect"
    EXECUTABLE ${RADAR_DETECT}_node
)

set(RADAR_RESOLVE "radar_resolve")
file(GLOB_RECURSE RADAR_RESOLVE_SRCS "resolve/src/*.cpp")

ament_auto_add_library(${RADAR_RESOLVE} SHARED
    ${RADAR_RESOLVE_SRCS}
)
rclcpp_components_register_node(${RADAR_RESOLVE}
    PLUGIN "tdt_radar::Resolve"
    EXECUTABLE ${RADAR_RESOLVE}_node
)

set(RADAR_CALIBRATE "radar_calibrate")
file(GLOB_RECURSE RADAR_CALIBRATE_SRCS "calibrate/src/*.cpp")
ament_auto_add_library(${RADAR_CALIBRATE} SHARED
    ${RADAR_CALIBRATE_SRCS}
)
rclcpp_components_register_node(${RADAR_CALIBRATE}
    PLUGIN "tdt_radar::Calibrate"
    EXECUTABLE ${RADAR_CALIBRATE}_node
)


ament_auto_package(
    USE_SCOPED_HEADER_INSTALL_DIR
    INSTALL_TO_SHARE
    launch
)
