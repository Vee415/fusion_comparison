cmake_minimum_required(VERSION 3.10)
project(late_2d_deploy LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CUDA ---
find_package(CUDA REQUIRED)
# --- OpenCV ---
find_package(OpenCV REQUIRED)
# --- TensorRT ---
# On Jetson: /usr/lib/aarch64-linux-gnu/libnvinfer.so + headers in /usr/include/aarch64-linux-gnu/
find_library(NVINFER nvinfer REQUIRED)
find_library(NVONPARSER nvonnxparser REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_executable(late_2d_deploy
    main.cpp
    trt_engine.cpp
    preprocess.cu
    decode.cpp
    nms.cpp
    merge.cpp
    calib_parser.cpp
)
target_link_libraries(late_2d_deploy
    ${NVINFER}
    ${NVONPARSER}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)
# GPU architecture for Jetson Orin: sm_87
set(CUDA_ARCHITECTURES 87)
set_target_properties(late_2d_deploy PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})