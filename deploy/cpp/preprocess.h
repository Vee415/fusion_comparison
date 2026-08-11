// Preprocessing function declarations (implemented in preprocess.cu).
#pragma once
#include <opencv2/opencv.hpp>

// Calib data for GPU kernel
struct CalibData {
    float P2[12];
    float R0[9];
    float Tr[12];
};

// Camera preprocess: BGR image -> RGB -> resize -> /255 -> CHW on GPU.
// gpu_out must be pre-allocated on GPU (3*H*W floats).
void preprocess_camera(const cv::Mat& img_bgr, float* gpu_out, int dst_w, int dst_h);

// LiDAR preprocess: project points to image, render [depth, height, intensity].
// gpu_out must be pre-allocated on GPU (3*H*W floats).
void preprocess_lidar(const float* points, int n_points, float* gpu_out,
                       const CalibData& calib, int H, int W);