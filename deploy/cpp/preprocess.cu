// CUDA preprocessing kernels for camera + LiDAR inputs.
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// --- Camera preprocess: resize + normalize + transpose on GPU ---
__global__ void cam_preprocess_kernel(const float* src_bgr, float* dst_chw,
                                        int src_w, int src_h, int dst_w, int dst_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    float sx = (float)x * src_w / dst_w;
    float sy = (float)y * src_h / dst_h;
    int x0 = (int)sx, y0 = (int)sy;
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    float dx = sx - x0, dy = sy - y0;

    for (int c = 0; c < 3; c++) {
        float val = (1 - dx) * (1 - dy) * src_bgr[(y0 * src_w + x0) * 3 + (2 - c)] +
                    dx * (1 - dy) * src_bgr[(y0 * src_w + x1) * 3 + (2 - c)] +
                    (1 - dx) * dy * src_bgr[(y1 * src_w + x0) * 3 + (2 - c)] +
                    dx * dy * src_bgr[(y1 * src_w + x1) * 3 + (2 - c)];
        dst_chw[c * dst_h * dst_w + y * dst_w + x] = val / 255.0f;
    }
}

void preprocess_camera(const cv::Mat& img_bgr, float* gpu_out, int dst_w, int dst_h) {
    cv::Mat src_f;
    img_bgr.convertTo(src_f, CV_32FC3);
    float* gpu_src;
    cudaMalloc(&gpu_src, src_f.total() * sizeof(float) * 3);
    cudaMemcpy(gpu_src, src_f.data, src_f.total() * sizeof(float) * 3, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    cam_preprocess_kernel<<<grid, block>>>(gpu_src, gpu_out, img_bgr.cols, img_bgr.rows, dst_w, dst_h);
    cudaDeviceSynchronize();
    cudaFree(gpu_src);
}

// --- LiDAR preprocess: project points to image, render 3 channels ---
// CalibData struct is defined in preprocess.h
struct CalibData {
    float P2[12];
    float R0[9];
    float Tr[12];
};

__global__ void lidar_render_kernel(const float* points, int n_points,
                                      const CalibData* calib,
                                      float* out_img, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    float vx = points[idx * 4 + 0];
    float vy = points[idx * 4 + 1];
    float vz = points[idx * 4 + 2];
    float intensity = points[idx * 4 + 3];

    // velo -> cam: Tr_velo_to_cam @ [x,y,z,1]
    float cx = calib->Tr[0] * vx + calib->Tr[1] * vy + calib->Tr[2] * vz + calib->Tr[3];
    float cy = calib->Tr[4] * vx + calib->Tr[5] * vy + calib->Tr[6] * vz + calib->Tr[7];
    float cz = calib->Tr[8] * vx + calib->Tr[9] * vy + calib->Tr[10] * vz + calib->Tr[11];

    // R0_rect @ cam
    float rx = calib->R0[0] * cx + calib->R0[1] * cy + calib->R0[2] * cz;
    float ry = calib->R0[3] * cx + calib->R0[4] * cy + calib->R0[5] * cz;
    float rz = calib->R0[6] * cx + calib->R0[7] * cy + calib->R0[8] * cz;

    if (rz <= 0) return;

    // P2 @ [rx, ry, rz, 1]
    float u = calib->P2[0] * rx + calib->P2[1] * ry + calib->P2[2] * rz + calib->P2[3];
    float v = calib->P2[4] * rx + calib->P2[5] * ry + calib->P2[6] * rz + calib->P2[7];
    float d = calib->P2[8] * rx + calib->P2[9] * ry + calib->P2[10] * rz + calib->P2[11];

    if (d <= 0) return;
    u /= d;
    v /= d;

    if (u < 0 || u >= W || v < 0 || v >= H) return;

    int ui = (int)u, vi = (int)v;
    int pixel = vi * W + ui;

    // Write 3 channels: [depth, height(velo_z), intensity]
    out_img[0 * H * W + pixel] = rz;
    out_img[1 * H * W + pixel] = vz;
    out_img[2 * H * W + pixel] = intensity;
}

void preprocess_lidar(const float* points, int n_points, float* gpu_out,
                       const CalibData& calib, int H, int W) {
    cudaMemset(gpu_out, 0, 3 * H * W * sizeof(float));

    float* gpu_points;
    cudaMalloc(&gpu_points, n_points * 4 * sizeof(float));
    cudaMemcpy(gpu_points, points, n_points * 4 * sizeof(float), cudaMemcpyHostToDevice);

    CalibData* gpu_calib;
    cudaMalloc(&gpu_calib, sizeof(CalibData));
    cudaMemcpy(gpu_calib, &calib, sizeof(CalibData), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n_points + 255) / 256;
    lidar_render_kernel<<<grid, block>>>(gpu_points, n_points, gpu_calib, gpu_out, H, W);
    cudaDeviceSynchronize();
    cudaFree(gpu_points);
    cudaFree(gpu_calib);
}