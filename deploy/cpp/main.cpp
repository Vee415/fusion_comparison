// Late-2D fusion deploy: load two TRT engines, run inference, decode, merge, visualize.
//
// Usage:
//   ./late_2d_deploy --cam_engine late_2d_cam.engine --lid_engine late_2d_lid.engine \
//     --image 000001.png --velodyne 000001.bin --calib 000001.txt --output result.png
//
// Mirrors deploy/demo_onnx.py exactly. Build on Jetson:
//   cd ~/fusion_deploy && cmake . && make
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include "trt_engine.h"
#include "decode.h"
#include "nms.h"
#include "merge.h"
#include "calib_parser.h"
#include "preprocess.h"

// Read a .bin point cloud as float32 (x,y,z,intensity) * N
std::vector<float> read_velodyne(const std::string& path, int& n_points) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { std::cerr << "ERROR: cannot open " << path << std::endl; exit(1); }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    n_points = size / (4 * sizeof(float));
    std::vector<float> pts(n_points * 4);
    fread(pts.data(), sizeof(float), n_points * 4, f);
    fclose(f);
    return pts;
}

void draw_detections(cv::Mat& img, const std::vector<Detection>& dets,
                     cv::Scalar color, const std::string& label, int thickness) {
    for (const auto& d : dets) {
        int x1 = std::max(0, (int)d.x1), y1 = std::max(0, (int)d.y1);
        int x2 = std::min(img.cols - 1, (int)d.x2), y2 = std::min(img.rows - 1, (int)d.y2);
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness);
        cv::putText(img, label + " " + std::to_string(d.score).substr(0, 4),
                    cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
    }
}

int main(int argc, char** argv) {
    std::string cam_engine_path = "late_2d_cam.engine";
    std::string lid_engine_path = "late_2d_lid.engine";
    std::string image_path, velodyne_path, calib_path;
    std::string output_path = "result.png";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--cam_engine" && i+1 < argc) cam_engine_path = argv[++i];
        else if (a == "--lid_engine" && i+1 < argc) lid_engine_path = argv[++i];
        else if (a == "--image" && i+1 < argc) image_path = argv[++i];
        else if (a == "--velodyne" && i+1 < argc) velodyne_path = argv[++i];
        else if (a == "--calib" && i+1 < argc) calib_path = argv[++i];
        else if (a == "--output" && i+1 < argc) output_path = argv[++i];
    }

    if (image_path.empty() || velodyne_path.empty() || calib_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --cam_engine ... --lid_engine ... "
                     "--image ... --velodyne ... --calib ... --output ..." << std::endl;
        return 1;
    }

    const int H = 384, W = 1280, stride = 16, Hg = H / stride, Wg = W / stride;
    const int k_topk = 40;
    const float decode_thresh = 0.1f, nms_thresh = 0.45f, merge_thresh = 0.5f;

    // --- Load engines ---
    std::cout << "=== loading TRT engines ===" << std::endl;
    TrtEngine cam_engine(cam_engine_path);
    TrtEngine lid_engine(lid_engine_path);

    // --- Load + preprocess camera image ---
    std::cout << "=== preprocessing camera ===" << std::endl;
    cv::Mat img_bgr = cv::imread(image_path);
    cv::Mat img_resized;
    cv::resize(img_bgr, img_resized, cv::Size(W, H));

    float* gpu_cam_input;
    cudaMalloc(&gpu_cam_input, 3 * H * W * sizeof(float));
    preprocess_camera(img_bgr, gpu_cam_input, W, H);

    // Copy to host for TRT (TRT API takes host buffers in our wrapper)
    std::vector<float> cam_input(3 * H * W);
    cudaMemcpy(cam_input.data(), gpu_cam_input, 3 * H * W * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(gpu_cam_input);

    // --- Load + preprocess LiDAR ---
    std::cout << "=== preprocessing LiDAR ===" << std::endl;
    int n_points;
    std::vector<float> points = read_velodyne(velodyne_path, n_points);
    std::cout << "  points: " << n_points << std::endl;

    Calib calib = parse_calib(calib_path);
    float* gpu_lid_input;
    cudaMalloc(&gpu_lid_input, 3 * H * W * sizeof(float));
    CalibData calib_data;
    memcpy(calib_data.P2, calib.P2, sizeof(float) * 12);
    memcpy(calib_data.R0, calib.R0, sizeof(float) * 9);
    memcpy(calib_data.Tr, calib.Tr, sizeof(float) * 12);
    preprocess_lidar(points.data(), n_points, gpu_lid_input, calib_data, H, W);

    std::vector<float> lid_input(3 * H * W);
    cudaMemcpy(lid_input.data(), gpu_lid_input, 3 * H * W * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(gpu_lid_input);

    // --- Run inference ---
    std::cout << "=== running inference ===" << std::endl;
    // Allocate output buffers
    int heat_size = 1 * Hg * Wg, off_size = 2 * Hg * Wg, size_size = 2 * Hg * Wg;
    std::vector<float> cam_heat(heat_size), cam_off(off_size), cam_size(size_size);
    std::vector<float> lid_heat(heat_size), lid_off(off_size), lid_size(size_size);

    std::vector<float*> cam_inputs = {cam_input.data()};
    std::vector<float*> cam_outputs = {cam_heat.data(), cam_off.data(), cam_size.data()};
    auto t0 = std::chrono::high_resolution_clock::now();
    cam_engine.infer(cam_inputs, cam_outputs);

    std::vector<float*> lid_inputs = {lid_input.data()};
    std::vector<float*> lid_outputs = {lid_heat.data(), lid_off.data(), lid_size.data()};
    lid_engine.infer(lid_inputs, lid_outputs);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  inference time: " << ms << " ms (both engines)" << std::endl;

    // --- Decode ---
    std::cout << "=== decoding ===" << std::endl;
    auto cam_dets = decode_head(cam_heat.data(), cam_off.data(), cam_size.data(),
                                  Hg, Wg, stride, k_topk, decode_thresh);
    auto lid_dets = decode_head(lid_heat.data(), lid_off.data(), lid_size.data(),
                                  Hg, Wg, stride, k_topk, decode_thresh);
    std::cout << "  cam detections: " << cam_dets.size() << std::endl;
    std::cout << "  lid detections: " << lid_dets.size() << std::endl;

    // --- NMS per stream ---
    auto cam_keep = nms(cam_dets, nms_thresh);
    auto lid_keep = nms(lid_dets, nms_thresh);
    std::vector<Detection> cam_nms, lid_nms;
    for (int idx : cam_keep) cam_nms.push_back(cam_dets[idx]);
    for (int idx : lid_keep) lid_nms.push_back(lid_dets[idx]);
    std::cout << "  cam after NMS: " << cam_nms.size() << std::endl;
    std::cout << "  lid after NMS: " << lid_nms.size() << std::endl;

    // --- Merge ---
    std::cout << "=== merging ===" << std::endl;
    auto merged = merge(cam_nms, lid_nms, merge_thresh);
    std::cout << "  merged detections: " << merged.size() << std::endl;

    // --- Visualize ---
    std::cout << "=== visualizing ===" << std::endl;
    cv::Mat result = img_resized.clone();
    draw_detections(result, cam_nms, cv::Scalar(255, 0, 0), "cam", 2);   // blue
    draw_detections(result, lid_nms, cv::Scalar(0, 255, 0), "lid", 2);   // green
    draw_detections(result, merged, cv::Scalar(0, 0, 255), "merge", 3);  // red

    cv::imwrite(output_path, result);
    std::cout << "saved " << output_path << std::endl;

    // --- Summary ---
    std::cout << "\n=== summary ===" << std::endl;
    std::cout << "  cam detections:  " << cam_nms.size() << std::endl;
    std::cout << "  lid detections:  " << lid_nms.size() << std::endl;
    std::cout << "  merged:          " << merged.size() << std::endl;
    std::cout << "  inference:       " << ms << " ms" << std::endl;
    std::cout << "\nLegend: blue=cam, green=lidar, red=merged (final output)" << std::endl;

    return 0;
}