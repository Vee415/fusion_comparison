// Parse KITTI calib txt files: P2, R0_rect, Tr_velo_to_cam.
#pragma once
#include <string>

struct Calib {
    float P2[12];      // 3x4 projection matrix (row-major)
    float R0[9];       // 3x3 rectification matrix (row-major)
    float Tr[12];      // 3x4 velo_to_cam transform (row-major)
};

Calib parse_calib(const std::string& path);