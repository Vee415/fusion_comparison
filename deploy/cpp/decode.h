// Decode CenterNet head outputs: sigmoid -> topk -> gather -> box reconstruction.
#pragma once
#include <vector>

struct Detection {
    float x1, y1, x2, y2;  // pixel coords
    float score;
};

// Decode heat (1,1,24,80) + off (1,2,24,80) + size (1,2,24,80) -> detections.
// Mirrors Python decode_boxes2d exactly.
std::vector<Detection> decode_head(const float* heat, const float* off,
                                     const float* size, int Hg, int Wg,
                                     int stride, int k, float thresh);