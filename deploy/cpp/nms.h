// NMS + IoU for 2D boxes.
#pragma once
#include "decode.h"
#include <vector>

// IoU between two boxes
float iou(const Detection& a, const Detection& b);

// Greedy NMS: sort by score, suppress overlaps >= iou_thresh.
std::vector<int> nms(const std::vector<Detection>& dets, float iou_thresh);