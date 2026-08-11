// Late fusion merge: greedy IoU match cam+lid, mean score, cam box retained, final NMS.
#pragma once
#include "decode.h"
#include <vector>

// Merge cam + lid detections. Mirrors Python _merge exactly.
// cam box geometry is retained for matched pairs; score = mean(cam, lid).
std::vector<Detection> merge(const std::vector<Detection>& cam,
                               const std::vector<Detection>& lid,
                               float merge_iou_thresh);