// NMS implementation.
#include "nms.h"
#include <algorithm>

float iou(const Detection& a, const Detection& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, inter_x2 - inter_x1);
    float ih = std::max(0.0f, inter_y2 - inter_y1);
    float inter = iw * ih;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter;
    return union_area > 1e-9f ? inter / union_area : 0.0f;
}

std::vector<int> nms(const std::vector<Detection>& dets, float iou_thresh) {
    std::vector<int> order(dets.size());
    for (size_t i = 0; i < dets.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(),
        [&](int a, int b) { return dets[a].score > dets[b].score; });

    std::vector<int> keep;
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < order.size(); i++) {
        int idx = order[i];
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        for (size_t j = i + 1; j < order.size(); j++) {
            int jdx = order[j];
            if (suppressed[jdx]) continue;
            if (iou(dets[idx], dets[jdx]) >= iou_thresh)
                suppressed[jdx] = true;
        }
    }
    return keep;
}