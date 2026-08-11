// Late fusion merge implementation.
#include "merge.h"
#include "nms.h"
#include <set>
#include <vector>

std::vector<Detection> merge(const std::vector<Detection>& cam,
                               const std::vector<Detection>& lid,
                               float merge_iou_thresh) {
    if (cam.empty()) return lid;
    if (lid.empty()) return cam;

    std::set<int> used;
    std::vector<Detection> result;

    int n_cam = (int)cam.size();
    int n_lid = (int)lid.size();

    for (int i = 0; i < n_cam; i++) {
        int best_j = -1;
        float best_iou = merge_iou_thresh;
        for (int j = 0; j < n_lid; j++) {
            if (used.count(j)) continue;
            float v = iou(cam[i], lid[j]);
            if (v > best_iou) {
                best_iou = v;
                best_j = j;
            }
        }
        if (best_j >= 0) {
            used.insert(best_j);
            Detection d = cam[i];
            d.score = (cam[i].score + lid[best_j].score) / 2.0f;
            result.push_back(d);
        } else {
            result.push_back(cam[i]);
        }
    }

    for (int j = 0; j < n_lid; j++) {
        if (!used.count(j))
            result.push_back(lid[j]);
    }

    auto keep = nms(result, merge_iou_thresh);
    std::vector<Detection> merged;
    for (int idx : keep) merged.push_back(result[idx]);
    return merged;
}