// Decode CenterNet head outputs -> detections.
#include "decode.h"
#include <algorithm>
#include <cmath>

// Simple top-k using partial_sort (k is small, e.g. 40)
struct CellScore {
    int idx;     // linear index into Hg*Wg
    float score;
};

std::vector<Detection> decode_head(const float* heat, const float* off,
                                     const float* size, int Hg, int Wg,
                                     int stride, int k, float thresh) {
    // 1. Sigmoid
    int n = Hg * Wg;
    std::vector<float> sig(n);
    for (int i = 0; i < n; i++)
        sig[i] = 1.0f / (1.0f + std::exp(-heat[i]));

    // 2. Top-k
    std::vector<CellScore> cells(n);
    for (int i = 0; i < n; i++) { cells[i] = {i, sig[i]}; }
    int actual_k = std::min(k, n);
    std::partial_sort(cells.begin(), cells.begin() + actual_k, cells.end(),
        [](const CellScore& a, const CellScore& b) { return a.score > b.score; });

    // 3. Decode each top-k cell
    std::vector<Detection> dets;
    for (int i = 0; i < actual_k; i++) {
        float score = cells[i].score;
        if (score <= thresh) continue;

        int idx = cells[i].idx;
        int y = idx / Wg;  // row (forward in BEV, but in 2D it's image row)
        int x = idx % Wg;  // col (image column)

        // Gather offset: off[0] = dy, off[1] = dx
        float off_y = off[0 * Hg * Wg + y * Wg + x];
        float off_x = off[1 * Hg * Wg + y * Wg + x];

        // Gather size: size[0] = h, size[1] = w
        float h = size[0 * Hg * Wg + y * Wg + x];
        float w = size[1 * Hg * Wg + y * Wg + x];

        float cy = (y + off_y) * stride;
        float cx = (x + off_x) * stride;

        Detection d;
        d.x1 = cx - w / 2;
        d.y1 = cy - h / 2;
        d.x2 = cx + w / 2;
        d.y2 = cy + h / 2;
        d.score = score;
        dets.push_back(d);
    }
    return dets;
}