// KITTI calib file parser.
#include "calib_parser.h"
#include <fstream>
#include <sstream>
#include <iostream>

static void parse_floats(const std::string& vals, float* out, int n) {
    std::istringstream ss(vals);
    for (int i = 0; i < n; i++) ss >> out[i];
}

Calib parse_calib(const std::string& path) {
    Calib c = {};
    std::ifstream f(path);
    if (!f.good()) {
        std::cerr << "ERROR: cannot open calib: " << path << std::endl;
        return c;
    }
    std::string line;
    while (std::getline(f, line)) {
        // Find the colon
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        // Trim whitespace
        while (!key.empty() && key.back() == ' ') key.pop_back();
        while (!key.empty() && key.front() == ' ') key.erase(0, 1);
        std::string vals = line.substr(colon + 1);

        if (key == "P2") parse_floats(vals, c.P2, 12);
        else if (key == "R0_rect") parse_floats(vals, c.R0, 9);
        else if (key == "Tr_velo_to_cam") parse_floats(vals, c.Tr, 12);
    }
    return c;
}