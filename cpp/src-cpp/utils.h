//
// Created by Achille Nazaret on 2/16/24.
//

#pragma once

#include <chrono>
#include <functional>
#include <utility>

inline double measure_time(const std::chrono::steady_clock::time_point start_time) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
                   std::chrono::steady_clock::now() - start_time)
            .count();
}

// Explicit hasher: specializing std::hash<pair<int,int>> is not permitted by C++.
struct IntPairHash {
    size_t operator()(const std::pair<int, int> &p) const noexcept {
        size_t h1 = std::hash<int>()(p.first);
        h1 ^= std::hash<int>()(p.second) + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        return h1;
    }
};
