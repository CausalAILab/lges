#pragma once

#include "ScorerInterface.h"
#include "Eigen/Dense"
#include <unordered_map>
#include <vector>

using Eigen::MatrixXd;

struct FlatSetHash {
    size_t operator()(const FlatSet &values) const noexcept {
        size_t result = 0;
        for (int value: values) {
            result ^=
                    std::hash<int>{}(value) + 0x9e3779b9 + (result << 6) + (result >> 2);
        }
        return result;
    }
};

/** Observational Gaussian BIC, using the biased covariance (division by n).
 * Owns its covariance, so the input can be released after construction.
 * Invalid data or singular/non-positive conditional variances throw rather than
 * putting NaN/infinity into an operator heap. Instances are not thread-safe.
 */
class BICScorer : public ScorerInterface {
public:
    // The cache budget counts parent sets across all targets. Zero disables it.
    static constexpr size_t default_cache_entries = 100000;
    BICScorer(const MatrixXd &data, double alpha,
              size_t max_cache_entries = default_cache_entries);
    double local_score(int target, const FlatSet &parents) override;

private:
    Eigen::Index n_samples;
    double alpha;
    MatrixXd covariance_matrix;
    std::vector<std::unordered_map<FlatSet, double, FlatSetHash>> cache;
    size_t max_cache_entries;
    size_t cache_entries = 0;
};
