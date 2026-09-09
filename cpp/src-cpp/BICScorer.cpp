#include "BICScorer.h"
#include <limits>

BICScorer::BICScorer(const MatrixXd &data, double alpha, size_t max_cache_entries)
    : n_samples(data.rows()), alpha(alpha), max_cache_entries(max_cache_entries) {
    if (data.rows() < 2 || data.cols() < 1 ||
        data.cols() > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("BIC requires at least two samples and one variable");
    }
    if (!std::isfinite(alpha) || alpha <= 0) {
        throw std::invalid_argument("BIC alpha must be finite and positive");
    }
    if (!data.allFinite()) {
        throw std::invalid_argument("Input data must contain only finite values");
    }
    const MatrixXd centered = data.rowwise() - data.colwise().mean();
    covariance_matrix = centered.adjoint() * centered / static_cast<double>(n_samples);
    if (!covariance_matrix.allFinite() ||
        (covariance_matrix.diagonal().array() <= 0).any()) {
        throw std::invalid_argument(
                "Input contains a constant column or an invalid covariance");
    }
    cache.resize(static_cast<size_t>(data.cols()));
}

double BICScorer::local_score(int target, const FlatSet &parents) {
    const auto valid_node = [this](int node) {
        return node >= 0 && static_cast<size_t>(node) < cache.size();
    };
    if (!valid_node(target) || parents.contains(target) ||
        !std::all_of(parents.begin(), parents.end(), valid_node)) {
        throw std::invalid_argument("Invalid target or parent set for BIC");
    }
    statistics["local_score-#calls-total"]++;
    auto &cache_target = cache[static_cast<size_t>(target)];
    if (const auto found = cache_target.find(parents); found != cache_target.end()) {
        return found->second;
    }
    statistics["local_score-#calls-nocache"]++;

    const double target_variance = covariance_matrix(target, target);
    double sigma = target_variance;
    if (!parents.empty()) {
        const auto count = static_cast<Eigen::Index>(parents.size());
        MatrixXd covariance(count, count);
        Eigen::VectorXd cross_covariance(count);
        const std::vector<int> parent_nodes(parents.begin(), parents.end());
        for (Eigen::Index i = 0; i < count; ++i) {
            const int parent_i = parent_nodes[static_cast<size_t>(i)];
            cross_covariance(i) = covariance_matrix(parent_i, target);
            for (Eigen::Index j = 0; j < count; ++j) {
                covariance(i, j) =
                        covariance_matrix(parent_i, parent_nodes[static_cast<size_t>(j)]);
            }
        }
        const Eigen::LLT<MatrixXd> factorization(covariance);
        if (factorization.info() != Eigen::Success) {
            throw std::runtime_error(
                    "Singular parent covariance while scoring variable " +
                    std::to_string(target));
        }
        const double rank_tolerance =
                64 * std::numeric_limits<double>::epsilon() * static_cast<double>(count);
        for (Eigen::Index i = 0; i < count; ++i) {
            const double pivot = factorization.matrixL().coeff(i, i);
            if (pivot * pivot <= rank_tolerance * covariance(i, i)) {
                throw std::runtime_error("Numerically singular parent covariance");
            }
        }
        const Eigen::VectorXd beta = factorization.solve(cross_covariance);
        sigma -= cross_covariance.dot(beta);
    }
    // A tiny positive residual from cancellation is not evidence of a valid
    // regression. Reject values at covariance-roundoff scale as singular.
    const double residual_tolerance = 64 * std::numeric_limits<double>::epsilon() *
                                      (static_cast<double>(parents.size()) + 1) *
                                      target_variance;
    if (!std::isfinite(sigma) || sigma <= residual_tolerance) {
        throw std::runtime_error(
                "Non-positive or numerically zero conditional variance for variable " +
                std::to_string(target) + "; input is singular or too ill-conditioned");
    }

    const double samples = static_cast<double>(n_samples);
    const double likelihood = -0.5 * samples * (1 + std::log(sigma));
    const double penalty =
            0.5 * std::log(samples) * (static_cast<double>(parents.size()) + 1) * alpha;
    const double score = require_finite_score(likelihood - penalty);

    if (max_cache_entries != 0) {
        // A generation reset bounds retained parent sets without an extra LRU
        // copy of each key. Eviction affects computation cost, never the score.
        if (cache_entries == max_cache_entries) {
            for (auto &target_cache: cache) {
                // Release buckets as well as keys: otherwise different targets
                // could each retain a former full generation's capacity.
                decltype(cache)::value_type{}.swap(target_cache);
            }
            cache_entries = 0;
            statistics["local_score-cache-resets"]++;
        }
        cache_target.emplace(parents, score);
        ++cache_entries;
        statistics["local_score-cache-peak-entries"] =
                std::max(statistics["local_score-cache-peak-entries"],
                         static_cast<double>(cache_entries));
    }
    return score;
}
