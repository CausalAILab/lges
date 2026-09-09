//
// Created by Achille Nazaret on 11/4/23.
//

#pragma once

#include "PDAG.h"
#include "utils.h"
#include <cmath>
#include <stdexcept>

/** A decomposable, score-equivalent score over DAG families.
 * Equivalent DAGs must have the same total score: CPDAG moves depend on this,
 * not just decomposability. Return finite scores or throw on unsupported data.
 * Mutable statistics/caches mean scorers and searches must not be shared across
 * threads without external synchronization.
 */
class ScorerInterface {
public:
    std::map<std::string, double> statistics;
    virtual ~ScorerInterface() = default;

    virtual double local_score(int target, const FlatSet &parents) = 0;

    static double require_finite_score(double score) {
        if (!std::isfinite(score)) {
            throw std::runtime_error("Scorer returned a non-finite score");
        }
        return score;
    }

    double checked_local_score(int target, const FlatSet &parents) {
        return require_finite_score(local_score(target, parents));
    }

    double score_insert(int target, const FlatSet &parents, int parent_to_add) {
        statistics["score_insert-#calls"]++;
        auto start_time = std::chrono::steady_clock::now();
        double score_without_new_parent = checked_local_score(target, parents);
        FlatSet parents_with_new_parent;
        parents_with_new_parent.reserve(parents.size() + 1);
        union_with_single_element(parents, parent_to_add, parents_with_new_parent);
        double score_with_new_parent =
                checked_local_score(target, parents_with_new_parent);

        statistics["score_insert-time"] += measure_time(start_time);
        return require_finite_score(score_with_new_parent - score_without_new_parent);
    }

    double score_delete(int target, const FlatSet &parents, int parent_to_remove) {
        if (!parents.contains(parent_to_remove)) {
            throw std::invalid_argument("Cannot score removal of a missing parent");
        }
        statistics["score_delete-#calls"]++;
        auto start_time = std::chrono::steady_clock::now();
        double score_with_old_parent = checked_local_score(target, parents);
        FlatSet parents_without_old_parent;
        parents_without_old_parent.reserve(parents.size() - 1);
        for (auto p: parents) {
            if (p != parent_to_remove) { parents_without_old_parent.insert(p); }
        }
        double score_without_old_parent =
                checked_local_score(target, parents_without_old_parent);

        statistics["score_delete-time"] += measure_time(start_time);
        return require_finite_score(score_without_old_parent - score_with_old_parent);
    }

    double score_pdag(const PDAG &pdag) {
        double score = 0;
        const PDAG dag = pdag.get_dag_extension();
        // compute the score at nodes that are variables
        for (int target: dag.get_nodes_variables()) {
            score += checked_local_score(target, dag.get_parents(target));
        }
        return require_finite_score(score);
    }
};
