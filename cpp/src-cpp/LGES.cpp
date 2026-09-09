// Pair-level insertion gates and exhaustive candidate selection.

#include "LGES.h"

#include <algorithm>
#include <stack>
#include <tuple>

namespace lges {

    bool check_legal_insert(const PDAG &dag_g, ScorerInterface &scorer, const int x,
                            const int y) {
        const FlatSet &parents_y = dag_g.get_parents(y);
        FlatSet parents_y_with_x;
        parents_y_with_x.reserve(parents_y.size() + 1);
        union_with_single_element(parents_y, x, parents_y_with_x);

        // Reject only a strict score decrease; exact ties keep the pair.
        return !(scorer.checked_local_score(y, parents_y_with_x) <
                 scorer.checked_local_score(y, parents_y));
    }

    InsertCandidates score_valid_insert_operators(PDAG &pdag, ScorerInterface &scorer,
                                                  const int x, const int y,
                                                  UnblockedPathsMap &unblocked_paths_map,
                                                  const bool prune) {
        InsertCandidates result;

        // NA_yx = Ne(y) ∩ Ad(x) must itself be a clique, otherwise no T can fix it.
        FlatSet neighbors_y_adjacent_x = pdag.get_neighbors_adjacent(y, x);
        if (!pdag.is_clique(neighbors_y_adjacent_x)) { return result; }

        const FlatSet neighbors_y_not_adjacent_x = pdag.get_neighbors_not_adjacent(y, x);
        const FlatSet &parents_y = pdag.get_parents(y);

        // Effective parents for T = {}: NA_yx ∪ Pa(y). Each T adds its own members.
        FlatSet effective_parents_y = neighbors_y_adjacent_x;
        effective_parents_y.insert(parents_y.begin(), parents_y.end());

        // Enumerate the T ⊆ Ne(y) \ Ad(x) for which NA_yx ∪ T is a clique, in
        // inclusion order: if the clique condition fails for T it fails for every
        // superset, so those branches are never pushed. This is the same walk as
        // XGES::find_inserts_to_y; the semi-directed path condition is applied on
        // top of it below.
        //
        // Stack entries: <T, iterator into neighbors_y_not_adjacent_x, effective
        // parents, whether some subset of T already blocked all semi-directed
        // paths>.
        std::stack<std::tuple<FlatSet, FlatSet::const_iterator, FlatSet, bool>> stack;
        stack.emplace(FlatSet{}, neighbors_y_not_adjacent_x.begin(), effective_parents_y,
                      false);

        while (!stack.empty()) {
            auto top = std::move(stack.top());
            stack.pop();
            auto &T = std::get<0>(top);
            auto it = std::get<1>(top);
            auto &effective_parents = std::get<2>(top);
            const bool subset_passed_blocking = std::get<3>(top);

            // Condition 3: NA_yx ∪ T blocks all semi-directed paths from y to x.
            // Blocking is monotone in T, so once a subset blocks, so does T.
            bool blocking = subset_passed_blocking;
            if (!blocking) {
                FlatSet blocked_nodes = neighbors_y_adjacent_x;
                blocked_nodes.insert(T.begin(), T.end());
                blocking = pdag.is_blocking_semi_directed_paths(
                        y, x, blocked_nodes, unblocked_paths_map, false);
            }

            if (blocking) {
                const double score = scorer.score_insert(y, effective_parents, x);
                result.operators.emplace_back(x, y, T, score, effective_parents);
                if (prune && score < 0) {
                    result.found_lower_scoring = true;
                    return result;
                }
            }

            // Extend T with each remaining z, keeping only the extensions for which
            // NA_yx ∪ T ∪ {z} is still a clique.
            while (it != neighbors_y_not_adjacent_x.end()) {
                const auto z = *it;
                ++it;
                const auto &adjacent_z = pdag.get_adjacent(z);
                if (std::ranges::includes(adjacent_z, T) &&
                    std::ranges::includes(adjacent_z, neighbors_y_adjacent_x)) {
                    auto T_prime = T;
                    T_prime.insert(z);
                    auto effective_parents_prime = effective_parents;
                    effective_parents_prime.insert(z);
                    stack.emplace(std::move(T_prime), it,
                                  std::move(effective_parents_prime), blocking);
                }
            }
        }

        return result;
    }

    std::vector<Insert>
    conservative_inserts_for_pair(PDAG &pdag, ScorerInterface &scorer, const int x,
                                  const int y, UnblockedPathsMap &unblocked_paths_map) {
        InsertCandidates candidates = score_valid_insert_operators(
                pdag, scorer, x, y, unblocked_paths_map, true);
        if (candidates.found_lower_scoring) { return {}; }
        return std::move(candidates.operators);
    }

    std::vector<std::pair<int, int>> non_adjacent_pairs(const PDAG &pdag) {
        std::vector<std::pair<int, int>> pairs;
        const auto &nodes = pdag.get_nodes_variables();
        for (const int x: nodes) {
            const auto &adjacent_x = pdag.get_adjacent(x);
            for (const int y: nodes) {
                if (x == y || adjacent_x.contains(y)) { continue; }
                pairs.emplace_back(x, y);
            }
        }
        return pairs;
    }

    /** Gate each pair, optionally veto negative gains, then select the best insert. */
    static std::optional<Insert>
    best_insert(PDAG &pdag, const PDAG &dag_g, ScorerInterface &scorer,
                const std::vector<std::pair<int, int>> &candidates,
                UnblockedPathsMap &unblocked_paths_map, const bool prune) {
        std::optional<Insert> best;

        for (const auto &[x, y]: candidates) {
            if (!check_legal_insert(dag_g, scorer, x, y)) { continue; }

            std::vector<Insert> operators;
            if (prune) {
                operators = conservative_inserts_for_pair(pdag, scorer, x, y,
                                                          unblocked_paths_map);
            } else {
                operators = std::move(score_valid_insert_operators(pdag, scorer, x, y,
                                                                   unblocked_paths_map)
                                              .operators);
            }

            for (auto &op: operators) {
                // Keep the first operator when scores tie.
                if (!best.has_value() || op.get_score() > best->get_score()) {
                    best = std::move(op);
                }
            }
        }

        return best;
    }

    std::optional<Insert>
    get_safe_insert(PDAG &pdag, const PDAG &dag_g, ScorerInterface &scorer,
                    const std::vector<std::pair<int, int>> &candidates,
                    UnblockedPathsMap &unblocked_paths_map) {
        return best_insert(pdag, dag_g, scorer, candidates, unblocked_paths_map, false);
    }

    std::optional<Insert>
    get_conservative_insert(PDAG &pdag, const PDAG &dag_g, ScorerInterface &scorer,
                            const std::vector<std::pair<int, int>> &candidates,
                            UnblockedPathsMap &unblocked_paths_map) {
        return best_insert(pdag, dag_g, scorer, candidates, unblocked_paths_map, true);
    }

}// namespace lges
