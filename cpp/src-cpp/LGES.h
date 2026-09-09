// Pair-level insertion gates and exhaustive candidate selection.

#pragma once

#include "Operators.h"
#include "PDAG.h"
#include "ScorerInterface.h"

#include <optional>
#include <utility>
#include <vector>

namespace lges {

    /** Result of enumerating the valid INSERT(x, y, T) for one ordered pair. */
    struct InsertCandidates {
        std::vector<Insert> operators;
        /** True if enumeration stopped early on a score-decreasing operator.
         *  Only ever set when `prune` is requested. */
        bool found_lower_scoring = false;
    };

    /** @brief Screen a pair using the target's local score in a DAG extension.
     *
     * Keep (x, y) if adding x to y's parents does not lower the local score.
     * Exact ties pass. This checks scores only, not ancestry or operator validity.
     *
     * @param dag_g A DAG member of the current MEC (e.g. PDAG::get_dag_extension).
     * @return True if the pair survives the gate.
     */
    bool check_legal_insert(const PDAG &dag_g, ScorerInterface &scorer, int x, int y);

    /** @brief Enumerate and score every valid INSERT(x, y, T) for a non-adjacent pair.
     *
     * Validity is the standard GES condition set:
     *   1. T ⊆ Ne(y) \ Ad(x)
     *   2. NA_yx ∪ T is a clique, where NA_yx = Ne(y) ∩ Ad(x)
     *   3. NA_yx ∪ T blocks every semi-directed path from y to x
     * and the score is s(y, NA_yx ∪ T ∪ Pa(y) ∪ {x}) − s(y, NA_yx ∪ T ∪ Pa(y)).
     *
     * Path blocking is checked during enumeration, so every returned operator
     * satisfies all three conditions. This function does not apply the DAG gate.
     *
     * @param prune If true, stop at the first score-decreasing operator and set
     *        `found_lower_scoring` (CONSERVATIVE INSERT). The returned vector is
     *        then a partial enumeration and the caller is expected to discard it.
     */
    InsertCandidates score_valid_insert_operators(PDAG &pdag, ScorerInterface &scorer,
                                                  int x, int y,
                                                  UnblockedPathsMap &unblocked_paths_map,
                                                  bool prune = false);

    /** @brief Enumerate a pair's valid inserts, discarding all if any gain is negative.
     *
     * A zero gain does not veto the pair. The DAG gate is the caller's responsibility.
     */
    std::vector<Insert>
    conservative_inserts_for_pair(PDAG &pdag, ScorerInterface &scorer, int x, int y,
                                  UnblockedPathsMap &unblocked_paths_map);

    /** @brief All ordered pairs (x, y), x ≠ y, that are non-adjacent in the PDAG.
     *
     * Returned in lexicographic order for deterministic candidate traversal.
     */
    std::vector<std::pair<int, int>> non_adjacent_pairs(const PDAG &pdag);

    /** @brief Select the best valid insert among pairs passing the DAG gate.
     *
     * Gate each candidate pair on `check_legal_insert`, enumerate all valid
     * operators for the survivors, and return the highest-scoring one.
     *
     * The returned gain may be non-positive; the caller decides whether to
     * apply it. Exact ties retain the first operator encountered.
     *
     * @return The highest-scoring operator, or nullopt if none is valid.
     */
    std::optional<Insert>
    get_safe_insert(PDAG &pdag, const PDAG &dag_g, ScorerInterface &scorer,
                    const std::vector<std::pair<int, int>> &candidates,
                    UnblockedPathsMap &unblocked_paths_map);

    /** @brief Select the best insert after the DAG gate and negative-gain pair veto.
     *
     * As `get_safe_insert`, but additionally discards every operator for a pair
     * as soon as one of its valid operators is score-decreasing.
     */
    std::optional<Insert>
    get_conservative_insert(PDAG &pdag, const PDAG &dag_g, ScorerInterface &scorer,
                            const std::vector<std::pair<int, int>> &candidates,
                            UnblockedPathsMap &unblocked_paths_map);

}// namespace lges
