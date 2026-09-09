//
// Created by Achille Nazaret on 11/6/23.
//

#pragma once

#include "PDAG.h"
#include "ScorerInterface.h"

#include "spdlog/logger.h"
#include <map>
#include <memory>
#include <optional>
#include <utility>

/** Filter insertion candidates before heap insertion and again when popped.
 *
 * Standard uses local candidate updates. Safe screens each pair using a DAG
 * extension; Conservative also rejects pairs with any negative valid insertion.
 * Both filtered strategies rebuild the insertion heap after every graph change,
 * since eligibility can change outside the local update neighborhood.
 * Reversal candidates are not filtered by these strategies.
 */
enum class InsertStrategy { Standard, Safe, Conservative };

class XGES {
public:
    // The scorer is borrowed and must outlive the search and any copies.
    // An omitted logger uses spdlog's default logger (or a silent fallback).
    XGES(int n_variables, ScorerInterface *scorer,
         std::shared_ptr<spdlog::logger> logger = nullptr);
    XGES(const XGES &other);

    void set_insert_strategy(InsertStrategy strategy) { insert_strategy = strategy; }

    void fit_xges(bool extended_search);
    void fit_ops(bool use_reverse);
    void fit_ges(bool use_reverse);
    void fit_ges_turn();
    // Phased deletion/reversal search with one insertion per cycle.
    void fit_lges(bool extended_search);
    void fit_lges0();
    void set_strict_lges(bool enabled) { strict_lges = enabled; }
    double get_score() const;
    double get_initial_score() const;
    const PDAG &get_pdag() const;

    std::map<std::string, double> statistics;

private:
    // Regression tests inspect cache and candidate invariants.
    friend struct XGESTestAccess;
    int n_variables;
    ScorerInterface *scorer;
    PDAG pdag;
    double initial_score = 0;
    double total_score = 0;
    std::shared_ptr<spdlog::logger> _logger;

    InsertStrategy insert_strategy = InsertStrategy::Standard;
    bool strict_lges = false;
    // A DAG extension used by the score gate shared by Safe and Conservative.
    // Recomputed lazily: `pdag_version` is bumped on every applied operator.
    long pdag_version = 0;
    long dag_cache_version = -1;
    std::unique_ptr<PDAG> dag_cache;

    /** The DAG member of the current MEC that the strategies are evaluated in. */
    const PDAG &current_dag();
    /** Apply the shared DAG score gate to the pair (x -> y). */
    bool passes_gate(int y, int x);
    /** CONSERVATIVE INSERT: does any valid operator here score below zero? */
    bool pair_is_discarded(const std::vector<Insert> &operators,
                           UnblockedPathsMap &unblocked_paths_map);
    /** Is the ordered pair (x -> y) admissible under `insert_strategy`? */
    bool pair_allowed_by_strategy(int y, int x, UnblockedPathsMap &unblocked_paths_map);
    /** Generate insert candidates for y, honouring `insert_strategy`. */
    void find_strategy_inserts_to_y(int y, std::vector<Insert> &candidate_inserts,
                                    UnblockedPathsMap &unblocked_paths_map,
                                    int parent_x = -1, bool positive_only = true);
    /** Re-apply the strategy to a candidate popped from the heap. */
    bool insert_allowed_by_strategy(const Insert &insert,
                                    UnblockedPathsMap &unblocked_paths_map);

    void heuristic_xges0(std::vector<Insert> &candidate_inserts,
                         std::vector<Reverse> &candidate_reverses,
                         std::vector<Delete> &candidate_deletes,
                         UnblockedPathsMap &unblocked_paths_map,
                         bool initialize_candidates = true);
    void update_operator_candidates_naive(std::vector<Insert> &candidate_inserts,
                                          std::vector<Reverse> &candidate_reverses,
                                          std::vector<Delete> &candidate_deletes,
                                          UnblockedPathsMap &unblocked_paths_map,
                                          bool use_reverse = true);
    void update_operator_candidates_efficient(EdgeModificationsMap &edge_modifications,
                                              std::vector<Insert> &candidate_inserts,
                                              std::vector<Reverse> &candidate_reverses,
                                              std::vector<Delete> &candidate_deletes,
                                              UnblockedPathsMap &unblocked_paths_map);
    void block_each_edge_and_research(UnblockedPathsMap &unblocked_paths_map);
    void heuristic_lges(std::optional<std::pair<int, int>> blocked_pair = std::nullopt);
    void block_each_edge_and_research_lges();
    bool apply_best_lges_move(char kind,
                             std::optional<std::pair<int, int>> blocked_pair = std::nullopt,
                             bool complete_turn = false);
    struct UndirectedTurn {
        int x, y;
        FlatSet C;
        double score;
    };
    std::optional<UndirectedTurn> best_undirected_turn();
    void apply_undirected_turn(const UndirectedTurn &turn);

    void find_inserts_to_y(int y, std::vector<Insert> &candidate_inserts,
                           int parent_x = -1, bool positive_only = true) const;

    // A preselected x may be a child of y when enumerating the insert part of
    // a reversal; the caller chooses the appropriate validity check.
    void find_inserts_to_y_from_x(int y, int x, std::vector<Insert> &candidate_inserts,
                                  bool positive_only) const;

    void find_delete_to_y_from_x(int y, int x, std::vector<Delete> &candidate_deletes,
                                 bool positive_only = true) const;
    void find_deletes_to_y(int y, std::vector<Delete> &candidate_deletes,
                           bool positive_only = true) const;

    void find_reverse_to_y_from_x(int y, int x,
                                  std::vector<Reverse> &candidate_reverses) const;
    void find_reverse_to_y(int y, std::vector<Reverse> &candidate_reverses) const;
};
