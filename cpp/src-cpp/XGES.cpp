//
// Created by Achille Nazaret on 11/6/23.
//
#include "XGES.h"
#include "LGES.h"
#include "set_ops.h"
#include "utils.h"
#include "EdgeQueueSet.h"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

#include <stack>
#include <type_traits>
#include "spdlog/sinks/null_sink.h"

using namespace std::chrono;

/** @brief Constructor of the XGES class.
 *
 * @param n_variables The number of variables in the dataset.
 * @param scorer The scorer to use to score the PDAGs.
 *
 */
XGES::XGES(const int n_variables, ScorerInterface *scorer,
           std::shared_ptr<spdlog::logger> logger)
    : n_variables(n_variables), scorer(scorer), pdag(n_variables),
      _logger(logger ? std::move(logger) : spdlog::default_logger()) {
    if (n_variables < 1) {
        throw std::invalid_argument("Search requires at least one variable");
    }
    if (!scorer) { throw std::invalid_argument("Search requires a scorer"); }
    if (!_logger) {
        _logger = std::make_shared<spdlog::logger>(
                "xges", std::make_shared<spdlog::sinks::null_sink_mt>());
    }
    initial_score = scorer->score_pdag(pdag);
    total_score = initial_score;
}

XGES::XGES(const XGES &other)
    : statistics(other.statistics), n_variables(other.n_variables), scorer(other.scorer),
      pdag(other.pdag), initial_score(other.initial_score),
      total_score(other.total_score), _logger(other._logger),
      insert_strategy(other.insert_strategy), strict_lges(other.strict_lges),
      pdag_version(other.pdag_version) {
    // dag_cache is deliberately left empty; it is rebuilt on first use.
}


/** @brief Fit the XGES algorithm using the scorer provided in the constructor.
 *
 * @param extended_search If true, the extended search of XGES is performed. Otherwise,
 * only XGES-0 is performed.
 */
void XGES::fit_xges(bool extended_search) {
    std::vector<Insert> candidate_inserts;
    std::vector<Reverse> candidate_reverses;
    std::vector<Delete> candidate_deletes;
    UnblockedPathsMap unblocked_paths_map;

    candidate_inserts.reserve(100 * static_cast<size_t>(n_variables));
    candidate_reverses.reserve(n_variables);
    candidate_deletes.reserve(n_variables);

    heuristic_xges0(candidate_inserts, candidate_reverses, candidate_deletes,
                    unblocked_paths_map, true);

    if (extended_search) { block_each_edge_and_research(unblocked_paths_map); }
}

/** @brief Extended search of XGES.
 *
 * Extended search is performed after XGES-0 has exhausted all possible operators.
 * It successively delete some edges, and resume the search with XGES-0 (preventing
 * the deleted edges from being inserted again). If the score increases, the new PDAG
 * is kept.
 *
 * At the beginning of the search, `all_edge_deletes` contains all the possible deletes
 * for the current PDAG. If the PDAG is updated, we keep iterating through `all_edge_deletes`
 * until all the deletes have been tried. Only then we recompute `all_edge_deletes` on the
 * last updated PDAG. Until no new deletes are found. This is more efficient than recomputing
 * all the deletes at each PDAG update, and re-testing all of them (as most of them will be
 * the same).
 *
 *
 * @param unblocked_paths_map The unblocked paths map used by XGES-0.
 */
void XGES::block_each_edge_and_research(UnblockedPathsMap &unblocked_paths_map) {
    std::vector<Delete> all_edge_deletes;
    bool deletes_of_pdag_are_updated = false;

    while (!all_edge_deletes.empty() || !deletes_of_pdag_are_updated) {
        if (all_edge_deletes.empty()) {
            for (const int y: pdag.get_nodes_variables()) {
                find_deletes_to_y(y, all_edge_deletes, false);
            }
            if (all_edge_deletes.empty()) { break; }
            deletes_of_pdag_are_updated = true;
        }
        // get delete from the heap
        std::pop_heap(all_edge_deletes.begin(), all_edge_deletes.end());
        auto delete_ = std::move(all_edge_deletes.back());
        all_edge_deletes.pop_back();
        if (!pdag.is_delete_valid(delete_)) { continue; }
        // Apply the delete
        XGES xges_copy(*this);
        // Trial counters are deltas. Merge all work, including rejected trials,
        // exactly once; the scorer is shared and already counts every trial.
        xges_copy.statistics.clear();
        xges_copy.pdag.statistics.clear();
        EdgeModificationsMap edge_modifications;
        std::vector<Insert> candidate_inserts;
        std::vector<Reverse> candidate_reverses;
        std::vector<Delete> candidate_deletes;
        xges_copy.pdag.apply_delete(delete_, edge_modifications);
        xges_copy.pdag_version++;
        xges_copy.total_score = ScorerInterface::require_finite_score(
                xges_copy.total_score + delete_.score);
        xges_copy.pdag.add_forbidden_insert(delete_.x, delete_.y);
        UnblockedPathsMap blocked_paths_map_copy = unblocked_paths_map;
        _logger->debug("EXTENDED SEARCH: {}", delete_);
        for (auto &[fst, snd]: edge_modifications) { _logger->trace("\tEdge {}", snd); }

        xges_copy.update_operator_candidates_efficient(
                edge_modifications, candidate_inserts, candidate_reverses,
                candidate_deletes, blocked_paths_map_copy);
        xges_copy.heuristic_xges0(candidate_inserts, candidate_reverses,
                                  candidate_deletes, blocked_paths_map_copy, false);
        for (const auto &[name, value]: xges_copy.statistics) {
            statistics[name] += value;
        }
        for (const auto &[name, value]: xges_copy.pdag.statistics) {
            pdag.statistics[name] += value;
        }
        if (pdag == xges_copy.pdag) {
            statistics["extended_search-unchanged"] += 1;
            continue;
        }
        if (xges_copy.total_score - total_score > 1e-7) {
            _logger->debug("EXTENDED SEARCH ACCEPTED: with increase {} and {}",
                           xges_copy.total_score - total_score, delete_);
            total_score = xges_copy.total_score;
            auto graph_statistics = std::move(pdag.statistics);
            pdag = std::move(xges_copy.pdag);
            pdag.statistics = std::move(graph_statistics);
            // Replacing the graph invalidates the parent's cached DAG too.
            ++pdag_version;
            dag_cache.reset();
            dag_cache_version = -1;
            unblocked_paths_map = std::move(blocked_paths_map_copy);
            deletes_of_pdag_are_updated = false;
            statistics["extended_search-accepted"] += 1;
        } else {
            _logger->debug("EXTENDED SEARCH REJECTED: {} {}", delete_,
                           xges_copy.total_score);
            statistics["extended_search-rejected"] += 1;
        }
    }
}

/** @brief Run the XGES-0 heuristic.
 *
 * XGES-0 is the main heuristic of the XGES algorithm. It maintains lists of all possible
 * operators and apply them in order: delete, reverse, and insert. XGES-0 stops when no
 * more operators can be applied.
 * XGES-0 uses update_operator_candidates_efficient to update the operators after each
 * PDAG update.
 *
 * @param candidate_inserts The vector of candidate inserts.
 * @param candidate_reverses The vector of candidate reverses.
 * @param candidate_deletes The vector of candidate deletes.
 * @param unblocked_paths_map The unblocked paths map used by XGES-0.
 * @param initialize_candidates If true, all enabled candidate types are initialized at the beginning
 * of the heuristic.
 */
void XGES::heuristic_xges0(std::vector<Insert> &candidate_inserts,
                           std::vector<Reverse> &candidate_reverses,
                           std::vector<Delete> &candidate_deletes,
                           UnblockedPathsMap &unblocked_paths_map,
                           bool initialize_candidates) {
    if (initialize_candidates) {
        // Also support resuming a search after another fit method or strategy.
        const auto start_init = steady_clock::now();
        update_operator_candidates_naive(candidate_inserts, candidate_reverses,
                                         candidate_deletes, unblocked_paths_map);
        statistics["initialize_operators-time"] += measure_time(start_init);
    }
    EdgeModificationsMap edge_modifications;
    int i_operations = 1;

    // XGES-0 main loop, in order: delete, reverse, insert; one operator per iteration
    while (!candidate_inserts.empty() || !candidate_reverses.empty() ||
           !candidate_deletes.empty()) {
        edge_modifications.clear();

        if (!candidate_deletes.empty()) {
            // apply the best delete if possible
            std::pop_heap(candidate_deletes.begin(), candidate_deletes.end());
            auto best_delete = std::move(candidate_deletes.back());
            candidate_deletes.pop_back();
            if (pdag.is_delete_valid(best_delete)) {
                pdag.apply_delete(best_delete, edge_modifications);
                pdag_version++;
                total_score = ScorerInterface::require_finite_score(total_score +
                                                                    best_delete.score);
                _logger->debug("{}: {}", i_operations, best_delete);
            } else {
                continue;
            }
        } else if (!candidate_reverses.empty()) {
            // apply the best reverse if possible (no delete available)
            std::pop_heap(candidate_reverses.begin(), candidate_reverses.end());
            auto best_reverse = std::move(candidate_reverses.back());
            candidate_reverses.pop_back();
            if (pdag.is_reverse_valid(best_reverse, unblocked_paths_map)) {
                pdag.apply_reverse(best_reverse, edge_modifications);
                pdag_version++;
                total_score = ScorerInterface::require_finite_score(total_score +
                                                                    best_reverse.score);
                _logger->debug("{}: {}", i_operations, best_reverse);
            } else {
                continue;
            }
        } else if (!candidate_inserts.empty()) {
            // apply the best insert if possible (no delete or reverse available)
            std::pop_heap(candidate_inserts.begin(), candidate_inserts.end());
            auto best_insert = std::move(candidate_inserts.back());
            candidate_inserts.pop_back();
            if (pdag.is_insert_valid(best_insert, unblocked_paths_map) &&
                insert_allowed_by_strategy(best_insert, unblocked_paths_map)) {
                pdag.apply_insert(best_insert, edge_modifications);
                pdag_version++;
                total_score = ScorerInterface::require_finite_score(total_score +
                                                                    best_insert.score);
                _logger->debug("{}: {}", i_operations, best_insert);
            } else {
                continue;
            }
        }
        // if we reach this point, we have applied an operator
        i_operations++;
        for (auto &edge_modification: edge_modifications) {
            _logger->trace("\tEdge {}", edge_modification.second);
        }

        // update the new possible operators
        update_operator_candidates_efficient(edge_modifications, candidate_inserts,
                                             candidate_reverses, candidate_deletes,
                                             unblocked_paths_map);
    }
}

void XGES::fit_ops(bool use_reverse) {
    std::vector<Insert> candidate_inserts;
    std::vector<Reverse> candidate_reverses;
    std::vector<Delete> candidate_deletes;
    UnblockedPathsMap unblocked_paths_map;

    candidate_inserts.reserve(100 * static_cast<size_t>(n_variables));
    candidate_reverses.reserve(n_variables);
    candidate_deletes.reserve(n_variables);

    update_operator_candidates_naive(candidate_inserts, candidate_reverses,
                                     candidate_deletes, unblocked_paths_map, use_reverse);
    EdgeModificationsMap edge_modifications;
    int i_operations = 1;

    while (!candidate_inserts.empty() || !candidate_reverses.empty() ||
           !candidate_deletes.empty()) {

        // Look for the operator with the highest score
        // Each of candidate_inserts, candidate_reverses, candidate_deletes are heaps
        // with the highest score at the front.
        double best_score = 0;
        int best_type = -1;// 0: insert, 1: reverse, 2: delete

        if (!candidate_inserts.empty()) {
            if (candidate_inserts.front().score > best_score) {
                best_type = 0;
                best_score = candidate_inserts.front().score;
            }
        }
        if (!candidate_reverses.empty() && use_reverse) {
            if (candidate_reverses.front().score > best_score) {
                best_type = 1;
                best_score = candidate_reverses.front().score;
            }
        }
        if (!candidate_deletes.empty()) {
            if (candidate_deletes.front().score > best_score) { best_type = 2; }
        }

        edge_modifications.clear();
        if (best_type == 0) {
            std::pop_heap(candidate_inserts.begin(), candidate_inserts.end());
            auto best_insert = std::move(candidate_inserts.back());
            candidate_inserts.pop_back();
            if (pdag.is_insert_valid(best_insert, unblocked_paths_map) &&
                insert_allowed_by_strategy(best_insert, unblocked_paths_map)) {
                pdag.apply_insert(best_insert, edge_modifications);
                pdag_version++;
                total_score = ScorerInterface::require_finite_score(total_score +
                                                                    best_insert.score);
                _logger->debug("{}: {}", i_operations, best_insert);
            } else {
                continue;
            }
        } else if (best_type == 1) {
            // apply the best reverse if possible (no delete available)
            std::pop_heap(candidate_reverses.begin(), candidate_reverses.end());
            auto best_reverse = std::move(candidate_reverses.back());
            candidate_reverses.pop_back();
            if (pdag.is_reverse_valid(best_reverse, unblocked_paths_map)) {
                pdag.apply_reverse(best_reverse, edge_modifications);
                pdag_version++;
                total_score = ScorerInterface::require_finite_score(total_score +
                                                                    best_reverse.score);
                _logger->debug("{}: {}", i_operations, best_reverse);
            } else {
                continue;
            }
        } else if (best_type == 2) {
            std::pop_heap(candidate_deletes.begin(), candidate_deletes.end());
            auto best_delete = std::move(candidate_deletes.back());
            candidate_deletes.pop_back();
            if (pdag.is_delete_valid(best_delete)) {
                pdag.apply_delete(best_delete, edge_modifications);
                pdag_version++;
                total_score = ScorerInterface::require_finite_score(total_score +
                                                                    best_delete.score);
                _logger->debug("{}: {}", i_operations, best_delete);
            } else {
                continue;
            }
        } else {
            throw std::logic_error("No positive operator in a nonempty candidate heap");
        }

        i_operations++;
        for (auto &edge_modification: edge_modifications) {
            _logger->trace("\tEdge {}", edge_modification.second);
        }

        update_operator_candidates_naive(candidate_inserts, candidate_reverses,
                                         candidate_deletes, unblocked_paths_map,
                                         use_reverse);
    }
}

void XGES::fit_ges(bool use_reverse) {
    std::vector<Insert> candidate_inserts;
    std::vector<Reverse> candidate_reverses;
    std::vector<Delete> candidate_deletes;
    UnblockedPathsMap unblocked_paths_map;

    candidate_inserts.reserve(100 * static_cast<size_t>(n_variables));
    candidate_reverses.reserve(n_variables);
    candidate_deletes.reserve(n_variables);


    EdgeModificationsMap edge_modifications;
    int i_operations = 1;
    int last_operation = -1;

    while (i_operations > last_operation) {
        last_operation = i_operations;

        // only apply inserts
        for (int y = 0; y < n_variables; ++y) {
            find_strategy_inserts_to_y(y, candidate_inserts, unblocked_paths_map);
        }
        while (!candidate_inserts.empty()) {
            edge_modifications.clear();
            std::pop_heap(candidate_inserts.begin(), candidate_inserts.end());
            auto best_insert = std::move(candidate_inserts.back());
            candidate_inserts.pop_back();
            if (pdag.is_insert_valid(best_insert, unblocked_paths_map) &&
                insert_allowed_by_strategy(best_insert, unblocked_paths_map)) {
                pdag.apply_insert(best_insert, edge_modifications);
                pdag_version++;
                total_score = ScorerInterface::require_finite_score(total_score +
                                                                    best_insert.score);
                _logger->debug("{}: {}", i_operations, best_insert);
                i_operations++;
                candidate_inserts.clear();
                for (int y = 0; y < n_variables; ++y) {
                    find_strategy_inserts_to_y(y, candidate_inserts, unblocked_paths_map);
                }
            }
        }
        // only apply deletes
        for (int y = 0; y < n_variables; ++y) { find_deletes_to_y(y, candidate_deletes); }
        while (!candidate_deletes.empty()) {
            edge_modifications.clear();
            std::pop_heap(candidate_deletes.begin(), candidate_deletes.end());
            auto best_delete = std::move(candidate_deletes.back());
            candidate_deletes.pop_back();
            if (pdag.is_delete_valid(best_delete)) {
                pdag.apply_delete(best_delete, edge_modifications);
                pdag_version++;
                total_score = ScorerInterface::require_finite_score(total_score +
                                                                    best_delete.score);
                _logger->debug("{}: {}", i_operations, best_delete);
                i_operations++;
                candidate_deletes.clear();
                for (int y = 0; y < n_variables; ++y) {
                    find_deletes_to_y(y, candidate_deletes);
                }
            }
        }
        // only apply reverses, if use_reverse is true
        // note that without reverse, there is no multiple
        // iterations of forward and backward
        if (!use_reverse) { return; }
        for (int y = 0; y < n_variables; ++y) {
            find_reverse_to_y(y, candidate_reverses);
        }
        while (!candidate_reverses.empty()) {
            edge_modifications.clear();
            std::pop_heap(candidate_reverses.begin(), candidate_reverses.end());
            auto best_reverse = std::move(candidate_reverses.back());
            candidate_reverses.pop_back();
            if (pdag.is_reverse_valid(best_reverse, unblocked_paths_map)) {
                pdag.apply_reverse(best_reverse, edge_modifications);
                pdag_version++;
                total_score = ScorerInterface::require_finite_score(total_score +
                                                                    best_reverse.score);
                _logger->debug("{}: {}", i_operations, best_reverse);
                i_operations++;
                candidate_reverses.clear();
                for (int y = 0; y < n_variables; ++y) {
                    find_reverse_to_y(y, candidate_reverses);
                }
            }
        }
    }
}

void XGES::fit_lges(bool extended_search) {
    heuristic_lges();
    if (extended_search) { block_each_edge_and_research_lges(); }
}

void XGES::fit_ges_turn() {
    for (char phase: {'I', 'D', 'R'}) {
        while (apply_best_lges_move(phase, std::nullopt, true)) {}
    }
}

void XGES::fit_lges0() {
    for (char phase: {'I', 'R', 'D'}) {
        while (apply_best_lges_move(phase)) {}
    }
}

/** Rebuild the active phase's candidates and apply its best positive valid move. */
bool XGES::apply_best_lges_move(char kind,
                               std::optional<std::pair<int, int>> blocked_pair, bool complete_turn) {
    UnblockedPathsMap paths;
    EdgeModificationsMap changes;
    const auto best_valid = [](auto &candidates, auto valid) {
        using Operator = typename std::decay_t<decltype(candidates)>::value_type;
        while (!candidates.empty()) {
            std::pop_heap(candidates.begin(), candidates.end());
            auto op = std::move(candidates.back());
            candidates.pop_back();
            if (valid(op)) { return std::optional<Operator>(std::move(op)); }
        }
        return std::optional<Operator>{};
    };
    double delta = 0;
    std::string description;
    if (kind == 'D') {
        std::vector<Delete> candidates;
        for (int y: pdag.get_nodes_variables()) { find_deletes_to_y(y, candidates); }
        auto op = best_valid(candidates, [this](const Delete &candidate) {
            return pdag.is_delete_valid(candidate);
        });
        if (!op) { return false; }
        pdag.apply_delete(*op, changes);
        delta = op->score;
        description = fmt::format("{}", *op);
    } else if (kind == 'R') {
        std::vector<Reverse> candidates;
        for (int y: pdag.get_nodes_variables()) { find_reverse_to_y(y, candidates); }
        auto op = best_valid(candidates, [this, &paths](const Reverse &candidate) {
            return pdag.is_reverse_valid(candidate, paths);
        });
        const auto turn = (strict_lges || complete_turn) ? best_undirected_turn() : std::nullopt;
        if (turn && (!op || turn->score > op->score)) {
            apply_undirected_turn(*turn);
            delta = turn->score;
            description = fmt::format("Turn: {} → {}, score = {}", turn->x, turn->y, delta);
            statistics["lges-undirected_turns"] += 1;
        } else if (op) {
            pdag.apply_reverse(*op, changes);
            delta = op->score;
            description = fmt::format("{}", *op);
        } else { return false; }
    } else {
        std::vector<Insert> candidates;
        for (int y: pdag.get_nodes_variables()) {
            find_strategy_inserts_to_y(y, candidates, paths);
        }
        auto op = best_valid(candidates, [this, &paths, &blocked_pair](const Insert &candidate) {
            if (blocked_pair) {
                const auto [x, y] = *blocked_pair;
                if ((candidate.x == x && candidate.y == y) ||
                    (candidate.x == y && candidate.y == x)) { return false; }
            }
            return pdag.is_insert_valid(candidate, paths);
        });
        if (!op) { return false; }
        pdag.apply_insert(*op, changes);
        delta = op->score;
        description = fmt::format("{}", *op);
    }
    ++pdag_version;
    total_score = ScorerInterface::require_finite_score(total_score + delta);
    _logger->debug("LGES {}: {}", pdag_version, description);
    return true;
}

/** Exhaust deletions and reversals, then attempt one insertion per cycle. */
void XGES::heuristic_lges(std::optional<std::pair<int, int>> blocked_pair) {
    statistics["heuristic_lges-#calls"] += 1;
    for (;;) {
        const long before = pdag_version;
        statistics["heuristic_lges-cycles"] += 1;
        while (apply_best_lges_move('D', blocked_pair)) {}
        while (apply_best_lges_move('R', blocked_pair)) {}
        apply_best_lges_move('I', blocked_pair);
        if (pdag_version == before) { return; }
    }
}

/** Enumerate clique subsets around y and test separation in its undirected component. */
std::optional<XGES::UndirectedTurn> XGES::best_undirected_turn() {
    std::optional<UndirectedTurn> best;
    for (int y: pdag.get_nodes_variables()) {
        for (int x: pdag.get_neighbors(y)) {
            auto pool = pdag.get_neighbors(y);
            pool.erase(x);
            const auto na = pdag.get_neighbors_adjacent(y, x);
            if (is_subset(pool, pdag.get_adjacent(x))) { continue; }
            const std::vector<int> nodes(pool.begin(), pool.end());
            std::vector<std::pair<FlatSet, size_t>> stack{{{}, 0}};
            while (!stack.empty()) {
                auto [C, next] = std::move(stack.back());
                stack.pop_back();
                if (!is_subset(C, pdag.get_adjacent(x))) {
                    FlatSet targets;
                    for (int n: na) { if (!C.contains(n)) { targets.insert(n); } }
                    std::vector<char> visited(n_variables, 0);
                    visited[x] = visited[y] = 1;
                    std::vector<int> queue(C.begin(), C.end());
                    for (int n: C) { visited[n] = 1; }
                    bool separated = true;
                    for (size_t head = 0; head < queue.size() && separated; ++head) {
                        for (int n: pdag.get_neighbors(queue[head])) {
                            if (visited[n]) { continue; }
                            if (targets.contains(n)) { separated = false; break; }
                            visited[n] = 1;
                            queue.push_back(n);
                        }
                    }
                    if (separated) {
                        auto py = pdag.get_parents(y);
                        py.insert(C.begin(), C.end());
                        auto px = pdag.get_parents(x);
                        for (int n: C) { if (na.contains(n)) { px.insert(n); } }
                        px.insert(y);
                        const double gain = ScorerInterface::require_finite_score(
                                scorer->score_insert(y, py, x) + scorer->score_delete(x, px, y));
                        if (gain > 0 && (!best || gain > best->score)) {
                            best = UndirectedTurn{x, y, C, gain};
                        }
                    }
                }
                for (size_t i = next; i < nodes.size(); ++i) {
                    if (!is_subset(C, pdag.get_adjacent(nodes[i]))) { continue; }
                    auto extension = C;
                    extension.insert(nodes[i]);
                    stack.emplace_back(std::move(extension), i + 1);
                }
            }
        }
    }
    return best;
}

void XGES::apply_undirected_turn(const UndirectedTurn &turn) {
    pdag.remove_undirected_edge(turn.x, turn.y);
    pdag.add_directed_edge(turn.x, turn.y);
    for (int c: turn.C) {
        pdag.remove_undirected_edge(c, turn.y);
        pdag.add_directed_edge(c, turn.y);
    }
    // Complete through a consistent extension so all newly created colliders survive.
    auto graph_statistics = std::move(pdag.statistics);
    pdag = pdag.get_cpdag();
    pdag.statistics = std::move(graph_statistics);
}

/** Try each valid deletion, then resume the phased search with that pair blocked.
 * Accept only a net improvement. Rebuild the trial set immediately on acceptance;
 * restrictions belong to the trial and never become part of the accepted graph.
 */
void XGES::block_each_edge_and_research_lges() {
    for (;;) {
        std::vector<Delete> candidates;
        for (int y: pdag.get_nodes_variables()) {
            find_deletes_to_y(y, candidates, /*positive_only=*/false);
        }
        bool accepted = false;
        while (!candidates.empty()) {
            std::pop_heap(candidates.begin(), candidates.end());
            auto deletion = std::move(candidates.back());
            candidates.pop_back();
            if (!pdag.is_delete_valid(deletion)) { continue; }

            XGES trial(*this);
            trial.statistics.clear();
            trial.pdag.statistics.clear();
            EdgeModificationsMap changes;
            trial.pdag.apply_delete(deletion, changes);
            ++trial.pdag_version;
            trial.total_score = ScorerInterface::require_finite_score(
                    trial.total_score + deletion.score);
            statistics["extended_search-trials"] += 1;
            _logger->debug("LGES EXTENDED SEARCH: {}", deletion);
            trial.heuristic_lges(std::pair{deletion.x, deletion.y});
            for (const auto &[name, value]: trial.statistics) { statistics[name] += value; }
            for (const auto &[name, value]: trial.pdag.statistics) {
                pdag.statistics[name] += value;
            }
            if (pdag == trial.pdag) {
                statistics["extended_search-unchanged"] += 1;
                continue;
            }
            if (trial.total_score - total_score <= (strict_lges ? 0.0 : 1e-7)) {
                statistics["extended_search-rejected"] += 1;
                continue;
            }

            total_score = trial.total_score;
            auto graph_statistics = std::move(pdag.statistics);
            pdag = std::move(trial.pdag);
            pdag.statistics = std::move(graph_statistics);
            ++pdag_version;
            dag_cache.reset();
            dag_cache_version = -1;
            statistics["extended_search-accepted"] += 1;
            accepted = true;
            break;
        }
        if (!accepted) { return; }
    }
}

void XGES::update_operator_candidates_naive(std::vector<Insert> &candidate_inserts,
                                            std::vector<Reverse> &candidate_reverses,
                                            std::vector<Delete> &candidate_deletes,
                                            UnblockedPathsMap &unblocked_paths_map,
                                            bool use_reverse) {
    unblocked_paths_map.clear();
    candidate_inserts.clear();
    candidate_reverses.clear();
    candidate_deletes.clear();
    for (int y = 0; y < n_variables; ++y) {
        find_strategy_inserts_to_y(y, candidate_inserts, unblocked_paths_map);
        if (use_reverse) { find_reverse_to_y(y, candidate_reverses); }
        find_deletes_to_y(y, candidate_deletes);
    }
}


static void add_pairs(std::set<std::pair<int, int>> &pairs, const FlatSet &xs, int y) {
    for (auto x: xs) { pairs.emplace(x, y); }
}
static void add_pairs(std::set<std::pair<int, int>> &pairs, int x, const FlatSet &ys) {
    for (auto y: ys) { pairs.emplace(x, y); }
}
static void add_pairs(std::set<std::pair<int, int>> &pairs, const FlatSet &xs,
                      const FlatSet &ys) {
    for (auto x: xs) {
        for (auto y: ys) { pairs.emplace(x, y); }
    }
}

/** @brief Update the candidate operators after a PDAG update.
 *
 * Edge changes identify affected endpoints and path dependencies. Regenerate
 * their insert, reverse and delete candidates; filtered insertion strategies
 * additionally require a full insertion refresh.
 *
 * @param edge_modifications The edge modifications that have been applied to the PDAG.
 * @param candidate_inserts The vector of candidate inserts.
 * @param candidate_reverses The vector of candidate reverses.
 * @param candidate_deletes The vector of candidate deletes.
 * @param unblocked_paths_map The unblocked paths map used by XGES-0.
 */
void XGES::update_operator_candidates_efficient(EdgeModificationsMap &edge_modifications,
                                                std::vector<Insert> &candidate_inserts,
                                                std::vector<Reverse> &candidate_reverses,
                                                std::vector<Delete> &candidate_deletes,
                                                UnblockedPathsMap &unblocked_paths_map) {

    auto start_time = steady_clock::now();
    // First, undo all the edge modifications
    for (auto &[fst, edge_modification]: edge_modifications) {
        pdag.apply_edge_modification(edge_modification, true);
    }

    std::set<int> full_insert_to_y;
    std::map<int, std::set<int>> partial_insert_to_y;
    std::set<int> full_delete_to_y;
    std::set<int> full_delete_from_x;
    std::set<std::pair<int, int>> delete_x_y;
    std::set<int> full_reverse_to_y;
    std::set<int> full_reverse_from_x;
    std::set<std::pair<int, int>> reverse_x_y;

    // Re-apply the edge modifications one by one and update the operators
    for (auto &[fst, edge_modification]: edge_modifications) {
        int a;
        int b;
        if (edge_modification.is_old_directed()) {
            a = edge_modification.get_old_source();
            b = edge_modification.get_old_target();
        } else if (edge_modification.is_new_directed()) {
            a = edge_modification.get_new_source();
            b = edge_modification.get_new_target();
        } else {
            a = edge_modification.x;
            b = edge_modification.y;
        }
        // Only Standard uses the local insertion dependency table. The LGES
        // strategies regenerate their insertion heaps globally below.
        if (insert_strategy == InsertStrategy::Standard) {
            // Track inserts
            switch (edge_modification.get_change()) {
                case EdgeChange::AddUndirected:// a  b becomes a -- b
                    // y = a
                    full_insert_to_y.insert(a);
                    [[fallthrough]];
                case EdgeChange::AddDirected:// a  b becomes a → b
                    // y = b
                    full_insert_to_y.insert(b);
                    // y \in Ne(a) ∩ Ne(b)
                    std::ranges::set_intersection(
                            pdag.get_neighbors(a), pdag.get_neighbors(b),
                            std::inserter(full_insert_to_y, full_insert_to_y.begin()));
                    // x=a and y \in Ne(b)
                    for (auto target: pdag.get_neighbors(b)) {
                        partial_insert_to_y[target].insert(a);
                    }
                    // x=b and y \in Ne(a)
                    for (auto target: pdag.get_neighbors(a)) {
                        partial_insert_to_y[target].insert(b);
                    }
                    break;
                case EdgeChange::RemoveUndirected:// a -- b becomes a  b
                    // x=a and y \in Ne(b) u {b}
                    for (auto target: pdag.get_neighbors(b)) {
                        if (target == a) { continue; }
                        partial_insert_to_y[target].insert(a);
                    }
                    partial_insert_to_y[b].insert(a);
                    // y=a and x \in Ad(b)
                    partial_insert_to_y[a].insert(pdag.get_adjacent(b).begin(),
                                                  pdag.get_adjacent(b).end());
                    // x=b and y \in Ne(a) u {a}
                    for (auto target: pdag.get_neighbors(a)) {
                        if (target == b) { continue; }
                        partial_insert_to_y[target].insert(b);
                    }
                    partial_insert_to_y[a].insert(b);
                    // y=b and x \in Ad(a)
                    partial_insert_to_y[b].insert(pdag.get_adjacent(a).begin(),
                                                  pdag.get_adjacent(a).end());
                    //SD(x,y,a,b)
                    if (unblocked_paths_map.contains({a, b})) {
                        for (auto [x, y]: unblocked_paths_map[{a, b}]) {
                            partial_insert_to_y[y].insert(x);
                        }
                    }
                    // Keep these witnesses until the reversal update consumes them.
                    // SD(x,y,b,a)
                    if (unblocked_paths_map.contains({b, a})) {
                        for (auto [x, y]: unblocked_paths_map[{b, a}]) {
                            partial_insert_to_y[y].insert(x);
                        }
                    }
                    // Keep these witnesses until the reversal update consumes them.
                    break;
                case EdgeChange::Orient:// a -- b becomes a → b
                    // y = a and x \in Ad(b)
                    partial_insert_to_y[a].insert(pdag.get_adjacent(b).begin(),
                                                  pdag.get_adjacent(b).end());
                    // y = b
                    full_insert_to_y.insert(b);
                    // SD(x,y,b,a)
                    if (unblocked_paths_map.contains({b, a})) {
                        for (auto [x, y]: unblocked_paths_map[{b, a}]) {
                            partial_insert_to_y[y].insert(x);
                        }
                    }
                    // Keep these witnesses until the reversal update consumes them.
                    break;
                case EdgeChange::RemoveDirected:// a → b becomes a  b
                    // x=a and y \in Ne(b) u {b}
                    for (auto target: pdag.get_neighbors(b)) {
                        partial_insert_to_y[target].insert(a);
                    }
                    partial_insert_to_y[b].insert(a);
                    // x=b and y \in Ne(a) u {a}
                    for (auto target: pdag.get_neighbors(a)) {
                        partial_insert_to_y[target].insert(b);
                    }
                    partial_insert_to_y[a].insert(b);
                    full_insert_to_y.insert(b);
                    // SD(x,y,a,b)
                    if (unblocked_paths_map.contains({a, b})) {
                        for (auto [x, y]: unblocked_paths_map[{a, b}]) {
                            partial_insert_to_y[y].insert(x);
                        }
                    }
                    // Keep these witnesses until the reversal update consumes them.
                    break;
                case EdgeChange::Unorient:// a → b becomes a -- b
                    full_insert_to_y.insert(a);
                    full_insert_to_y.insert(b);
                    break;
                case EdgeChange::Reverse:// a → b becomes a ← b
                    full_insert_to_y.insert(a);
                    full_insert_to_y.insert(b);
                    // SD(x,y,a,b)
                    if (unblocked_paths_map.contains({a, b})) {
                        for (auto [x, y]: unblocked_paths_map[{a, b}]) {
                            partial_insert_to_y[y].insert(x);
                        }
                    }
                    // Keep these witnesses until the reversal update consumes them.
                    break;
            }
        }
        // Track deletes
        FlatSet x_intersection;
        switch (edge_modification.get_change()) {
            case EdgeChange::AddUndirected:// a  b becomes a -- b
                full_delete_to_y.insert(a);
                [[fallthrough]];
            case EdgeChange::AddDirected:// a  b becomes a → b
                full_delete_to_y.insert(b);
                full_delete_from_x.insert(a);
                full_delete_from_x.insert(b);
                // x \in  Ad(a) ∩ Ad(b) and y \in Ne(a) ∩ Ne(b) [almost never happens]
                std::ranges::set_intersection(
                        pdag.get_adjacent(a), pdag.get_adjacent(b),
                        std::inserter(x_intersection, x_intersection.begin()));
                if (!x_intersection.empty()) {
                    FlatSet y_intersection;
                    std::ranges::set_intersection(
                            pdag.get_neighbors(a), pdag.get_neighbors(b),
                            std::inserter(y_intersection, y_intersection.begin()));
                    add_pairs(delete_x_y, x_intersection, y_intersection);
                }
                break;
            case EdgeChange::RemoveUndirected:// a -- b becomes a  b
                break;
            case EdgeChange::Orient:        // a -- b becomes a → b
            case EdgeChange::RemoveDirected:// a → b becomes a  b
                full_delete_to_y.insert(b);
                break;
            case EdgeChange::Unorient:// a → b becomes a -- b
            case EdgeChange::Reverse: // a → b becomes a ← b
                full_delete_to_y.insert(a);
                full_delete_to_y.insert(b);
                break;
        }

        // Track reverse
        switch (edge_modification.get_change()) {
            case EdgeChange::AddUndirected:// a  b becomes a -- b
                // y \in {a, b}
                full_reverse_to_y.insert(a);
                [[fallthrough]];
            case EdgeChange::AddDirected:// a  b becomes a → b
                // y = b
                full_reverse_to_y.insert(b);
                // y \in Ne(a) ∩ Ne(b)
                std::ranges::set_intersection(
                        pdag.get_neighbors(a), pdag.get_neighbors(b),
                        std::inserter(full_reverse_to_y, full_reverse_to_y.begin()));
                // x \in {a, b}
                full_reverse_from_x.insert(a);
                full_reverse_from_x.insert(b);
                break;
            case EdgeChange::RemoveUndirected:// a -- b becomes a  b
                // y \in {a, b} or x \in {a, b}
                full_reverse_to_y.insert(a);
                full_reverse_to_y.insert(b);
                full_reverse_from_x.insert(a);
                full_reverse_from_x.insert(b);
                // SD(x,y,a,b)
                if (unblocked_paths_map.contains({a, b})) {
                    for (auto [x, y]: unblocked_paths_map[{a, b}]) {
                        reverse_x_y.emplace(x, y);
                    }
                    unblocked_paths_map.erase({a, b});
                }
                // SD(x,y,b,a)
                if (unblocked_paths_map.contains({b, a})) {
                    for (auto [x, y]: unblocked_paths_map[{b, a}]) {
                        reverse_x_y.emplace(x, y);
                    }
                    unblocked_paths_map.erase({b, a});
                }
                break;
            case EdgeChange::Orient:// a -- b becomes a → b
                // y \in {a, b} or x = b
                full_reverse_to_y.insert(a);
                full_reverse_to_y.insert(b);
                full_reverse_from_x.insert(b);
                // SD(x,y,b,a)
                if (unblocked_paths_map.contains({b, a})) {
                    for (auto [x, y]: unblocked_paths_map[{b, a}]) {
                        reverse_x_y.emplace(x, y);
                    }
                    unblocked_paths_map.erase({b, a});
                }
                break;
            case EdgeChange::RemoveDirected:// a → b becomes a  b
                // y = b or x \in {a, b}
                full_reverse_to_y.insert(b);
                full_reverse_from_x.insert(a);
                full_reverse_from_x.insert(b);
                // SD(x,y,a,b)
                if (unblocked_paths_map.contains({a, b})) {
                    for (auto [x, y]: unblocked_paths_map[{a, b}]) {
                        reverse_x_y.emplace(x, y);
                    }
                    unblocked_paths_map.erase({a, b});
                }
                break;
            case EdgeChange::Unorient:// a → b becomes a -- b
                // y \in {a, b} or x = b
                full_reverse_to_y.insert(a);
                full_reverse_to_y.insert(b);
                full_reverse_from_x.insert(b);
                break;
            case EdgeChange::Reverse:// a → b becomes a ← b
                // y \in {a, b} or x \in {a, b}
                full_reverse_to_y.insert(a);
                full_reverse_to_y.insert(b);
                full_reverse_from_x.insert(a);
                full_reverse_from_x.insert(b);
                // SD(x,y,a,b)
                if (unblocked_paths_map.contains({a, b})) {
                    for (auto [x, y]: unblocked_paths_map[{a, b}]) {
                        reverse_x_y.emplace(x, y);
                    }
                    unblocked_paths_map.erase({a, b});
                }
                break;
        }
        pdag.apply_edge_modification(edge_modification);
    }

    if (insert_strategy != InsertStrategy::Standard) {
        // A changed DAG extension can flip a gate at a distant target. A local
        // orientation/deletion can also remove a ConservativeInsert veto even
        // when the surviving positive insert's own validity stays unchanged.
        // The Standard invalidation table does not cover these dependencies.
        candidate_inserts.clear();
        for (int y: pdag.get_nodes_variables()) {
            find_strategy_inserts_to_y(y, candidate_inserts, unblocked_paths_map);
        }
        statistics["lges-full_insert_refresh-#calls"] += 1;
    } else {
        // Find the inserts
        // step 1: remove partial inserts that are now full
        std::vector<int> keys_to_erase;
        for (const auto &[y, xs]: partial_insert_to_y) {
            if (full_insert_to_y.contains(y)) { keys_to_erase.push_back(y); }
        }
        for (auto key: keys_to_erase) { partial_insert_to_y.erase(key); }
        // step 2: find the partial inserts
        for (const auto &[y, xs]: partial_insert_to_y) {
            for (auto x: xs) {
                // check that x is not adjacent to y
                if (!pdag.get_adjacent(y).contains(x) && x != y) {
                    find_strategy_inserts_to_y(y, candidate_inserts, unblocked_paths_map,
                                               x);
                }
            }
        }
        // step 3: find the full inserts
        for (auto y: full_insert_to_y) {
            find_strategy_inserts_to_y(y, candidate_inserts, unblocked_paths_map);
        }
    }

    // Find the deletes
    // step 1: form the edges to delete
    for (auto x: full_delete_from_x) { add_pairs(delete_x_y, x, pdag.get_neighbors(x)); }
    for (auto x: full_delete_from_x) { add_pairs(delete_x_y, x, pdag.get_children(x)); }
    for (auto y: full_delete_to_y) { add_pairs(delete_x_y, pdag.get_parents(y), y); }
    for (auto y: full_delete_to_y) { add_pairs(delete_x_y, pdag.get_neighbors(y), y); }
    // step 2: find the deletes
    for (auto [x, y]: delete_x_y) {
        if (x != y) { find_delete_to_y_from_x(y, x, candidate_deletes); }
    }

    // Find the reverses
    // step 1: form the edges to reverse
    for (auto x: full_reverse_from_x) { add_pairs(reverse_x_y, x, pdag.get_parents(x)); }
    for (auto y: full_reverse_to_y) { add_pairs(reverse_x_y, pdag.get_children(y), y); }
    // step 2: find the reverses
    for (auto [x, y]: reverse_x_y) {
        // check that x ← y
        if (pdag.has_directed_edge(y, x) && x != y) {
            find_reverse_to_y_from_x(y, x, candidate_reverses);
        }
    }
    statistics["update_operators-time"] += measure_time(start_time);
}


/**
 * @brief Find and score all possible inserts to y.
 *
 * The candidate  Insert(x, y, T, E) are such that:
 *  1. x is not adjacent to y (x ∉ Ad(y))
 *  2. T ⊆ Ne(y) \ Ad(x)
 *  3. [Ne(y) ∩ Ad(x)] ∪ T is a clique
 *  Not enforced at that stage: [Ne(y) ∩ Ad(x)] ∪ T blocks all semi-directed paths from y to x
 *  5. E = [Ne(y) ∩ Ad(x)] ∪ T ∪ Pa(y)
 *
 * @param y The target node.
 * @param candidate_inserts The vector of candidate inserts to append inserts to.
 * @param parent_x The parent of y, if known. If -1, no pre-selection is done.
 * @param positive_only If true, only inserts with a positive score are considered.
 */
void XGES::find_inserts_to_y(int y, std::vector<Insert> &candidate_inserts, int parent_x,
                             bool positive_only) const {
    if (parent_x != -1) {
        find_inserts_to_y_from_x(y, parent_x, candidate_inserts, positive_only);
        return;
    }
    const auto &adjacent_y = pdag.get_adjacent(y);
    for (int x: pdag.get_nodes_variables()) {
        if (x != y && !adjacent_y.contains(x)) {
            find_inserts_to_y_from_x(y, x, candidate_inserts, positive_only);
        }
    }
}

void XGES::find_inserts_to_y_from_x(int y, int x, std::vector<Insert> &candidate_inserts,
                                    bool positive_only) const {
    const auto &parents_y = pdag.get_parents(y);
    // 3. [Ne(y) ∩ Ad(x)] ∪ T is a clique
    // So in particular, [Ne(y) ∩ Ad(x)] must be a clique.
    auto neighbors_y_adjacent_x = pdag.get_neighbors_adjacent(y, x);
    if (!pdag.is_clique(neighbors_y_adjacent_x)) { return; }

    // 2. T ⊆ Ne(y) \ Ad(x)
    auto neighbors_y_not_adjacent_x = pdag.get_neighbors_not_adjacent(y, x);

    // We enumerate all T ⊆ Ne(y) \ Ad(x) such that [Ne(y) ∩ Ad(x)] ∪ T is a clique (noted C(x,y,T)).
    // If the condition fails for some T, it will fail for all its supersets.
    // Hence, we enumerate the T in inclusion order, to minimize the number of T we need to check.
    // We simulate a recursive search using a stack.
    // The stack contains valid T, and an iterator for the next entries in neighbors_y_not_adjacent_x to consider.

    // The effective parents_y are [Ne(y) ∩ Ad(x)] ∪ T ∪ Pa(y).
    // In a CPDAG every parent of y is adjacent to every neighbor of y;
    // including Pa(y) in the subsequent clique checks is therefore harmless.
    FlatSet &effective_parents_y =
            neighbors_y_adjacent_x;// just renaming it, no copy necessary
    effective_parents_y.insert(parents_y.begin(), parents_y.end());
    // <T: set of nodes, iterator over neighbors_y_not_adjacent_x, effective_parents: set of nodes>
    std::stack<std::tuple<FlatSet, FlatSet::iterator, FlatSet>> stack;
    // we know that T = {} is valid
    stack.emplace(FlatSet{}, neighbors_y_not_adjacent_x.begin(), effective_parents_y);

    while (!stack.empty()) {
        auto top = std::move(stack.top());
        stack.pop();
        auto &T = std::get<0>(top);
        auto it = std::get<1>(top);
        auto &effective_parents = std::get<2>(top);

        double score = scorer->score_insert(y, effective_parents, x);
        if (score > 0 || !positive_only) {
            // Keep these sets intact: the subset expansion below still uses them.
            candidate_inserts.emplace_back(x, y, T, score, effective_parents);
            std::push_heap(candidate_inserts.begin(), candidate_inserts.end());
        }

        // Look for other candidate T using the iterator, which gives us the next elements z to consider.
        while (it != neighbors_y_not_adjacent_x.end()) {
            // We define T' = T ∪ {z} and we check C(x,y,T') is a clique.
            // Since C(x,y,T) was a clique, we only need to check that z is adjacent to all nodes in C(x,y,T).
            auto z = *it;
            ++it;
            auto &adjacent_z = pdag.get_adjacent(z);
            // We check that C(x,y,T) ⊆ Ad(z); i.e. T ⊆ Ad(z) and [Ne(y) ∩ Ad(x)] ⊆ Ad(z).
            if (std::ranges::includes(adjacent_z, T) &&
                std::ranges::includes(adjacent_z, neighbors_y_adjacent_x)) {
                // T' is a candidate
                auto T_prime = T;
                T_prime.insert(z);
                auto effective_parents_prime = effective_parents;
                effective_parents_prime.insert(z);
                stack.emplace(std::move(T_prime), it, std::move(effective_parents_prime));
            }
        }
    }
}

/** Find all possible deletes to y from x.
 *
 * The candidate deletes (x, y, C, E) are such that:
 *  1. x is a parent or a neighbor of y
 *  2. C ⊆ [Ne(y) ∩ Ad(x)]
 *  3. C is a clique
 *  4. E = Pa(y) ∪ C ∪ {x}
 *
 *
 * @param y The target node.
 * @param x The source node.
 * @param candidate_deletes The vector of candidate deletes to append deletes to.
 * @param positive_only If true, only deletes with a positive score are considered.
 */
void XGES::find_delete_to_y_from_x(int y, int x, std::vector<Delete> &candidate_deletes,
                                   bool positive_only) const {
    const FlatSet &parents_y = pdag.get_parents(y);
    auto neighbors_y_adjacent_x = pdag.get_neighbors_adjacent(y, x);
    bool directed_xy = pdag.has_directed_edge(x, y);

    // find all possible C ⊆ [Ne(y) ∩ Ad(x)] that are cliques
    // <C set of nodes, iterator over neighbors_y_adjacent_x, set of effective_parents>
    std::stack<std::tuple<FlatSet, FlatSet::iterator, FlatSet>> stack;
    FlatSet effective_parents_init;
    effective_parents_init.reserve(parents_y.size() + neighbors_y_adjacent_x.size() + 1);
    // Include x explicitly: an undirected x-y edge does not put x in Pa(y).
    union_with_single_element(parents_y, x, effective_parents_init);
    // note that C = {} is valid
    stack.emplace(FlatSet{}, neighbors_y_adjacent_x.begin(), effective_parents_init);

    while (!stack.empty()) {
        auto top = std::move(stack.top());
        stack.pop();
        auto C = std::get<0>(top);
        auto it = std::get<1>(top);
        auto effective_parents = std::get<2>(top);

        double score = scorer->score_delete(y, effective_parents, x);
        if (score > 0 || !positive_only) {
            candidate_deletes.emplace_back(x, y, C, score, effective_parents,
                                           directed_xy);
            std::push_heap(candidate_deletes.begin(), candidate_deletes.end());
        }

        // Look for other candidate C using the iterator, which gives us the next elements
        // z to consider.
        while (it != neighbors_y_adjacent_x.end()) {
            // We define C' = C ∪ {z} and we check if C' is a clique.
            // Since C is a clique, we only need to check that z is adjacent to all nodes
            // in C.
            auto z = *it;
            ++it;
            auto &adjacent_z = pdag.get_adjacent(z);
            // We check C ⊆ Ad(z)
            if (std::ranges::includes(adjacent_z, C)) {
                // C' is a candidate
                auto C_prime = C;
                C_prime.insert(z);
                auto effective_parents_prime = effective_parents;
                effective_parents_prime.insert(z);
                stack.emplace(std::move(C_prime), it, std::move(effective_parents_prime));
            }
        }
    }
}

/** @brief Find all possible deletes to y.
 *
 * @see find_delete_to_y_from_x
 *
 * @param y The target node.
 * @param candidate_deletes The vector of candidate deletes to append deletes to.
 * @param positive_only If true, only deletes with a positive score are considered.
 */
void XGES::find_deletes_to_y(const int y, std::vector<Delete> &candidate_deletes,
                             bool positive_only) const {
    auto &neighbors_y = pdag.get_neighbors(y);
    auto &parents_y = pdag.get_parents(y);

    for (int x: parents_y) {
        find_delete_to_y_from_x(y, x, candidate_deletes, positive_only);
    }
    for (int x: neighbors_y) {
        find_delete_to_y_from_x(y, x, candidate_deletes, positive_only);
    }
}

void XGES::find_reverse_to_y(int y, std::vector<Reverse> &candidate_reverses) const {
    for (int x: pdag.get_children(y)) {
        find_reverse_to_y_from_x(y, x, candidate_reverses);
    }
}

/** @brief Find all possible reverses to y from x.
 *
 * @see find_inserts_to_y
 *
 * The candidate Reverse(x, y, T, E, F) are such that:
 * 1. x ← y
 * 2. T ⊆ Ne(y) \ Ad(x)
 * 3. [Ne(y) ∩ Ad(x)] ∪ T is a clique
 * 4. All semi-directed paths from y to x are blocked by [Ne(y) ∩ Ad(x)] ∪ T ∪ Ne(x)
 * 5. E = [Ne(y) ∩ Ad(x)] ∪ T ∪ Pa(y)
 * 6. F = Pa(x)
 *
 * A reverse is composed of a delete and an insert. The function first finds all possible
 * inserts to y from x using `find_inserts_to_y` and then evaluate the delete part.
 *
 */
void XGES::find_reverse_to_y_from_x(int y, int x,
                                    std::vector<Reverse> &candidate_reverses) const {
    if (!pdag.has_directed_edge(y, x)) { return; }
    std::vector<Insert> candidate_inserts;
    find_inserts_to_y(y, candidate_inserts, x, false);
    auto &parents_x = pdag.get_parents(x);
    if (candidate_inserts.empty()) { return; }
    const double delete_score = scorer->score_delete(x, parents_x, y);
    for (auto &insert: candidate_inserts) {
        double score = ScorerInterface::require_finite_score(insert.score + delete_score);
        if (score > 0) {
            candidate_reverses.emplace_back(insert, score, parents_x);
            std::push_heap(candidate_reverses.begin(), candidate_reverses.end());
        }
    }
}


const PDAG &XGES::get_pdag() const { return pdag; }
double XGES::get_score() const { return total_score; }
double XGES::get_initial_score() const { return initial_score; }


// ---------------------------------------------------------------------------
// LGES insertion strategies. See XGES.h for how they compose with this search.
// ---------------------------------------------------------------------------

const PDAG &XGES::current_dag() {
    if (!dag_cache || dag_cache_version != pdag_version) {
        const auto start_time = steady_clock::now();
        dag_cache = std::make_unique<PDAG>(pdag.get_dag_extension());
        dag_cache_version = pdag_version;
        statistics["lges-dag_extension-#calls"] += 1;
        statistics["lges-dag_extension-time"] += measure_time(start_time);
    }
    return *dag_cache;
}

/** @brief Is the ordered pair (x -> y) admissible under the active strategy?
 *
 * SAFE INSERT is the gate on its own. CONSERVATIVE INSERT additionally rejects
 * the pair if any *valid* insert for it scores below zero.
 *
 * The validity test is deliberately lazy. `find_inserts_to_y` enumerates the T
 * satisfying the clique condition and leaves the semi-directed path condition
 * to `PDAG::is_insert_valid`; since `&&` short-circuits, the path BFS runs only
 * for the operators that actually score below zero, and the scan stops at the
 * first one that is valid. Resolving the path condition for every T instead --
 * as `lges::score_valid_insert_operators` does -- would add path searches for
 * candidates that cannot establish a negative-score veto.
 *
 * Using `is_insert_valid` also means the extended search's forbidden edges are
 * honoured here, which the standalone LGES enumeration does not check.
 */
bool XGES::passes_gate(const int y, const int x) {
    if (insert_strategy == InsertStrategy::Standard) { return true; }
    const auto gate_start = steady_clock::now();
    bool passes;
    if (strict_lges) {
        const auto &dag = current_dag();
        std::vector<char> visited(n_variables, 0);
        std::vector<int> queue{y};
        visited[y] = 1;
        for (size_t head = 0; head < queue.size() && !visited[x]; ++head) {
            for (int child: dag.get_children(queue[head])) {
                if (!visited[child]) { visited[child] = 1; queue.push_back(child); }
            }
        }
        if (visited[x]) {
            statistics["lges-descendant_rejections"] += 1;
            return false;
        }
        auto parents = dag.get_parents(y);
        const double before = scorer->checked_local_score(y, parents);
        parents.insert(x);
        passes = scorer->checked_local_score(y, parents) > before;
    } else {
        passes = lges::check_legal_insert(current_dag(), *scorer, x, y);
    }
    statistics["lges-gate-#calls"] += 1;
    statistics["lges-gate-time"] += measure_time(gate_start);
    if (!passes) { statistics["lges-pairs_rejected_by_gate"] += 1; }
    return passes;
}

bool XGES::pair_is_discarded(const std::vector<Insert> &operators,
                             UnblockedPathsMap &unblocked_paths_map) {
    for (const auto &op: operators) {
        if (op.get_score() < 0 && pdag.is_insert_valid(op, unblocked_paths_map)) {
            statistics["lges-pairs_discarded"] += 1;
            return true;
        }
    }
    return false;
}

bool XGES::pair_allowed_by_strategy(const int y, const int x,
                                    UnblockedPathsMap &unblocked_paths_map) {
    if (!passes_gate(y, x)) { return false; }
    if (insert_strategy != InsertStrategy::Conservative) { return true; }

    std::vector<Insert> operators;
    find_inserts_to_y(y, operators, x, /*positive_only=*/false);
    return !pair_is_discarded(operators, unblocked_paths_map);
}

/** @brief Generate insert candidates for y under the active insertion strategy.
 *
 * Standard delegates directly to `find_inserts_to_y`. Safe first applies the
 * DAG gate to each pair. Conservative also enumerates non-positive gains and
 * discards the pair if a valid negative-gain operator exists.
 */
void XGES::find_strategy_inserts_to_y(const int y, std::vector<Insert> &candidate_inserts,
                                      UnblockedPathsMap &unblocked_paths_map,
                                      const int parent_x, const bool positive_only) {
    if (insert_strategy == InsertStrategy::Standard) {
        find_inserts_to_y(y, candidate_inserts, parent_x, positive_only);
        return;
    }

    std::vector<int> possible_parents;
    if (parent_x != -1) {
        possible_parents.push_back(parent_x);
    } else {
        const auto &adjacent_y = pdag.get_adjacent(y);
        for (const int x: pdag.get_nodes_variables()) {
            if (x != y && !adjacent_y.contains(x)) { possible_parents.push_back(x); }
        }
    }

    for (const int x: possible_parents) {
        if (!passes_gate(y, x)) { continue; }

        if (insert_strategy != InsertStrategy::Conservative) {
            find_inserts_to_y(y, candidate_inserts, x, positive_only);
            continue;
        }

        // CONSERVATIVE INSERT needs to see the pair's operators before deciding
        // whether to keep any, so enumerate once and reuse that enumeration for
        // both the decision and the candidates. Going through
        // `pair_allowed_by_strategy` here instead would enumerate the pair
        // twice.
        std::vector<Insert> operators;
        find_inserts_to_y(y, operators, x, /*positive_only=*/false);
        if (pair_is_discarded(operators, unblocked_paths_map)) { continue; }

        for (auto &op: operators) {
            if (positive_only && op.get_score() <= 0) { continue; }
            candidate_inserts.push_back(std::move(op));
            std::push_heap(candidate_inserts.begin(), candidate_inserts.end());
        }
    }
}

bool XGES::insert_allowed_by_strategy(const Insert &insert,
                                      UnblockedPathsMap &unblocked_paths_map) {
    if (insert_strategy == InsertStrategy::Standard) { return true; }
    if (pair_allowed_by_strategy(insert.get_y(), insert.get_x(), unblocked_paths_map)) {
        return true;
    }
    statistics["lges-stale_candidates_rejected"] += 1;
    return false;
}
