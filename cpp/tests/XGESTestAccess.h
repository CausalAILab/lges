#pragma once
#include "XGES.h"

// A narrow test adapter keeps private-access macros out of the regression tests.
struct XGESTestAccess {
    static void set_state(XGES &search, const PDAG &graph) {
        search.pdag = graph;
        ++search.pdag_version;
        search.dag_cache.reset();
        search.total_score = search.scorer->score_pdag(graph);
    }
    static std::vector<Insert> valid_inserts(XGES &search) {
        std::vector<Insert> candidates, valid;
        UnblockedPathsMap paths;
        for (int y: search.pdag.get_nodes_variables()) {
            search.find_strategy_inserts_to_y(y, candidates, paths);
        }
        for (const auto &op: candidates) {
            if (search.pdag.is_insert_valid(op, paths)) { valid.push_back(op); }
        }
        return valid;
    }
    static std::vector<PDAG> successors(XGES &search, char kind) {
        std::vector<PDAG> result;
        UnblockedPathsMap paths;
        EdgeModificationsMap changes;
        if (kind == 'D') {
            std::vector<Delete> candidates;
            for (int y: search.pdag.get_nodes_variables()) {
                search.find_deletes_to_y(y, candidates);
            }
            for (const auto &op: candidates) {
                if (!search.pdag.is_delete_valid(op)) { continue; }
                auto graph = search.pdag;
                changes.clear();
                graph.apply_delete(op, changes);
                result.push_back(std::move(graph));
            }
        } else if (kind == 'R') {
            std::vector<Reverse> candidates;
            for (int y: search.pdag.get_nodes_variables()) {
                search.find_reverse_to_y(y, candidates);
            }
            for (const auto &op: candidates) {
                if (!search.pdag.is_reverse_valid(op, paths)) { continue; }
                auto graph = search.pdag;
                changes.clear();
                graph.apply_reverse(op, changes);
                result.push_back(std::move(graph));
            }
        } else {
            for (const auto &op: valid_inserts(search)) {
                auto graph = search.pdag;
                graph.apply_insert(op, changes);
                result.push_back(std::move(graph));
            }
        }
        return result;
    }
    static void run_lges_blocked(XGES &search, int x, int y) {
        search.heuristic_lges(std::pair{x, y});
    }
    static double apply_best_undirected_turn(XGES &search) {
        auto turn = search.best_undirected_turn();
        if (!turn) { return 0; }
        search.apply_undirected_turn(*turn);
        return turn->score;
    }
    static bool gate(XGES &search, int x, int y) { return search.passes_gate(y, x); }
    static bool cache_is_current(XGES &search) {
        return search.current_dag() == search.pdag.get_dag_extension();
    }
};
