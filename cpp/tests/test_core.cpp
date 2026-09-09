#include "BICScorer.h"
#include "DataIO.h"
#include "LGES.h"
#include "XGES.h"
#include "XGESTestAccess.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/base_sink.h"
#include <mutex>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>

namespace fs = std::filesystem;
namespace {
    int checks = 0;
    void check(bool condition, const std::string &message) {
        ++checks;
        if (!condition) { throw std::runtime_error(message); }
    }
    template<class F>
    void throws(F &&action, const std::string &message) {
        bool caught = false;
        try {
            action();
        } catch (const std::exception &) { caught = true; }
        check(caught, message);
    }
    void near(double a, double b, const std::string &message) {
        check(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) < 1e-7, message);
    }
    struct TempDirectory {
        fs::path path =
                fs::temp_directory_path() /
                ("xges-core-" +
                 std::to_string(
                         std::chrono::steady_clock::now().time_since_epoch().count()));
        TempDirectory() { fs::create_directory(path); }
        ~TempDirectory() {
            std::error_code error;
            fs::remove_all(path, error);
        }
    };

    void graph_tests() {
        FlatSet out{9}, a{1, 3};
        union_with_single_element(a, 1, out);
        check(out == a, "union must replace output and deduplicate");
        union_with_single_element(out, 2, out);
        check(out == FlatSet({1, 2, 3}), "union must support output aliasing input");
        PDAG graph(6);
        graph.add_undirected_edge(0, 1);
        graph.add_undirected_edge(1, 2);
        graph.add_directed_edge(2, 3);
        graph.add_directed_edge(4, 3);
        check(graph.get_number_of_edges() == 4, "edge count");
        throws([&] { graph.add_undirected_edge(0, 1); }, "duplicate edge must throw");
        throws([&] { graph.add_directed_edge(0, 1); },
               "conflicting edge type must throw");
        throws([&] { graph.remove_directed_edge(0, 1); }, "wrong removal must throw");
        throws([&] { graph.add_directed_edge(0, 6); }, "out-of-range node must throw");
        throws([&] { graph.add_directed_edge(2, 2); }, "self-loop must throw");
        throws([&] { graph.get_parents(-1); }, "negative node must throw");
        check(graph.get_number_of_edges() == 4,
              "failed mutations must preserve edge count");
        throws([] { PDAG invalid(-1); }, "negative graph size must throw");
        throws([&] { graph.shd(PDAG(2)); }, "SHD must check graph sizes");
        UnblockedPathsMap paths;
        check(!graph.is_blocking_semi_directed_paths(0, 3, {}, paths), "reachable path");
        check(graph.is_blocking_semi_directed_paths(0, 3, {1}, paths), "blocked path");
        PDAG copied = graph;
        PDAG assigned(0);
        assigned = copied;
        for (int i = 0; i < 20; ++i) {
            check(!assigned.is_blocking_semi_directed_paths(0, 3, {}, paths),
                  "copied BFS storage");
        }
        TempDirectory temporary;
        const auto file = temporary.path / "graph.csv";
        {
            std::ofstream output(file);
            output << graph.get_adj_string();
        }
        check(PDAG::from_file(file.string()) == graph, "mixed graph CSV round trip");
        {
            std::ofstream output(file);
            output << PDAG(0).get_adj_string();
        }
        check(PDAG::from_file(file.string()) == PDAG(0), "empty graph round trip");
        const std::vector<std::string> invalid = {"",
                                                  "0,0\n0,1\n1,0\n",
                                                  "0,2\n0,1\n1,0\n",
                                                  "0,1\n0,1\n",
                                                  "0,1\n0,1,0\n1,0\n",
                                                  "0,1\n0,2\n1,0\n",
                                                  "0,1\n1,0\n0,0\n",
                                                  "0,1\n0,1\n1,0\n0,0\n",
                                                  "0,1\n0,1junk\n1,0\n",
                                                  "0,1,\n0,1\n1,0\n"};
        for (const auto &text: invalid) {
            {
                std::ofstream output(file);
                output << text;
            }
            throws([&] { (void) PDAG::from_file(file.string()); },
                   "malformed CSV must throw");
        }
        {
            std::ofstream output(file);
            output << "1,0\n0,1\n0,0\n";
        }
        const auto permuted = PDAG::from_file(file.string());
        check(permuted.has_directed_edge(1, 0), "CSV header controls node order");
        throws([&] { (void) PDAG::from_file((temporary.path / "missing").string()); },
               "missing graph must throw");
        PDAG cycle(4), chordless(4);
        for (int i = 0; i < 4; ++i) {
            cycle.add_directed_edge(i, (i + 1) % 4);
            chordless.add_undirected_edge(i, (i + 1) % 4);
        }
        check(!cycle.check_is_cpdag(), "directed cycle is not a CPDAG");
        check(!chordless.check_is_cpdag(), "nonchordal component is not a CPDAG");
        // Every topologically ordered five-node DAG: completion preserves the MEC.
        for (int mask = 0; mask < (1 << 10); ++mask) {
            PDAG dag(5);
            int bit = 0;
            for (int x = 0; x < 5; ++x) {
                for (int y = x + 1; y < 5; ++y, ++bit) {
                    if (mask & (1 << bit)) { dag.add_directed_edge(x, y); }
                }
            }
            const PDAG cpdag = dag.get_cpdag();
            check(cpdag.check_is_cpdag(), "DAG completion must return a CPDAG");
            check(cpdag.get_number_of_edges() == dag.get_number_of_edges(),
                  "completion must preserve skeleton");
            check(cpdag.get_dag_extension().get_cpdag() == cpdag,
                  "extension and completion must agree");
        }
        EdgeQueueSet queue;
        throws([&] { queue.pop(); }, "empty edge queue must throw");
        queue.push_undirected(1, 0);
        queue.push_undirected(0, 1);
        queue.pop();
        check(queue.empty(), "edge worklist deduplicates endpoints");
        EdgeModificationsMap changes;
        changes.update_edge_directed(2, 0, EdgeType::DIRECTED_TO_X);
        check(changes.begin()->second.get_change() == EdgeChange::Reverse,
              "canonical reverse transition");
        changes.clear();
        changes.update_edge_directed(2, 0, EdgeType::DIRECTED_TO_Y);
        check(changes.begin() == changes.end(),
              "identical edge type is not a modification");
        changes.update_edge_directed(2, 0, EdgeType::NONE);
        changes.update_edge_none(2, 0, EdgeType::DIRECTED_TO_Y);
        check(changes.begin() == changes.end(), "net no-op modifications cancel");
    }

    Eigen::MatrixXd data_matrix() {
        std::mt19937 generator(239);
        std::normal_distribution<double> normal;
        Eigen::MatrixXd data(180, 8);
        for (Eigen::Index i = 0; i < data.rows(); ++i) {
            for (Eigen::Index j = 0; j < data.cols(); ++j) {
                data(i, j) = normal(generator) + (j > 0 ? 0.8 * data(i, j - 1) : 0);
            }
        }
        return data;
    }
    struct InvalidScorer : ScorerInterface {
        double local_score(int, const FlatSet &) override {
            return std::numeric_limits<double>::quiet_NaN();
        }
    };
    void scorer_tests(const Eigen::MatrixXd &data) {
        BICScorer cached(data, 1, 3), uncached(data, 1, 0);
        for (int repeat = 0; repeat < 2; ++repeat) {
            for (int target = 1; target < data.cols(); ++target) {
                for (const FlatSet &parents: {FlatSet{}, FlatSet{0}}) {
                    near(cached.local_score(target, parents),
                         uncached.local_score(target, parents),
                         "cache eviction changes a score");
                }
            }
        }
        check(cached.statistics["local_score-cache-resets"] > 0,
              "cache eviction exercised");
        check(cached.statistics["local_score-cache-peak-entries"] <= 3,
              "cache entry budget");
        throws([&] { cached.local_score(-1, {}); }, "invalid target");
        throws([&] { cached.local_score(0, {0}); }, "self parent");
        throws([&] { cached.local_score(0, {99}); }, "invalid parent");
        throws([&] { cached.score_delete(0, {}, 1); }, "removing missing parent");
        throws([&] { BICScorer invalid(data, 0); }, "zero alpha");
        throws([&] { BICScorer invalid(data, std::numeric_limits<double>::infinity()); },
               "infinite alpha");
        throws([] { BICScorer invalid(Eigen::MatrixXd::Ones(5, 2), 1); },
               "constant columns");
        auto invalid_data = data;
        invalid_data(0, 0) = std::numeric_limits<double>::quiet_NaN();
        throws([&] { BICScorer invalid(invalid_data, 1); }, "NaN data");
        invalid_data = data;
        invalid_data.col(1) = invalid_data.col(0);
        BICScorer singular(invalid_data, 1);
        throws([&] { singular.local_score(1, {0}); }, "zero residual variance");
        throws([&] { singular.local_score(2, {0, 1}); }, "singular parent covariance");
        // A temporary data matrix is safe: the scorer owns its covariance.
        BICScorer owns_covariance(data_matrix(), 1);
        near(owns_covariance.local_score(2, {0}), uncached.local_score(2, {0}),
             "scorer data ownership");
        InvalidScorer invalid;
        throws([&] { XGES search(3, &invalid); }, "nonfinite custom scorer");
        throws([] { XGES search(3, nullptr); }, "null scorer");
        throws([&] { XGES search(0, &cached); }, "zero-variable search");
    }

    void statistics_consistent(const XGES &search, const BICScorer &scorer, int p) {
        const auto count = [](const auto &stats, const std::string &name) {
            const auto it = stats.find(name);
            return it == stats.end() ? 0.0 : it->second;
        };
        const double score_calls = count(scorer.statistics, "score_insert-#calls") +
                                   count(scorer.statistics, "score_delete-#calls");
        const double gates = count(search.statistics, "lges-gate-#calls");
        near(count(scorer.statistics, "local_score-#calls-total"),
             p + 2 * (score_calls + gates),
             "trial gate counters must include every local-score call exactly once");
    }
    // Set potentials define score-equivalent local scores while creating varied
    // finite search landscapes, independent of Gaussian sampling noise.
    class SetScorer : public ScorerInterface {
        std::vector<double> potentials;
    public:
        SetScorer(int nodes, unsigned seed) : potentials(1u << nodes) {
            std::mt19937 rng(seed);
            for (auto &value: potentials) { value = static_cast<double>(rng() % 10000); }
        }
        double local_score(int target, const FlatSet &parents) override {
            unsigned mask = 0;
            for (int parent: parents) { mask |= 1u << parent; }
            return potentials[mask | (1u << target)] - potentials[mask];
        }
    };

    class SearchTrace : public spdlog::sinks::base_sink<std::mutex> {
    public:
        XGES *search = nullptr;
        std::vector<std::pair<char, PDAG>> states;
    protected:
        void sink_it_(const spdlog::details::log_msg &msg) override {
            const std::string text(msg.payload.data(), msg.payload.size());
            if (!text.starts_with("LGES ")) { return; }
            const auto colon = text.find(": ");
            if (colon != std::string::npos) {
                states.emplace_back(text.at(colon + 2), search->get_pdag());
            }
        }
        void flush_() override {}
    };

    void lges_schedule_tests() {
        bool deferred_deletion = false;
        bool saw_reversal = false;
        for (unsigned seed: {0u, 1u, 2u, 3u, 52u}) {
            SetScorer scorer(6, seed), oracle_scorer(6, seed);
            auto trace = std::make_shared<SearchTrace>();
            auto logger = std::make_shared<spdlog::logger>("lges-trace", trace);
            logger->set_level(spdlog::level::debug);
            XGES search(6, &scorer, logger), oracle(6, &oracle_scorer);
            PDAG initial(6);
            std::mt19937 edges(seed + 10000);
            for (int x = 0; x < 6; ++x) {
                for (int y = x + 1; y < 6; ++y) {
                    if (edges() % 3 == 0) { initial.add_directed_edge(x, y); }
                }
            }
            initial = initial.get_cpdag();
            XGESTestAccess::set_state(search, initial);
            XGESTestAccess::set_state(oracle, initial);
            trace->search = &search;
            search.fit_lges(false);
            char phase = 'D';
            for (const auto &[kind, graph]: trace->states) {
                auto successors = XGESTestAccess::successors(oracle, phase);
                for (int skipped = 0; successors.empty() && skipped < 3; ++skipped) {
                    phase = phase == 'D' ? 'R' : phase == 'R' ? 'I' : 'D';
                    successors = XGESTestAccess::successors(oracle, phase);
                }
                check(!successors.empty() && kind == phase, "LGES phase order");
                double best_score = -std::numeric_limits<double>::infinity();
                bool legal = false;
                for (const auto &next: successors) {
                    best_score = std::max(best_score, oracle_scorer.score_pdag(next));
                    legal = legal || next == graph;
                }
                check(legal, "LGES applied an invalid phase operator");
                near(scorer.score_pdag(graph), best_score, "LGES must choose best phase gain");
                saw_reversal = saw_reversal || kind == 'R';
                if (phase != 'D' && !XGESTestAccess::successors(oracle, 'D').empty()) {
                    deferred_deletion = true;
                }
                XGESTestAccess::set_state(oracle, graph);
                if (phase == 'I') { phase = 'D'; }
            }
            for (char kind: {'D', 'R', 'I'}) {
                check(XGESTestAccess::successors(oracle, kind).empty(),
                      "LGES stopped before exhausting improving moves");
            }
            near(search.get_score(), scorer.score_pdag(search.get_pdag()), "LGES trace score");
        }
        check(saw_reversal, "phase tests must exercise reversals");
        check(deferred_deletion, "phase tests must distinguish LGES from deletion preemption");
    }

    void strict_lges_tests(const fs::path &fixtures) {
        std::ifstream cases(fixtures / "undirected_turns.txt");
        check(static_cast<bool>(cases), "missing undirected TURN fixture");
        std::string line;
        int positive = 0;
        while (std::getline(cases, line)) {
            if (line.empty() || line[0] == '#') { continue; }
            std::istringstream row(line);
            int p;
            std::string data_file;
            double expected;
            row >> p >> data_file >> expected;
            std::vector<int> adjacency(p * p);
            for (int &edge: adjacency) { row >> edge; }
            check(static_cast<bool>(row), "malformed TURN fixture");
            PDAG graph(p);
            for (int x = 0; x < p; ++x) {
                for (int y = x + 1; y < p; ++y) {
                    if (adjacency[x*p+y] && adjacency[y*p+x]) { graph.add_undirected_edge(x,y); }
                    else if (adjacency[x*p+y]) { graph.add_directed_edge(x,y); }
                    else if (adjacency[y*p+x]) { graph.add_directed_edge(y,x); }
                }
            }
            BICScorer scorer(load_npy_data((fixtures/data_file).string()), 1);
            XGES search(p, &scorer);
            XGESTestAccess::set_state(search, graph);
            const double before = scorer.score_pdag(graph);
            const double delta = XGESTestAccess::apply_best_undirected_turn(search);
            near(delta, expected, "undirected TURN gain differs from exhaustive fixture");
            near(scorer.score_pdag(search.get_pdag()) - before, delta, "TURN full-score delta");
            check(search.get_pdag().check_is_cpdag(), "TURN completion");
            positive += delta > 0;
        }
        check(positive > 0, "TURN fixtures must exercise positive moves");
        class ZeroScorer : public ScorerInterface {
            double local_score(int, const FlatSet &) override { return 0; }
        } zero;
        XGES search(3, &zero);
        search.set_insert_strategy(InsertStrategy::Safe);
        check(XGESTestAccess::gate(search, 0, 2), "compatibility gate keeps ties");
        search.set_strict_lges(true);
        check(!XGESTestAccess::gate(search, 0, 2), "strict gate rejects ties");
        PDAG dag(3);
        dag.add_directed_edge(0,1);
        dag.add_directed_edge(1,2);
        XGESTestAccess::set_state(search, dag);
        check(!XGESTestAccess::gate(search, 2, 0), "strict gate rejects descendants");
        check(search.statistics["lges-descendant_rejections"] == 1, "ancestry check exercised");
    }

    void lges_restart_tests(const fs::path &fixtures) {
        const auto data = load_npy_data((fixtures / "safe_refresh.npy").string());
        const int p = static_cast<int>(data.cols());
        for (auto strategy: {InsertStrategy::Standard, InsertStrategy::Safe,
                             InsertStrategy::Conservative}) {
            BICScorer scorer(data, 1);
            XGES search(p, &scorer);
            search.set_insert_strategy(strategy);
            search.fit_lges(false);
            const double before = search.get_score();
            (void) XGESTestAccess::cache_is_current(search);
            // Reset both counter sources after the first run and account for the
            // initial score expected by statistics_consistent below.
            search.statistics.clear();
            scorer.statistics.clear();
            (void) scorer.score_pdag(search.get_pdag());
            search.fit_lges(true);
            statistics_consistent(search, scorer, p);
            check(search.get_score() > before + 1e-7, "LGES+ must accept improving restart");
            check(search.statistics["extended_search-accepted"] > 0 &&
                          search.statistics["extended_search-rejected"] > 0,
                  "restart tests must cover both acceptance and rejection");
            near(search.statistics["extended_search-trials"],
                 search.statistics["extended_search-accepted"] +
                         search.statistics["extended_search-rejected"] +
                         search.statistics["extended_search-unchanged"],
                 "every restart must have exactly one outcome");
            check(XGESTestAccess::cache_is_current(search), "LGES+ cache after acceptance");
            check(search.get_pdag().check_is_cpdag(), "LGES+ completion");
            near(search.get_score(), scorer.score_pdag(search.get_pdag()), "LGES+ total score");
            for (int x = 0; x < p; ++x) {
                for (int y = 0; y < p; ++y) {
                    if (x != y) {
                        check(!search.get_pdag().is_insert_forbidden(x, y),
                              "LGES+ trial restriction leaked into accepted graph");
                    }
                }
            }
        }

        Eigen::MatrixXd pair_data(80, 2);
        for (int row = 0; row < pair_data.rows(); ++row) {
            pair_data(row, 0) = std::sin(row * 0.7);
            pair_data(row, 1) = pair_data(row, 0) + 0.1 * std::cos(row * 1.3);
        }
        BICScorer scorer(pair_data, 1);
        for (auto strategy: {InsertStrategy::Standard, InsertStrategy::Safe,
                             InsertStrategy::Conservative}) {
            XGES search(2, &scorer);
            search.set_insert_strategy(strategy);
            XGESTestAccess::run_lges_blocked(search, 0, 1);
            check(search.get_pdag().get_number_of_edges() == 0,
                  "blocked pair must exclude both insertion directions");
            search.fit_lges(false);
            check(search.get_pdag().get_number_of_edges() == 1,
                  "trial restriction must expire before an unrestricted search");
        }
    }

    void search_tests(const Eigen::MatrixXd &data, const fs::path &fixtures) {
        // No named logger registration: the standalone library must be usable.
        for (auto strategy: {InsertStrategy::Standard, InsertStrategy::Safe,
                             InsertStrategy::Conservative}) {
            for (int mode = 0; mode < 8; ++mode) {
                BICScorer scorer(data, 1);
                XGES search(static_cast<int>(data.cols()), &scorer);
                search.set_insert_strategy(strategy);
                if (mode < 2) {
                    search.fit_xges(mode == 1);
                } else if (mode < 4) {
                    search.fit_ops(mode == 3);
                } else if (mode < 6) {
                    search.fit_ges(mode == 5);
                } else {
                    search.fit_lges(mode == 7);
                }
                statistics_consistent(search, scorer, static_cast<int>(data.cols()));
                near(search.get_score(), scorer.score_pdag(search.get_pdag()),
                     "search score bookkeeping");
                check(search.get_pdag().check_is_cpdag(),
                      "search returned invalid CPDAG");
                check(XGESTestAccess::cache_is_current(search),
                      "search DAG cache must match graph");
            }
        }
        for (const auto &[filename, strategy]:
             {std::pair{"safe_refresh.npy", InsertStrategy::Safe},
              std::pair{"conservative_refresh.npy", InsertStrategy::Conservative}}) {
            const auto regression_data = load_npy_data((fixtures / filename).string());
            BICScorer scorer(regression_data, 1);
            const int p = static_cast<int>(regression_data.cols());
            XGES search(p, &scorer);
            search.set_insert_strategy(strategy);
            search.fit_xges(false);
            statistics_consistent(search, scorer, p);
            // This independent exhaustive oracle found positive inserts after the
            // old incremental updater stopped, on both saved counterexamples.
            PDAG graph = search.get_pdag();
            const auto dag = graph.get_dag_extension();
            const auto pairs = lges::non_adjacent_pairs(graph);
            UnblockedPathsMap paths;
            const auto best =
                    strategy == InsertStrategy::Safe
                            ? lges::get_safe_insert(graph, dag, scorer, pairs, paths)
                            : lges::get_conservative_insert(graph, dag, scorer, pairs,
                                                            paths);
            check(!best || best->get_score() <= 1e-8,
                  "XGES-0 stopped with an admissible improving insert");
            if (strategy == InsertStrategy::Safe) {
                (void) XGESTestAccess::cache_is_current(
                        search);// Populate before graph replacement.
                search.fit_xges(true);
                check(search.statistics["extended_search-accepted"] > 0,
                      "accepted trial regression must exercise graph replacement");
                check(XGESTestAccess::cache_is_current(search),
                      "accepted trial left stale DAG cache");
                XGES copy(search);
                search.fit_ops(false);
                copy.fit_ops(false);
                check(copy.get_pdag() == search.get_pdag(),
                      "reusing a search differs from a fresh-cache copy");
            }
        }
    }
}// namespace

int main(int argc, char **argv) {
    try {
        spdlog::set_level(spdlog::level::off);
        const fs::path fixtures = argc > 1
                                          ? fs::path(argv[1])
                                          : fs::path(__FILE__).parent_path() / "fixtures";
        graph_tests();
        const auto data = data_matrix();
        scorer_tests(data);
        search_tests(data, fixtures);
        lges_schedule_tests();
        lges_restart_tests(fixtures);
        strict_lges_tests(fixtures);
        std::cout << checks << " core checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL after " << checks << " checks: " << error.what() << '\n';
        return 1;
    }
}
