// Replay recorded CPDAG states, checking pair gates, valid insertion sets,
// scores and the best operator selected by each insertion strategy.
// tests/gen_fixtures.py generates the data and expected results together.

#include "BICScorer.h"
#include "LGES.h"
#include "PDAG.h"

#include "Eigen/Dense"
#include "DataIO.h"
#include "XGES.h"
#include "XGESTestAccess.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Eigen::MatrixXd;

// Local score subtraction and regression solves accumulate rounding error;
// compare recorded gains with an absolute tolerance.
constexpr double kTol = 1e-8;

// Decisions taken on the sign of a floating-point quantity (the gate's
// comparison of two local scores, CONSERVATIVE INSERT's `score < 0`) can only
// be reproduced exactly if the quantity is not itself within rounding distance
// of the boundary. The test reports how close it ever came.
constexpr double kBoundaryWarn = 1e-6;

namespace {

    int g_checks = 0;
    int g_failures = 0;
    double g_closest_gate_margin = std::numeric_limits<double>::infinity();
    double g_closest_score_to_zero = std::numeric_limits<double>::infinity();

    void fail(const std::string &context, const std::string &message) {
        if (++g_failures <= 25) {
            std::cerr << "FAIL [" << context << "] " << message << "\n";
        }
    }

    void check(bool condition, const std::string &context, const std::string &message) {
        ++g_checks;
        if (!condition) { fail(context, message); }
    }

    /** Build a PDAG from a flat row-major adjacency: A[i][j] != 0 means i -> j, and
 *  both directions set means i - j. */
    PDAG pdag_from_adjacency(const std::vector<int> &adjacency, int p) {
        PDAG pdag(p);
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < p; ++j) {
                const bool forward = adjacency[i * p + j] != 0;
                const bool backward = adjacency[j * p + i] != 0;
                if (!forward) { continue; }
                if (backward) {
                    if (i < j) { pdag.add_undirected_edge(i, j); }
                } else {
                    pdag.add_directed_edge(i, j);
                }
            }
        }
        return pdag;
    }

    /** An operator reduced to what the comparison cares about. */
    struct OpRecord {
        std::vector<int> T;
        double score;

        bool operator<(const OpRecord &rhs) const {
            if (T != rhs.T) { return T < rhs.T; }
            return score < rhs.score;
        }
    };

    std::vector<OpRecord> to_records(const std::vector<Insert> &operators) {
        std::vector<OpRecord> records;
        records.reserve(operators.size());
        for (const auto &op: operators) {
            records.push_back({std::vector<int>(op.get_T().begin(), op.get_T().end()),
                               op.get_score()});
        }
        std::sort(records.begin(), records.end());
        return records;
    }

    std::string describe(const std::vector<OpRecord> &records) {
        std::ostringstream os;
        os.precision(17);
        for (const auto &r: records) {
            os << " T={";
            for (size_t i = 0; i < r.T.size(); ++i) { os << (i ? "," : "") << r.T[i]; }
            os << "}:" << r.score;
        }
        return os.str();
    }

    void compare_ops(const std::string &context, const std::vector<OpRecord> &expected,
                     const std::vector<OpRecord> &actual) {
        ++g_checks;
        if (expected.size() != actual.size()) {
            fail(context, "operator count: python=" + std::to_string(expected.size()) +
                                  " cpp=" + std::to_string(actual.size()) +
                                  "\n  python:" + describe(expected) +
                                  "\n  cpp:   " + describe(actual));
            return;
        }
        for (size_t i = 0; i < expected.size(); ++i) {
            if (expected[i].T != actual[i].T) {
                fail(context, "T differs at index " + std::to_string(i) +
                                      "\n  python:" + describe(expected) +
                                      "\n  cpp:   " + describe(actual));
                return;
            }
            if (!std::isfinite(expected[i].score) || !std::isfinite(actual[i].score) ||
                !(std::abs(expected[i].score - actual[i].score) <= kTol)) {
                std::ostringstream os;
                os.precision(17);
                os << "score differs at index " << i << ": python=" << expected[i].score
                   << " cpp=" << actual[i].score
                   << " diff=" << std::abs(expected[i].score - actual[i].score);
                fail(context, os.str());
                return;
            }
        }
    }

    /** Line-oriented fixture reader. */
    class Fixture {
    public:
        explicit Fixture(const fs::path &path) : file_(path) {
            if (!file_) { throw std::runtime_error("cannot open " + path.string()); }
        }

        /** Advance to the next non-empty, non-comment line and return its keyword. */
        bool next() {
            std::string line;
            while (std::getline(file_, line)) {
                if (line.empty() || line[0] == '#') { continue; }
                tokens_.clear();
                std::istringstream ss(line);
                std::string token;
                while (ss >> token) { tokens_.push_back(token); }
                if (tokens_.empty()) { continue; }
                return true;
            }
            return false;
        }

        const std::string &keyword() const { return tokens_[0]; }
        size_t size() const { return tokens_.size(); }
        const std::string &str(size_t i) const { return tokens_.at(i); }
        int integer(size_t i) const { return std::stoi(tokens_.at(i)); }
        double real(size_t i) const { return std::stod(tokens_.at(i)); }

    private:
        std::ifstream file_;
        std::vector<std::string> tokens_;
    };

}// namespace

int main(int argc, char *argv[]) {
    const fs::path fixture_dir =
            argc > 1 ? fs::path(argv[1]) : fs::path(__FILE__).parent_path() / "fixtures";
    const fs::path fixture_file = fixture_dir / "lges_operators.txt";

    std::cout << "fixtures: " << fixture_file << "\n";
    Fixture fixture(fixture_file);

    double alpha = 1.0;
    int n_cases = 0, n_pairs = 0;
    int n_dag_compared = 0, n_dag_agreed = 0;

    // Per-case state, filled as the directives stream past.
    std::string case_name;
    int p = 0;
    MatrixXd data;
    std::unique_ptr<BICScorer> scorer;
    std::unique_ptr<PDAG> cpdag, dag;
    std::vector<Insert> safe_pool, cons_pool;
    bool gate_for_current_pair = false;

    while (fixture.next()) {
        // By value: reading the `op` continuation lines below rewrites the
        // token vector that keyword() refers into.
        const std::string key = fixture.keyword();

        if (key == "alpha") {
            alpha = fixture.real(1);
        } else if (key == "case") {
            case_name = fixture.str(1);
            safe_pool.clear();
            cons_pool.clear();
        } else if (key == "p") {
            p = fixture.integer(1);
        } else if (key == "data") {
            data = load_npy_data((fixture_dir / fixture.str(1)).string());
            scorer = std::make_unique<BICScorer>(data, alpha);
        } else if (key == "cpdag" || key == "dag") {
            std::vector<int> adjacency;
            adjacency.reserve(static_cast<size_t>(p) * p);
            for (size_t i = 1; i < fixture.size(); ++i) {
                adjacency.push_back(fixture.integer(i));
            }
            if (static_cast<int>(adjacency.size()) != p * p) {
                std::cerr << "malformed " << key << " row in case " << case_name << "\n";
                return 2;
            }
            (key == "cpdag" ? cpdag : dag) =
                    std::make_unique<PDAG>(pdag_from_adjacency(adjacency, p));
            if (key == "dag") {
                // Use the recorded DAG for gate checks: different valid extensions
                // can give different gate scores. Track whether the generated
                // extension matches it without requiring the same node order.
                ++n_dag_compared;
                if (cpdag->get_dag_extension() == *dag) { ++n_dag_agreed; }
            }
        } else if (key == "gate") {
            const int x = fixture.integer(1), y = fixture.integer(2);
            const bool expected = fixture.integer(3) != 0;
            const bool actual = lges::check_legal_insert(*dag, *scorer, x, y);
            check(actual == expected,
                  case_name + " gate(" + std::to_string(x) + "," + std::to_string(y) +
                          ")",
                  std::string("python=") + (expected ? "pass" : "reject") +
                          " cpp=" + (actual ? "pass" : "reject"));
            gate_for_current_pair = expected;

            // Track how close the gate's comparison ever came to a tie.
            FlatSet parents_with_x;
            union_with_single_element(dag->get_parents(y), x, parents_with_x);
            const double margin = std::abs(scorer->local_score(y, parents_with_x) -
                                           scorer->local_score(y, dag->get_parents(y)));
            g_closest_gate_margin = std::min(g_closest_gate_margin, margin);
            ++n_pairs;
        } else if (key == "valid" || key == "cons") {
            const int x = fixture.integer(1), y = fixture.integer(2);
            const int count = fixture.integer(3);

            std::vector<OpRecord> expected;
            expected.reserve(count);
            for (int i = 0; i < count; ++i) {
                if (!fixture.next() || fixture.keyword() != "op") {
                    std::cerr << "expected " << count << " op lines after " << key
                              << "\n";
                    return 2;
                }
                OpRecord record{{}, fixture.real(1)};
                const int t_size = fixture.integer(2);
                for (int t = 0; t < t_size; ++t) {
                    record.T.push_back(fixture.integer(3 + t));
                }
                std::sort(record.T.begin(), record.T.end());
                expected.push_back(std::move(record));
            }
            std::sort(expected.begin(), expected.end());

            UnblockedPathsMap unblocked;
            std::vector<Insert> actual_ops;
            if (key == "valid") {
                actual_ops = lges::score_valid_insert_operators(*cpdag, *scorer, x, y,
                                                                unblocked)
                                     .operators;
            } else {
                actual_ops = lges::conservative_inserts_for_pair(*cpdag, *scorer, x, y,
                                                                 unblocked);
            }
            for (const auto &op: actual_ops) {
                g_closest_score_to_zero =
                        std::min(g_closest_score_to_zero, std::abs(op.get_score()));
            }

            compare_ops(case_name + " " + key + "(" + std::to_string(x) + "," +
                                std::to_string(y) + ")",
                        expected, to_records(actual_ops));

            if (gate_for_current_pair) {
                auto &pool = (key == "valid") ? safe_pool : cons_pool;
                pool.insert(pool.end(), actual_ops.begin(), actual_ops.end());
            }
        } else if (key == "best_safe" || key == "best_cons") {
            const bool expected_found = fixture.integer(1) != 0;
            const double expected_score = fixture.real(2);

            UnblockedPathsMap unblocked;
            const auto candidates = lges::non_adjacent_pairs(*cpdag);
            const auto best =
                    (key == "best_safe")
                            ? lges::get_safe_insert(*cpdag, *dag, *scorer, candidates,
                                                    unblocked)
                            : lges::get_conservative_insert(*cpdag, *dag, *scorer,
                                                            candidates, unblocked);

            const std::string context = case_name + " " + key;
            check(best.has_value() == expected_found, context,
                  std::string("python found=") + (expected_found ? "yes" : "no") +
                          " cpp found=" + (best.has_value() ? "yes" : "no"));

            // Ties may select different operators, but their scores must agree.
            if (best.has_value() && expected_found) {
                std::ostringstream os;
                os.precision(17);
                os << "python=" << expected_score << " cpp=" << best->get_score();
                check(std::abs(best->get_score() - expected_score) <= kTol, context,
                      os.str());
            }

            // The pools accumulated pair by pair must select the same operator
            // as the whole-state routine, which is an internal consistency
            // check on get_safe_insert / get_conservative_insert.
            const auto &pool = (key == "best_safe") ? safe_pool : cons_pool;
            double pool_best = -std::numeric_limits<double>::infinity();
            for (const auto &op: pool) {
                pool_best = std::max(pool_best, op.get_score());
            }
            if (best.has_value()) {
                check(std::abs(pool_best - best->get_score()) <= kTol, context + " pool",
                      "per-pair maximum disagrees with selector");
            } else {
                check(pool.empty(), context + " pool",
                      "selector found nothing but pairs contributed operators");
            }
        } else if (key == "endcase") {
            // Exercise the production candidate generator, which uses lazy path
            // validity instead of LGES.cpp's exhaustive enumeration.
            for (auto strategy: {InsertStrategy::Safe, InsertStrategy::Conservative}) {
                XGES search(p, scorer.get());
                XGESTestAccess::set_state(search, *cpdag);
                search.set_insert_strategy(strategy);
                const auto actual = XGESTestAccess::valid_inserts(search);
                const PDAG production_dag = cpdag->get_dag_extension();
                for (const auto &[x, y]: lges::non_adjacent_pairs(*cpdag)) {
                    std::vector<Insert> expected_pair, actual_pair;
                    UnblockedPathsMap paths;
                    if (lges::check_legal_insert(production_dag, *scorer, x, y)) {
                        auto all = lges::score_valid_insert_operators(
                                *cpdag, *scorer, x, y, paths,
                                strategy == InsertStrategy::Conservative);
                        if (!all.found_lower_scoring) {
                            for (auto &op: all.operators) {
                                if (op.get_score() > 0) { expected_pair.push_back(op); }
                            }
                        }
                    }
                    for (const auto &op: actual) {
                        if (op.get_x() == x && op.get_y() == y) {
                            actual_pair.push_back(op);
                        }
                    }
                    compare_ops(case_name + " production(" + std::to_string(x) + "," +
                                        std::to_string(y) + ")",
                                to_records(expected_pair), to_records(actual_pair));
                }
            }
            ++n_cases;
        } else {
            std::cerr << "unknown directive '" << key << "'\n";
            return 2;
        }
    }

    std::cout << n_cases << " cases, " << n_pairs << " non-adjacent pairs, " << g_checks
              << " checks\n";
    std::cout.precision(6);
    std::cout << "closest gate margin to a tie: " << g_closest_gate_margin << "\n";
    std::cout << "operator score closest to zero: " << g_closest_score_to_zero << "\n";
    std::cout << "PDAG::get_dag_extension agrees with the Python DAG in " << n_dag_agreed
              << "/" << n_dag_compared << " cases\n";
    if (g_closest_gate_margin < kBoundaryWarn ||
        g_closest_score_to_zero < kBoundaryWarn) {
        std::cout << "note: a decision boundary was approached within " << kBoundaryWarn
                  << "; exact agreement there is not guaranteed by construction\n";
    }

    if (n_cases == 0) {
        std::cerr << "no cases were read\n";
        return 2;
    }
    if (g_failures > 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all checks passed\n";
    return 0;
}
