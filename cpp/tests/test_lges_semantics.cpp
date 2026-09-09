// A scripted scorer exercises exact ties and zero gains that sampled fixtures
// rarely reach. Gate ties keep the pair; only a strictly negative valid gain
// triggers the Conservative pair veto.

#include "LGES.h"
#include "PDAG.h"
#include "ScorerInterface.h"

#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace {

    int g_failures = 0;
    int g_checks = 0;

    void check(bool condition, const std::string &what) {
        ++g_checks;
        if (!condition) {
            ++g_failures;
            std::cerr << "FAIL: " << what << "\n";
        }
    }

    /** A scorer whose local scores are dictated by the test. */
    class ScriptedScorer : public ScorerInterface {
    public:
        void set(int target, const std::vector<int> &parents, double score) {
            scores_[{target, parents}] = score;
        }

        double local_score(int target, const FlatSet &parents) override {
            const std::vector<int> key(parents.begin(), parents.end());
            const auto it = scores_.find({target, key});
            if (it == scores_.end()) {
                std::ostringstream os;
                os << "unscripted local_score(" << target << ", {";
                for (size_t i = 0; i < key.size(); ++i) {
                    os << (i ? "," : "") << key[i];
                }
                os << "})";
                throw std::runtime_error(os.str());
            }
            return it->second;
        }

    private:
        std::map<std::pair<int, std::vector<int>>, double> scores_;
    };

    /** Gate convention: reject strictly-lower only. */
    void test_gate_tie_semantics() {
        // G: 2 -> 1 (so Pa_G(1) = {2}); the pair under test is (0, 1).
        PDAG dag(3);
        dag.add_directed_edge(2, 1);

        struct Case {
            double with_x;
            double without_x;
            bool expected;
            const char *description;
        };
        const Case cases[] = {
                {-10.0, -11.0, true, "adding x raises the score -> keep"},
                {-11.0, -11.0, true, "adding x leaves the score unchanged -> keep (tie)"},
                {-11.5, -11.0, false, "adding x lowers the score -> reject"},
        };

        for (const auto &c: cases) {
            ScriptedScorer scorer;
            scorer.set(1, {2}, c.without_x);// s(y, Pa_G(y))
            scorer.set(1, {0, 2}, c.with_x);// s(y, Pa_G(y) ∪ {x})
            const bool actual = lges::check_legal_insert(dag, scorer, 0, 1);
            check(actual == c.expected, std::string("check_legal_insert: ") +
                                                c.description + " (got " +
                                                (actual ? "keep" : "reject") + ")");
        }
    }

    /** CONSERVATIVE INSERT convention: discard on strictly-negative only. */
    void test_conservative_zero_semantics() {
        // An edgeless PDAG, so the pair (0, 1) has exactly one valid operator: the
        // one with T = {}. Ne(1) is empty, hence T0 is empty, and with no edges at
        // all every semi-directed path condition holds vacuously.
        struct Case {
            double score;
            bool expect_kept;
            const char *description;
        };
        const Case cases[] = {
                {1.0, true, "positive score -> keep"},
                {0.0, true, "exactly zero -> keep"},
                {-0.0, true, "negative zero -> keep (-0.0 < 0 is false)"},
                {-1e-12, false, "barely negative -> discard"},
                {-3.0, false, "negative -> discard"},
        };

        for (const auto &c: cases) {
            PDAG pdag(3);
            ScriptedScorer scorer;
            scorer.set(1, {}, 0.0);     // s(y, effective parents)
            scorer.set(1, {0}, c.score);// s(y, effective parents ∪ {x})

            UnblockedPathsMap unblocked;
            const auto kept =
                    lges::conservative_inserts_for_pair(pdag, scorer, 0, 1, unblocked);
            check(kept.size() == (c.expect_kept ? 1u : 0u),
                  std::string("conservative_inserts_for_pair: ") + c.description +
                          " (got " + std::to_string(kept.size()) + " operators)");

            // The unpruned enumeration always keeps the operator, whatever its sign.
            UnblockedPathsMap unblocked_all;
            const auto all = lges::score_valid_insert_operators(pdag, scorer, 0, 1,
                                                                unblocked_all, false);
            check(all.operators.size() == 1 && !all.found_lower_scoring,
                  std::string("score_valid_insert_operators keeps it regardless: ") +
                          c.description);
        }
    }

    /** One negative operator discards the pair even when others score well. */
    void test_conservative_discards_whole_pair() {
        // 0 and 3 are the non-adjacent pair; 1 - 3 and 2 - 3 are undirected and
        // neither 1 nor 2 is adjacent to 0, so T ranges over the subsets of {1, 2}.
        // 1 - 2 is present so every subset is a clique.
        PDAG pdag(4);
        pdag.add_undirected_edge(1, 3);
        pdag.add_undirected_edge(2, 3);
        pdag.add_undirected_edge(1, 2);

        // NA_{3,0} = Ne(3) ∩ Ad(0) = {}, Pa(3) = {}, so the effective parents of
        // each operator are exactly T.
        const auto script = [](ScriptedScorer &s, double t_empty, double t1, double t2,
                               double t12) {
            s.set(3, {}, 0.0);
            s.set(3, {0}, t_empty);
            s.set(3, {1}, 0.0);
            s.set(3, {0, 1}, t1);
            s.set(3, {2}, 0.0);
            s.set(3, {0, 2}, t2);
            s.set(3, {1, 2}, 0.0);
            s.set(3, {0, 1, 2}, t12);
        };

        {
            ScriptedScorer scorer;
            script(scorer, 5.0, 4.0, 3.0, 2.0);// all positive
            UnblockedPathsMap unblocked;
            const auto kept =
                    lges::conservative_inserts_for_pair(pdag, scorer, 0, 3, unblocked);
            check(kept.size() == 4, "all-positive pair keeps all 4 operators (got " +
                                            std::to_string(kept.size()) + ")");
        }
        {
            ScriptedScorer scorer;
            script(scorer, 5.0, 4.0, -0.5, 2.0);// one negative among four
            UnblockedPathsMap unblocked;
            const auto kept =
                    lges::conservative_inserts_for_pair(pdag, scorer, 0, 3, unblocked);
            check(kept.empty(),
                  "a single negative operator discards the whole pair (got " +
                          std::to_string(kept.size()) + ")");

            // ... while the unpruned enumeration still reports all four.
            UnblockedPathsMap unblocked_all;
            const auto all = lges::score_valid_insert_operators(pdag, scorer, 0, 3,
                                                                unblocked_all, false);
            check(all.operators.size() == 4,
                  "unpruned enumeration is unaffected by the negative operator (got " +
                          std::to_string(all.operators.size()) + ")");
        }
    }

    /** The semi-directed path condition actually rejects operators. */
    void test_blocking_condition_rejects() {
        // The directed chain 0 -> 1 -> 2 -> 3 leaves 0 and 3 non-adjacent.
        //
        // Pair (x=3, y=0): Ne(0) is empty, so NA_{0,3} = {} and T can only be {};
        // the blocking set is therefore empty and the semi-directed path
        // 0 -> 1 -> 2 -> 3 runs from y to x unobstructed. No operator is valid.
        //
        // Pair (x=0, y=3): Ne(3) is empty for the same reason, and 3 has no
        // outgoing or undirected edge, so no semi-directed path leaves it. The
        // single operator with T = {} is valid, over effective parents Pa(3) = {2}.
        PDAG pdag(4);
        pdag.add_directed_edge(0, 1);
        pdag.add_directed_edge(1, 2);
        pdag.add_directed_edge(2, 3);

        ScriptedScorer scorer;
        scorer.set(3, {2}, 0.0);
        scorer.set(3, {0, 2}, 1.0);

        UnblockedPathsMap unblocked_a;
        const auto blocked_direction = lges::score_valid_insert_operators(
                pdag, scorer, 3, 0, unblocked_a, false);
        check(blocked_direction.operators.empty(),
              "insert(3 -> 0) is invalid: 0 -> 1 -> 2 -> 3 is an unblocked semi-directed "
              "path (got " +
                      std::to_string(blocked_direction.operators.size()) + ")");

        UnblockedPathsMap unblocked_b;
        const auto open_direction = lges::score_valid_insert_operators(
                pdag, scorer, 0, 3, unblocked_b, false);
        check(open_direction.operators.size() == 1,
              "insert(0 -> 3) is valid: no semi-directed path leaves 3 (got " +
                      std::to_string(open_direction.operators.size()) + ")");
        if (open_direction.operators.size() == 1) {
            const auto &parents = open_direction.operators[0].get_effective_parents();
            check(std::vector<int>(parents.begin(), parents.end()) == std::vector<int>{2},
                  "insert(0 -> 3) scores against effective parents Pa(3) = {2}");
        }
    }

    /** Candidate pairs are exactly the ordered non-adjacent ones. */
    void test_non_adjacent_pairs() {
        PDAG pdag(4);
        pdag.add_directed_edge(0, 1);
        pdag.add_undirected_edge(2, 3);

        // Adjacency is symmetric regardless of edge orientation, so both (0,1) and
        // (1,0) are excluded, as are (2,3) and (3,2).
        const std::vector<std::pair<int, int>> expected = {
                {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 0}, {2, 1}, {3, 0}, {3, 1}};

        const auto pairs = lges::non_adjacent_pairs(pdag);
        check(pairs.size() == expected.size(),
              "non_adjacent_pairs count (got " + std::to_string(pairs.size()) +
                      ", expected " + std::to_string(expected.size()) + ")");
        check(pairs == expected, "non_adjacent_pairs contents and order");
    }

}// namespace

int main() {
    try {
        test_gate_tie_semantics();
        test_conservative_zero_semantics();
        test_conservative_discards_whole_pair();
        test_blocking_condition_rejects();
        test_non_adjacent_pairs();
    } catch (const std::exception &e) {
        std::cerr << "unexpected exception: " << e.what() << "\n";
        return 2;
    }

    std::cout << g_checks << " checks\n";
    if (g_failures > 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all checks passed\n";
    return 0;
}
