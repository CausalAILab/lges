#include "BICScorer.h"
#include "DataIO.h"
#include "PDAG.h"
#include "XGES.h"
#include "utils.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>

#include "cxxopts.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace fs = std::filesystem;
using namespace std::chrono;

namespace {
    bool same_file(const fs::path &a, const fs::path &b) {
        std::error_code error;
        return fs::weakly_canonical(a) == fs::weakly_canonical(b) ||
               fs::equivalent(a, b, error);
    }

    void write_file(const fs::path &path, const std::string &contents) {
        std::ofstream output(path);
        if (!output) { throw std::runtime_error("Cannot open output: " + path.string()); }
        output << contents;
        output.close();
        if (!output) {
            throw std::runtime_error("Failed to write output: " + path.string());
        }
    }

    int run(int argc, char *argv[]) {
        cxxopts::Options options("xges",
                                 "Run XGES, LGES or a GES/OPS baseline on float64 NumPy data");
        options.add_options()("input", "Input 2-D float64 .npy file (C or Fortran order)",
                              cxxopts::value<std::string>())(
                "output", "Output adjacency CSV",
                cxxopts::value<std::string>()->default_value("xges-graph.txt"))(
                "alpha,a", "Finite positive BIC penalty multiplier",
                cxxopts::value<double>()->default_value("2"))(
                "stats", "Output statistics CSV",
                cxxopts::value<std::string>()->default_value("xges-stats.txt"))(
                "cache-size", "Maximum cached local scores; 0 disables caching",
                cxxopts::value<size_t>()->default_value(
                        std::to_string(BICScorer::default_cache_entries)))(
                "0,xges0", "Run XGES-0 without extended search",
                cxxopts::value<bool>()->default_value("false"))(
                "baseline,b", "Search mode: ops, ops-r, ges, ges-r, ges-t, lges0, lges, or lges+",
                cxxopts::value<std::string>()->default_value(""))(
                "strict-lges", "Use strict DAG gates and complete TURN selection in LGES modes",
                cxxopts::value<bool>()->default_value("false"))(
                "graph_truth,g",
                "DAG/PDAG adjacency CSV; report SHD between completed CPDAGs",
                cxxopts::value<std::string>())("verbose,v", "Verbosity (0-3)",
                                               cxxopts::value<int>()->default_value("1"))(
                "insert", "Insertion strategy: standard, safe, or conservative",
                cxxopts::value<std::string>()->default_value("standard"))("h,help",
                                                                          "Print usage");
        const auto args = options.parse(argc, argv);
        if (args.count("help")) {
            std::cout << options.help() << '\n';
            return 0;
        }
        if (!args.count("input")) {
            throw std::invalid_argument("--input is required; use --help for usage");
        }
        const auto baseline = args["baseline"].as<std::string>();
        if (!baseline.empty() && baseline != "ops" && baseline != "ops-r" &&
            baseline != "ges" && baseline != "ges-r" && baseline != "ges-t" &&
            baseline != "lges0" && baseline != "lges" && baseline != "lges+") {
            throw std::invalid_argument("Invalid baseline: " + baseline);
        }
        if (!baseline.empty() && args["xges0"].as<bool>()) {
            throw std::invalid_argument("--xges0 cannot be combined with --baseline");
        }
        if (args["strict-lges"].as<bool>() && baseline != "lges0" &&
            baseline != "lges" && baseline != "lges+") {
            throw std::invalid_argument("--strict-lges requires an LGES search mode");
        }
        const auto strategy_name = args["insert"].as<std::string>();
        InsertStrategy strategy;
        if (strategy_name == "standard") {
            strategy = InsertStrategy::Standard;
        } else if (strategy_name == "safe") {
            strategy = InsertStrategy::Safe;
        } else if (strategy_name == "conservative") {
            strategy = InsertStrategy::Conservative;
        } else {
            throw std::invalid_argument("Invalid insert strategy: " + strategy_name);
        }
        const int verbosity = args["verbose"].as<int>();
        if (verbosity < 0 || verbosity > 3) {
            throw std::invalid_argument("Verbosity must be 0-3");
        }
        auto logger = spdlog::stdout_color_mt("xges-cli");
        logger->set_pattern("[%^%l%$] %v");
        const spdlog::level::level_enum levels[] = {
                spdlog::level::off, spdlog::level::info, spdlog::level::debug,
                spdlog::level::trace};
        logger->set_level(levels[verbosity]);

        const fs::path input_path(args["input"].as<std::string>());
        const fs::path output_path(args["output"].as<std::string>());
        const fs::path stats_path(args["stats"].as<std::string>());
        if (same_file(output_path, stats_path) || same_file(input_path, output_path) ||
            same_file(input_path, stats_path)) {
            throw std::invalid_argument(
                    "Input, graph output and statistics paths must be distinct");
        }
        if (input_path.extension() != ".npy") {
            throw std::invalid_argument("Input must be a .npy file");
        }
        logger->info("Loading input: {}", input_path.string());
        Eigen::MatrixXd data = load_npy_data(input_path.string());
        const int n_variables = static_cast<int>(data.cols());
        logger->info("Input loaded. Shape: {} x {}", data.rows(), data.cols());
        const auto covariance_start = steady_clock::now();
        BICScorer scorer(data, args["alpha"].as<double>(),
                         args["cache-size"].as<size_t>());
        logger->info("Covariance computed in {} seconds", measure_time(covariance_start));
        data.resize(0, 0);// The scorer owns its covariance; release the sample matrix.
        XGES xges(n_variables, &scorer, logger);
        xges.set_insert_strategy(strategy);
        xges.set_strict_lges(args["strict-lges"].as<bool>());
        xges.statistics["config-strict_lges"] = args["strict-lges"].as<bool>();
        logger->info("Insertion strategy: {}", strategy_name);

        std::optional<PDAG> ground_truth;
        if (args.count("graph_truth")) {
            const fs::path truth_path(args["graph_truth"].as<std::string>());
            if (same_file(truth_path, output_path) || same_file(truth_path, stats_path)) {
                throw std::invalid_argument(
                        "Output paths must not overwrite the ground truth");
            }
            ground_truth = PDAG::from_file(truth_path.string()).get_cpdag();
            if (ground_truth->get_nodes_variables().size() !=
                static_cast<size_t>(n_variables)) {
                throw std::invalid_argument(
                        "Ground truth and data must have the same number of variables");
            }
        }

        // Search counters exclude covariance, the initial empty-graph score and the
        // final independent rescore. All extended trials are included.
        scorer.statistics.clear();
        const auto search_start = steady_clock::now();
        if (baseline == "ops" || baseline == "ops-r") {
            xges.fit_ops(baseline == "ops-r");
        } else if (baseline == "ges" || baseline == "ges-r") {
            xges.fit_ges(baseline == "ges-r");
        } else if (baseline == "ges-t") {
            xges.fit_ges_turn();
        } else if (baseline == "lges0") {
            xges.fit_lges0();
        } else if (baseline == "lges" || baseline == "lges+") {
            xges.fit_lges(baseline == "lges+");
        } else {
            xges.fit_xges(!args["xges0"].as<bool>());
        }
        const double elapsed_secs = measure_time(search_start);
        const auto scorer_statistics = scorer.statistics;
        const double score_check = scorer.score_pdag(xges.get_pdag());
        if (std::abs(score_check - xges.get_score()) >
            1e-8 + 1e-10 * std::abs(score_check)) {
            throw std::runtime_error(
                    "Accumulated score does not match an independent graph rescore");
        }
        logger->info("Search completed in {} seconds; score: {}", elapsed_secs,
                     xges.get_score());

        std::ostringstream stats;
        stats << std::setprecision(17) << "time, " << elapsed_secs << '\n'
              << "score, " << xges.get_score() << '\n'
              << "score check, " << score_check << '\n'
              << "score_empty, " << xges.get_initial_score() << '\n'
              << "score_increase, " << xges.get_score() - xges.get_initial_score()
              << '\n';
        if (ground_truth) {
            stats << "shd, " << xges.get_pdag().shd(*ground_truth, false) << '\n';
        }
        for (const auto &counters:
             {xges.statistics, xges.get_pdag().statistics, scorer_statistics}) {
            for (const auto &[name, value]: counters) {
                stats << name << ", " << value << '\n';
            }
        }
        write_file(output_path, xges.get_pdag().get_adj_string());
        write_file(stats_path, stats.str());
        return 0;
    }
}// namespace

int main(int argc, char *argv[]) {
    try {
        return run(argc, argv);
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
