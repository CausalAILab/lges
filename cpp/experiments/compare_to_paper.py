#!/usr/bin/env python3
"""Compare GES/XGES insertion strategies with 10,000 samples by default.

Weight normalization is unconditional by default, and node labels follow
topological order. Settings and source identities are saved in
compare_to_paper.json; per-configuration results are checkpointed to CSV.

F1 counts ordered edges: an undirected adjacency contributes both directions.
The empty/empty case is defined as perfect agreement (precision=recall=F1=1).
"""
import numpy as np

from insert_strategies import (ALGORITHMS, STRATEGIES, argument_parser,
                               load_dag_to_cpdag, load_results, result_caption,
                               run_experiment, validate_args)


def f1(estimate, truth):
    """Precision, recall and F1 over ordered edges, with explicit empty cases."""
    est = estimate != 0
    tru = truth != 0
    n_est, n_true = int(est.sum()), int(tru.sum())
    if n_est == 0 and n_true == 0:
        return 1.0, 1.0, 1.0
    tp = int((est & tru).sum())
    precision = tp / n_est if n_est else 0.0
    recall = tp / n_true if n_true else 0.0
    return precision, recall, 2 * tp / (n_est + n_true)


def extra_metrics(estimate, truth):
    precision, recall, score = f1(estimate, truth)
    return {"precision": precision, "recall": recall, "f1": score}


def print_table(rows, metadata):
    print(result_caption(metadata))
    for p in sorted({row["p"] for row in rows}):
        subset = [row for row in rows if row["p"] == p]
        print(f"\n=== p={p} ===")
        fields = ["shd", "excess", "missing", "misoriented", "f1", "runtime"]
        print(f"  {'configuration':34}" + "".join(f"{field:>12}" for field in fields))
        for algorithm, _ in ALGORITHMS:
            for strategy, _, _, _ in STRATEGIES:
                group = [row for row in subset if row["algorithm"] == algorithm and row["strategy"] == strategy]
                print(f"  {algorithm + ' / ' + strategy:34}" +
                      "".join(f"{np.mean([row[field] for row in group]):12.3f}" for field in fields))


def main():
    ap = argument_parser()
    ap.description = __doc__
    ap.set_defaults(num_vars=[50, 100, 150], num_samples=10000, num_graphs=5, seed=11)
    ap.add_argument("--report-only", action="store_true", help="print the saved result table")
    args = ap.parse_args()
    try:
        if args.report_only:
            if args.resume:
                raise ValueError("--report-only and --resume cannot be combined")
            rows, metadata = load_results(args.out_dir, "compare_to_paper")
        else:
            validate_args(args)
            rows, metadata = run_experiment(args, load_dag_to_cpdag(args.lges_repo),
                                             report_name="compare_to_paper", extra_metrics=extra_metrics)
        print_table(rows, metadata)
    except (OSError, ValueError, RuntimeError) as error:
        ap.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
