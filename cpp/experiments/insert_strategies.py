#!/usr/bin/env python3
"""Compare the LGES insertion strategies against the standard one, in GES,
XGES-0 and XGES.

Nine configurations: {GES, XGES-0, XGES} x {standard, SafeInsert,
ConservativeInsert}, all from the one C++ binary via `--baseline/-0` and
`--insert`. This script is the Python wrapper around it: it generates the data,
drives the runs, scores the output against the true CPDAG, and plots.

Generate Erdos-Renyi DAGs with p variables and `--er` * p edges in expectation.
Draw weights from U([-2,-0.5] u [0.5,2]) and normalize each nonzero row to unit
l1 norm unless --raw-weights is set. Noise variances are drawn from U([0.1,0.5]).
Samples are linear-Gaussian. Scores use BIC with alpha=1.0, giving a
0.5*log(n) penalty per parameter; the CLI default is 2.0.

Metrics, all against the true CPDAG:
  runtime                 search time reported by the binary, excluding process
                          start-up and I/O
  SHD                     excess + missing + incorrect orientations
  excess adjacencies      edges in the estimate whose pair is non-adjacent in truth
  missing adjacencies     edges in truth whose pair is non-adjacent in the estimate
  incorrect orientations  pairs adjacent in both, oriented differently

Usage:
    python3 experiments/insert_strategies.py            # run and plot
    python3 experiments/insert_strategies.py --plot-only
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from python_support import default_lges_repo
BINARY = os.path.join(REPO, "build", "src-cpp", "xges")

# (label, extra CLI flags) -- the three search algorithms.
ALGORITHMS = [
    ("GES", ["--baseline", "ges"]),
    ("XGES-0", ["-0"]),
    ("XGES", []),
]
# (label, --insert value, colour, marker)
STRATEGIES = [
    ("Standard", "standard", "#4C72B0", "o"),
    ("SafeInsert", "safe", "#DD8452", "s"),
    ("ConservativeInsert", "conservative", "#55A868", "^"),
]
METRICS = [
    ("runtime", "Runtime (s)"),
    ("shd", "SHD"),
    ("excess", "Excess adjacencies"),
    ("missing", "Missing adjacencies"),
    ("misoriented", "Incorrect orientations"),
]
ALPHA = 1.0


def load_dag_to_cpdag(lges_repo):
    """Load the DAG-to-CPDAG utility through the fixture module loader."""
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from gen_fixtures import load_lges

    _, utils, _ = load_lges(lges_repo)
    return utils.dag_to_cpdag


def sample_er_dag(p, er, rng):
    """Erdos-Renyi DAG with `er * p` edges in expectation, in topological order."""
    n_pairs = p * (p - 1) / 2
    prob = min(1.0, er * p / n_pairs)
    upper = np.triu(np.ones((p, p), dtype=bool), 1)
    return ((rng.random((p, p)) < prob) & upper).astype(int)


def sample_data(dag, n, rng, l1_normalise=True):
    """Sample a linear-Gaussian model with optional row-l1 weight normalization.

    When enabled, every nonzero weight row is rescaled to unit l1 norm.
    Node labels follow topological order; the manifest records both choices.
    """
    p = len(dag)
    magnitude = rng.uniform(0.5, 2.0, (p, p))
    sign = rng.choice([-1.0, 1.0], (p, p))
    W = dag * magnitude * sign
    if l1_normalise:
        norm = np.abs(W).sum(axis=1, keepdims=True)
        W = np.divide(W, norm, out=np.zeros_like(W), where=norm > 0)
    noise_sd = np.sqrt(rng.uniform(0.1, 0.5, p))

    X = np.zeros((n, p))
    for j in range(p):  # already topologically ordered
        X[:, j] = X @ W[:, j] + rng.normal(0, noise_sd[j], n)
    return np.ascontiguousarray(X, dtype=np.float64)


def read_cpdag(path, p):
    """Read a complete binary adjacency CSV, respecting its node-ID header."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) != p + 1 or any(len(row) != p for row in rows):
        raise ValueError(f"{path}: expected a header and a {p} x {p} adjacency matrix")
    header = [int(value) for value in rows[0]]
    if sorted(header) != list(range(p)):
        raise ValueError(f"{path}: invalid node-ID header")
    body = np.array([[int(value) for value in row] for row in rows[1:]], dtype=int)
    if not np.isin(body, [0, 1]).all() or np.diag(body).any():
        raise ValueError(f"{path}: adjacency must be binary with a zero diagonal")
    result = np.zeros((p, p), dtype=int)
    result[np.ix_(header, header)] = body
    return result


def compare(estimate, truth):
    """Excess / missing / misoriented counts against the true CPDAG."""
    p = len(truth)
    iu = np.triu_indices(p, 1)
    est_adj = ((estimate + estimate.T) > 0)[iu]
    true_adj = ((truth + truth.T) > 0)[iu]

    excess = int((est_adj & ~true_adj).sum())
    missing = int((~est_adj & true_adj).sum())

    both = est_adj & true_adj
    # Among pairs adjacent in both, count those whose orientation differs.
    est_pair = np.stack([estimate[iu], estimate.T[iu]], axis=1)
    true_pair = np.stack([truth[iu], truth.T[iu]], axis=1)
    differs = (est_pair != true_pair).any(axis=1)
    misoriented = int((both & differs).sum())

    return {"excess": excess, "missing": missing, "misoriented": misoriented,
            "shd": excess + missing + misoriented}


def run_one(data_path, out_dir, algo_flags, strategy, p, *, binary=BINARY, timeout=300):
    stem = hashlib.sha256(json.dumps([strategy, algo_flags]).encode()).hexdigest()[:16]
    out_path = os.path.join(out_dir, stem + ".graph")
    stats_path = os.path.join(out_dir, stem + ".stats")
    cmd = ([str(binary), "--input", str(data_path), "--output", out_path,
            "--stats", stats_path, "-a", str(ALPHA),
            "--insert", strategy, "-v0"] + algo_flags)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Search exceeded {timeout:g}s: {cmd}") from error
    if proc.returncode != 0:
        raise RuntimeError(f"Search exited {proc.returncode}: {cmd}\n{proc.stderr}\n{proc.stdout}")
    stats = {}
    with open(stats_path) as f:
        for line in f:
            key, separator, value = line.partition(",")
            key = key.strip()
            if not separator or not key or key in stats:
                raise ValueError(f"Malformed or duplicate statistic: {line.strip()}")
            stats[key] = float(value)
    if not all(math.isfinite(value) for value in stats.values()):
        raise ValueError("Search reported non-finite statistics")
    if not {"time", "score", "score check"}.issubset(stats):
        raise ValueError("Search omitted required time/score statistics")
    if stats["time"] < 0 or not math.isclose(stats["score"], stats["score check"],
                                           rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("Invalid search time or inconsistent score bookkeeping")
    return read_cpdag(out_path, p), stats


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_manifest(args, report_name):
    # Load just our lightweight import helper; the source hashes include edits
    # that a commit ID alone would miss.
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from gen_fixtures import reference_provenance
    source_files = sorted(Path(REPO).glob("src-cpp/*.*"))
    source_files += sorted(Path(HERE).glob("*.py"))
    source_files += [Path(REPO) / "tests" / "gen_fixtures.py", Path(REPO) / "python_support.py"]
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=10)
    return {
        "format_version": 1, "report": report_name,
        "settings": {"num_vars": args.num_vars, "num_samples": args.num_samples,
                     "num_graphs": args.num_graphs, "er": args.er, "seed": args.seed,
                     "alpha": ALPHA, "timeout": args.timeout,
                     "normalization": "none" if args.raw_weights else "always_row_l1",
                     "node_order": "topological", "rng": "SeedSequence([seed, p, graph]); PCG64"},
        "configurations": {"algorithms": ALGORITHMS,
                           "strategies": [(label, value) for label, value, _, _ in STRATEGIES]},
        "binary_sha256": sha256(args.binary),
        "source_commit": head.stdout.strip() if head.returncode == 0 else None,
        "source_files": {str(path.relative_to(REPO)): sha256(path) for path in source_files},
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "reference": reference_provenance(args.lges_repo),
    }


def write_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def row_key(row):
    return row["p"], row["graph"], row["algorithm"], row["strategy"]


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No results in {path}")
    required = {"p", "graph", "algorithm", "strategy", "runtime", "shd", "excess", "missing", "misoriented"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing result columns in {path}")
    seen = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Truncated or malformed result row in {path}")
        for key in row:
            if key in ("p", "graph", "true_edges"):
                row[key] = int(row[key])
            elif key not in ("algorithm", "strategy"):
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError(f"Non-finite result in {path}")
        key = row_key(row)
        if key in seen:
            raise ValueError(f"Duplicate result {key}")
        seen.add(key)
    return rows


def expected_keys(settings):
    return {(p, graph, algorithm, strategy)
            for p in settings["num_vars"] for graph in range(settings["num_graphs"])
            for algorithm, _ in ALGORITHMS for strategy, _, _, _ in STRATEGIES}


def run_experiment(args, dag_to_cpdag, *, report_name="insert_strategies", extra_metrics=None):
    """Checkpoint every completed configuration; retain recoverable partial runs.

    Resume requires identical settings, source hashes, imported modules, environment and
    binary. Each dataset has an independent seed, so skipped rows consume no RNG.
    """
    validate_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{report_name}.csv"
    meta_path = out_dir / f"{report_name}.json"
    partial_csv = out_dir / f"{report_name}.partial.csv"
    partial_meta = out_dir / f"{report_name}.partial.json"
    manifest = json.loads(json.dumps(run_manifest(args, report_name)))
    rows = []
    if args.resume:
        if not partial_csv.exists() or not partial_meta.exists():
            raise ValueError("No partial run to resume; omit --resume to start a run")
        previous = json.loads(partial_meta.read_text())
        if previous["manifest"] != manifest:
            raise ValueError("Cannot resume: settings, binary, sources or reference have changed")
        # Discard only the uncommitted suffix of a torn append. The manifest's
        # byte offset is advanced atomically after each flushed, fsynced row.
        committed_bytes = previous["checkpoint_bytes"]
        if partial_csv.stat().st_size < committed_bytes:
            raise ValueError("Partial CSV is shorter than its committed checkpoint")
        with partial_csv.open("r+b") as checkpoint:
            checkpoint.truncate(committed_bytes)
        if committed_bytes:
            rows = read_csv(partial_csv)
        if len(rows) != previous["completed_rows"]:
            raise ValueError("Partial row count does not match its checkpoint")
    elif partial_csv.exists() or partial_meta.exists():
        raise ValueError("A partial run already exists; use --resume or a different --out-dir")
    completed = {row_key(row) for row in rows}
    expected = expected_keys(manifest["settings"])
    if not completed.issubset(expected):
        raise ValueError("Partial results contain configurations outside this experiment")
    metadata = {"manifest": manifest, "status": "running", "completed_rows": len(rows),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "checkpoint_bytes": partial_csv.stat().st_size if args.resume else 0}
    write_json(metadata, partial_meta)
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="lges_exp_") as tmp, \
                partial_csv.open("a" if args.resume else "w", newline="") as checkpoint:
            writer = None
            data_path = os.path.join(tmp, "data.npy")
            for p in args.num_vars:
                for graph in range(args.num_graphs):
                    pending = [(algo, flags, strategy, value)
                               for algo, flags in ALGORITHMS for strategy, value, _, _ in STRATEGIES
                               if (p, graph, algo, strategy) not in completed]
                    if not pending:
                        continue
                    rng = np.random.default_rng(np.random.SeedSequence([args.seed, p, graph]))
                    dag = sample_er_dag(p, args.er, rng)
                    np.save(data_path, sample_data(dag, args.num_samples, rng,
                                                   l1_normalise=not args.raw_weights))
                    truth = dag_to_cpdag(dag)
                    for algo, flags, strategy, value in pending:
                        estimate, stats = run_one(data_path, tmp, flags, value, p,
                                                  binary=args.binary, timeout=args.timeout)
                        row = {"p": p, "graph": graph, "algorithm": algo, "strategy": strategy,
                               "runtime": stats["time"], "score": stats["score"],
                               "score_check": stats["score check"], "true_edges": int(dag.sum())}
                        row.update(compare(estimate, truth))
                        if extra_metrics:
                            row.update(extra_metrics(estimate, truth))
                        if writer is None:
                            writer = csv.DictWriter(checkpoint, fieldnames=list(row))
                            if not rows:
                                writer.writeheader()
                        writer.writerow(row)
                        checkpoint.flush()
                        os.fsync(checkpoint.fileno())
                        rows.append(row)
                        completed.add(row_key(row))
                        metadata.update(completed_rows=len(rows), checkpoint_bytes=checkpoint.tell())
                        write_json(metadata, partial_meta)
                print(f"  p={p:4d} done ({time.monotonic() - started:.1f}s elapsed)", flush=True)
        if completed != expected:
            raise ValueError("Experiment did not produce every requested configuration")
        metadata.update(status="complete", completed_rows=len(rows),
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        csv_sha256=sha256(partial_csv))
        write_json(metadata, partial_meta)
        os.replace(partial_csv, csv_path)
        os.replace(partial_meta, meta_path)
    except BaseException as error:
        # The persisted checkpoint is authoritative if an interrupt landed
        # between writing a row, committing its manifest, and updating memory.
        failed_metadata = json.loads(partial_meta.read_text())
        failed_metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                               error=str(error))
        write_json(failed_metadata, partial_meta)
        raise
    print(f"wrote {csv_path} ({len(rows)} rows); total wall time {time.monotonic() - started:.1f}s")
    return rows, metadata


def load_results(out_dir, report_name="insert_strategies"):
    csv_path = Path(out_dir) / f"{report_name}.csv"
    rows = read_csv(csv_path)
    meta_path = csv_path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if metadata is not None:
        if metadata["status"] != "complete" or metadata["csv_sha256"] != sha256(csv_path):
            raise ValueError("Result CSV and manifest do not match a completed run")
        expected = expected_keys(metadata["manifest"]["settings"])
    else:
        # Historical CSVs lack metadata. Validate their grid but do not invent
        # sample sizes, normalization or binary versions from CLI defaults.
        expected = {(r["p"], r["graph"], a, s) for r in rows
                    for a, _ in ALGORITHMS for s, _, _, _ in STRATEGIES}
    if {row_key(row) for row in rows} != expected:
        raise ValueError("Incomplete result grid")
    return rows, metadata


def result_caption(metadata):
    if metadata is None:
        return "Legacy results: generation settings and binary revision unrecorded"
    settings = metadata["manifest"]["settings"]
    return (f"ER-{settings['er']:g}, n={settings['num_samples']}, "
            f"{settings['num_graphs']} graphs per size, BIC alpha={settings['alpha']:g}; "
            f"weights: {settings['normalization']}")


def paired_summary(rows, path, metadata=None):
    """Paired comparison of each strategy against Standard.

    Every configuration runs on the same generated dataset, so differences can
    be taken graph by graph. That removes the between-graph variance which
    dominates the error bands in the plot, and is the sensitive test of whether
    a strategy changes anything at all.
    """
    sizes = sorted({row["p"] for row in rows})
    index = {(r["p"], r["graph"], r["algorithm"], r["strategy"]): r for r in rows}
    graphs = {p: sorted({r["graph"] for r in rows if r["p"] == p}) for p in sizes}
    lines = [result_caption(metadata), "Paired differences vs Standard (negative = the strategy is better).",
             "Mean over graphs +/- s.e.; * marks |mean| > 2 s.e.", ""]

    for metric, metric_label in METRICS:
        lines.append(f"== {metric_label} ==")
        lines.append("  {:<34}".format("") + "".join(f"{('p=' + str(p)):>16}"
                                                     for p in sizes))
        for algo_label, _ in ALGORITHMS:
            for strat_label, _, _, _ in STRATEGIES:
                if strat_label == "Standard":
                    continue
                cells = []
                for p in sizes:
                    diffs = [index[(p, g, algo_label, strat_label)][metric]
                             - index[(p, g, algo_label, "Standard")][metric]
                             for g in graphs[p]]
                    mean = float(np.mean(diffs))
                    err = (float(np.std(diffs, ddof=1)) / np.sqrt(len(diffs))
                           if len(diffs) > 1 else 0.0)
                    mark = "*" if err > 0 and abs(mean) > 2 * err else " "
                    # Preserve small runtime differences instead of printing
                    # a rounded zero with a nonzero-effect marker.
                    cells.append((f"{mean:+.3g}+-{err:.2g}{mark}" if metric == "runtime"
                                  else f"{mean:+9.2f}+-{err:5.2f}{mark}").rjust(16))
                lines.append("  {:<34}".format(f"{algo_label} / {strat_label}")
                             + "".join(cells))
        lines.append("")

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(text)


def plot(rows, path, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted({row["p"] for row in rows})
    fig, axes = plt.subplots(len(METRICS), len(ALGORITHMS),
                             figsize=(4.1 * len(ALGORITHMS), 2.7 * len(METRICS)),
                             sharex=True, squeeze=False)

    for r, (metric, metric_label) in enumerate(METRICS):
        # One shared y-scale per metric row, so the three algorithms are
        # directly comparable against each other.
        row_axes = axes[r]
        for c, (algo_label, _) in enumerate(ALGORITHMS):
            ax = row_axes[c]
            for strat_label, _, colour, marker in STRATEGIES:
                means, errors = [], []
                for p in sizes:
                    values = [row[metric] for row in rows
                              if row["p"] == p and row["algorithm"] == algo_label
                              and row["strategy"] == strat_label]
                    means.append(np.mean(values))
                    errors.append(np.std(values, ddof=1) / np.sqrt(len(values))
                                  if len(values) > 1 else 0.0)
                means, errors = np.array(means), np.array(errors)
                if len(sizes) == 1:
                    # A filled band has zero width at one x value; show its
                    # uncertainty explicitly instead of dropping the interval.
                    ax.errorbar(sizes, means, yerr=errors, marker=marker, color=colour,
                                label=strat_label, markersize=4.5, capsize=3)
                else:
                    ax.plot(sizes, means, marker=marker, color=colour, label=strat_label,
                            markersize=4.5, linewidth=1.6)
                    ax.fill_between(sizes, means - errors, means + errors,
                                    color=colour, alpha=0.18, linewidth=0)

            ax.set_xticks(sizes)
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.tick_params(labelsize=8)
            if r == 0:
                ax.set_title(algo_label, fontsize=12, fontweight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            if r == len(METRICS) - 1:
                ax.set_xlabel("Number of variables $p$", fontsize=10)
            if metric == "runtime":
                ax.set_yscale("log")

        lo = min(ax.get_ylim()[0] for ax in row_axes)
        hi = max(ax.get_ylim()[1] for ax in row_axes)
        for ax in row_axes:
            ax.set_ylim(lo, hi)
            ax.tick_params(labelleft=True)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.004))
    fig.suptitle("LGES insertion strategies across three searches\n" +
                 result_caption(metadata) + " (mean $\\pm$ s.e.)", fontsize=11, y=0.998)
    fig.tight_layout(rect=(0, 0.028, 1, 0.972))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def argument_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-vars", type=int, nargs="+", default=[10, 20, 40, 60, 80, 100])
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--num-graphs", type=int, default=25)
    ap.add_argument("--er", type=float, default=2.0, help="expected edges per variable")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--raw-weights", action="store_true", help="skip unconditional row-l1 normalization")
    ap.add_argument("--lges-repo", default=default_lges_repo())
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--binary", default=BINARY, help="compiled xges executable")
    ap.add_argument("--timeout", type=float, default=300, help="maximum seconds per search process")
    ap.add_argument("--resume", action="store_true", help="continue an identical interrupted run")
    return ap


def validate_args(args):
    if not args.num_vars or any(p < 2 for p in args.num_vars) or len(set(args.num_vars)) != len(args.num_vars):
        raise ValueError("--num-vars must be distinct integers >= 2")
    if args.num_samples < 2 or args.num_graphs < 1 or args.seed < 0:
        raise ValueError("Require samples >= 2, graphs >= 1 and seed >= 0")
    if not math.isfinite(args.er) or args.er < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("Require finite ER >= 0 and timeout > 0")
    if not os.path.isfile(args.binary) or not os.access(args.binary, os.X_OK):
        raise ValueError(f"Executable not found: {args.binary}; build first or pass --binary")


def main():
    ap = argument_parser()
    ap.add_argument("--plot-only", action="store_true", help="replot using the saved run's metadata")
    ap.add_argument("--no-plot", action="store_true", help="save results and summary without matplotlib")
    args = ap.parse_args()
    try:
        if args.plot_only:
            if args.resume:
                raise ValueError("--plot-only and --resume cannot be combined")
            rows, metadata = load_results(args.out_dir)
        else:
            validate_args(args)
            rows, metadata = run_experiment(args, load_dag_to_cpdag(args.lges_repo))
        paired_summary(rows, os.path.join(args.out_dir, "paired_summary.txt"), metadata)
        if not args.no_plot:
            plot(rows, os.path.join(args.out_dir, "insert_strategies.png"), metadata)
    except (OSError, ValueError, RuntimeError) as error:
        ap.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
