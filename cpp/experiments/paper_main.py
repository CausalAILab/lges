#!/usr/bin/env python3
"""Run a fixed observational grid with explicit search configurations and resumable records."""

from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, platform, subprocess, sys, time, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from python_support import default_lges_repo, resolve_lges_repo

NODES = [5, 10, 15, 20, 25, 35, 50, 75, 100, 150]
CONFIGS = {
    "GES": (
        ["--baseline", "ges-t", "--insert", "standard"],
        "forward → backward → full TURN, once; standard INSERT",
    ),
    "XGES-0": (
        ["--xges0", "--insert", "standard"],
        "delete > directed reverse > insert, reconsider after each move",
    ),
    "XGES": (["--insert", "standard"], "XGES-0 plus forced-deletion restarts"),
}
for mode, label in [("lges0", "LGES-0"), ("lges", "LGES"), ("lges+", "LGES+")]:
    for strategy in ("safe", "conservative"):
        schedule = {
            "lges0": "forward → full TURN → backward",
            "lges": "exhaust delete → exhaust full TURN → one insert; repeat",
            "lges+": "LGES plus pair-blocked forced-deletion trials, refresh after acceptance",
        }[mode]
        CONFIGS[f"{label} ({strategy.title()})"] = (
            ["--baseline", mode, "--insert", strategy, "--strict-lges"],
            schedule,
        )


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_utils(lges_repo=None):
    p = resolve_lges_repo(lges_repo) / "ges/ges/utils.py"
    spec = importlib.util.spec_from_file_location("graph_utils", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def dataset(folder, p, trial, lges_repo=None):
    import sempler
    from sempler.generators import dag_avg_deg

    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "dataset.json").exists():
        m = json.loads((folder / "dataset.json").read_text())
        for name, h in m["hashes"].items():
            if digest(folder / name) != h:
                raise ValueError("Dataset hash mismatch")
        return m
    seed = 42 + trial
    G = dag_avg_deg(p, 4, 1, 1, random_state=seed)
    rng = np.random.RandomState(seed)
    W = G * rng.choice([-1, 1], size=G.shape) * rng.uniform(0.5, 2, size=G.shape)
    means = rng.normal(0, 1, p)
    variances = rng.uniform(0.1, 0.5, p)
    normalized = False
    reason = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            data = sempler.LGANM(W, means, variances).sample(n=10000, random_state=seed)
    except (RuntimeWarning, np.linalg.LinAlgError) as error:
        normalized = True
        reason = str(error)
        sums = np.abs(W).sum(axis=1, keepdims=True)
        W = np.divide(W, sums, out=np.zeros_like(W), where=sums > 0)
        data = sempler.LGANM(W, means, variances).sample(n=10000, random_state=seed)
    if data.shape != (10000, p) or not np.isfinite(data).all():
        raise ValueError("Invalid generated data")
    truth = load_utils(lges_repo).dag_to_cpdag(G.astype(int))
    for name, array in [
        ("data.npy", data),
        ("dag.npy", G),
        ("weights.npy", W),
        ("means.npy", means),
        ("variances.npy", variances),
        ("truth.npy", truth),
    ]:
        np.save(folder / name, array)
    m = {
        "p": p,
        "trial": trial,
        "seed": seed,
        "samples": 10000,
        "expected_edges": 2 * p,
        "actual_edges": int(G.sum()),
        "normalized": normalized,
        "normalization_reason": reason,
        "hashes": {
            name: digest(folder / name)
            for name in [
                "data.npy",
                "dag.npy",
                "weights.npy",
                "means.npy",
                "variances.npy",
                "truth.npy",
            ]
        },
    }
    write_json(folder / "dataset.json", m)
    return m


def read_graph(path, p):
    a = np.loadtxt(path, delimiter=",", skiprows=1, dtype=int, ndmin=2)
    if a.shape != (p, p) or np.diag(a).any() or not np.isin(a, [0, 1]).all():
        raise ValueError("Malformed estimate")
    return a


def metrics(a, t):
    se = (a + a.T) > 0
    st = (t + t.T) > 0
    u = np.triu(np.ones(t.shape, dtype=bool), 1)
    excess = int((se & ~st & u).sum())
    missing = int((st & ~se & u).sum())
    wrong = int((se & st & ((a != t) | (a.T != t.T)) & u).sum())
    tp = int(((a != 0) & (t != 0)).sum())
    ne = int((a != 0).sum())
    nt = int((t != 0).sum())
    return {
        "shd": excess + missing + wrong,
        "excess": excess,
        "missing": missing,
        "misoriented": wrong,
        "precision": tp / ne if ne else float(nt == 0),
        "recall": tp / nt if nt else float(ne == 0),
        "f1": 2 * tp / (ne + nt) if ne + nt else 1.0,
    }


def run_dataset(args, p, trial, algorithms):
    folder = args.output / "datasets" / f"p{p:03d}_trial{trial:02d}"
    meta = dataset(folder, p, trial, args.lges_repo)
    truth = np.load(folder / "truth.npy")
    results = []
    for label in algorithms:
        tag = (
            label.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "plus")
        )
        out = folder / tag
        out.mkdir(exist_ok=True)
        record = out / "result.json"
        if record.exists():
            results.append(json.loads(record.read_text()))
            continue
        row = {
            "p": p,
            "trial": trial,
            "algorithm": label,
            "data_sha256": meta["hashes"]["data.npy"],
            "status": "ok",
        }
        started = time.time()
        try:
            flags = CONFIGS[label][0]
            command = [
                str(args.binary),
                "--input",
                str(folder / "data.npy"),
                "--alpha",
                "1",
                "--output",
                str(out / "graph.csv"),
                "--stats",
                str(out / "stats.csv"),
                "-v0",
                *flags,
            ]
            proc = subprocess.run(
                command, capture_output=True, text=True, timeout=args.timeout
            )
            (out / "stdout.txt").write_text(proc.stdout)
            (out / "stderr.txt").write_text(proc.stderr)
            if proc.returncode:
                raise RuntimeError(proc.stderr.strip())
            stats = {
                k.strip(): float(v) for k, v in csv.reader((out / "stats.csv").open())
            }
            if not all(np.isfinite(list(stats.values()))):
                raise ValueError("Nonfinite statistics")
            if abs(stats["score"] - stats["score check"]) > 1e-6:
                raise ValueError("Score mismatch")
            if label.startswith("LGES") and stats.get("config-strict_lges") != 1:
                raise ValueError("Strict LGES was not enabled")
            a = read_graph(out / "graph.csv", p)
            row["runtime"] = stats["time"]
            row["score"] = stats["score"]
            np.save(out / "cpdag.npy", a)
            row.update(metrics(a, truth))
            row["estimate_sha256"] = digest(out / "cpdag.npy")
        except Exception as error:
            row.update(status="failed", error=str(error))
        row["wall_seconds"] = time.time() - started
        write_json(record, row)
        results.append(row)
        print(
            f"p={p} trial={trial:02d} {label}: {row['status']} SHD={row.get('shd', '-')} time={row.get('runtime', 0):.3f}s",
            flush=True,
        )
    return results


def summarize(output):
    rows = [
        json.loads(p.read_text())
        for p in sorted((output / "datasets").glob("*/*/result.json"))
    ]
    fields = [
        "p",
        "trial",
        "algorithm",
        "status",
        "shd",
        "excess",
        "missing",
        "misoriented",
        "precision",
        "recall",
        "f1",
        "runtime",
        "wall_seconds",
        "score",
        "error",
        "data_sha256",
        "estimate_sha256",
    ]
    with (output / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for p, label in sorted({(r["p"], r["algorithm"]) for r in rows}):
        group = [r for r in rows if r["p"] == p and r["algorithm"] == label]
        good = [r for r in group if r["status"] == "ok"]
        entry = {
            "p": p,
            "algorithm": label,
            "completed": len(good),
            "failed": len(group) - len(good),
        }
        for metric in ["shd", "runtime", "f1", "excess", "missing", "misoriented"]:
            values = np.array([r[metric] for r in good])
            entry[metric + "_mean"] = float(values.mean()) if len(values) else None
            entry[metric + "_se"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else None
            )
        summary.append(entry)
    write_json(output / "summary.json", summary)
    return rows, summary


def plot(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, summary = summarize(output)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for label in sorted({r["algorithm"] for r in summary}):
        subset = sorted(
            [r for r in summary if r["algorithm"] == label and r["completed"]],
            key=lambda r: r["p"],
        )
        for ax, metric in zip(axs, ["shd", "runtime"]):
            ax.errorbar(
                [r["p"] for r in subset],
                [r[metric + "_mean"] for r in subset],
                yerr=[r[metric + "_se"] or 0 for r in subset],
                marker=".",
                capsize=2,
                label=label,
            )
    for ax, title in zip(
        axs, ["CPDAG structural Hamming distance", "Search runtime (seconds)"]
    ):
        ax.set(xlabel="Variables", ylabel=title)
        ax.grid(alpha=0.2)
    axs[1].set_yscale("log")
    axs[0].legend(fontsize=7, ncol=2)
    manifest = json.loads((output / "manifest.json").read_text())
    fig.suptitle(
        f"Observational experiment: ER2, n=10,000, {manifest['trials']} graphs per size\nMean ± standard error; local C++ search runtimes"
    )
    fig.tight_layout()
    fig.savefig(output / "main_results.png", dpi=180)
    fig.savefig(output / "main_results.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--lges-repo", type=Path, default=default_lges_repo())
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=10000)
    ap.add_argument("--nodes", type=int, nargs="+", default=NODES)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    args.binary = args.binary.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(args.output)
        sys.exit()
    if (
        args.jobs < 1
        or args.trials < 1
        or not args.nodes
        or any(p < 5 for p in args.nodes)
        or len(set(args.nodes)) != len(args.nodes)
    ):
        ap.error("Require jobs/trials >= 1 and distinct node counts >= 5")
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        ap.error("Require a finite positive timeout")
    args.lges_repo = resolve_lges_repo(args.lges_repo)
    algorithms = list(CONFIGS)
    manifest = {
        "nodes": args.nodes,
        "trials": args.trials,
        "samples": 10000,
        "expected_edges_per_variable": 2,
        "sempler_average_degree": 4,
        "weight_magnitudes": [0.5, 2],
        "noise_mean_distribution": "N(0,1)",
        "noise_variances": [0.1, 0.5],
        "alpha": 1,
        "graph_and_sample_seed": "42 + trial",
        "normalization": "original weight rows scaled to unit l1 only if sempler raises a numerical warning/error",
        "algorithms": {
            k: {"flags": v[0], "schedule": v[1]} for k, v in CONFIGS.items()
        },
        "binary_sha256": digest(args.binary),
        "runner_sha256": digest(__file__),
        "sources": {
            str(p.relative_to(ROOT)): digest(p)
            for p in sorted((ROOT / "src-cpp").glob("*"))
            if p.is_file()
        },
        "python": sys.version,
        "platform": platform.platform(),
        "jobs": args.jobs,
        "timeout_seconds": args.timeout,
        "definition_source": "LGES published paper, main Fig.4a / D.1.1",
        "runtime_scope": "C++ library search timer, excluding covariance and final rescore; local shared hardware, not original Python-cluster timing",
        "reference_utils_sha256": digest(args.lges_repo / "ges/ges/utils.py"),
        "path_helper_sha256": digest(ROOT / "python_support.py"),
    }
    import importlib.metadata as md

    manifest["packages"] = {
        name: md.version(name)
        for name in ["numpy", "scipy", "sempler", "matplotlib", "networkx"]
    }
    mpath = args.output / "manifest.json"
    if mpath.exists() and json.loads(mpath.read_text()) != manifest:
        raise ValueError("Manifest differs; use a new output directory")
    write_json(mpath, manifest)
    for p in args.nodes:
        for trial in range(args.trials):
            dataset(
                args.output / "datasets" / f"p{p:03d}_trial{trial:02d}",
                p,
                trial,
                args.lges_repo,
            )
    print(f"Prepared {len(args.nodes) * args.trials} datasets", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(run_dataset, args, p, trial, algorithms)
            for p in args.nodes
            for trial in range(args.trials)
        ]
        for future in as_completed(futures):
            future.result()
            summarize(args.output)
    plot(args.output)
    rows, _ = summarize(args.output)
    failed = sum(r["status"] != "ok" for r in rows)
    print(f"FINISHED {len(rows)} rows, {failed} failures", flush=True)
    sys.exit(1 if failed else 0)
