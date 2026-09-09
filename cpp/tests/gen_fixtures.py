#!/usr/bin/env python3
"""Generate data and expected insertion results for the C++ fixture tests.

For each CPDAG, record a DAG extension, every non-adjacent pair's gate decision,
valid insertion operators and scores, the negative-gain veto, and the best
operator under each strategy. Scores use biased covariance and alpha=1.
Source hashes, seeds and data hashes accompany the generated fixtures.

Usage: python3 tests/gen_fixtures.py [--lges-repo PATH] [--out DIR]
"""
import argparse
import importlib.util
import os
import sys
import types
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from python_support import default_lges_repo, resolve_lges_repo

ALPHA = 1.0  # Penalty multiplier used by the recorded score fixtures.


def load_lges(repo_root):
    """Load the scoring, graph and operator modules from repo_root by file path.

    Bypass package initialization to avoid loading the unused discrete scorer
    and its torch dependency.
    """
    repo_root = resolve_lges_repo(repo_root)
    pkg_dir = os.path.join(repo_root, "ges", "ges")
    if not os.path.isdir(pkg_dir):
        raise FileNotFoundError(f"{pkg_dir} not found; pass --lges-repo")

    for name, path in (("ges", os.path.join(repo_root, "ges")),
                       ("ges.ges", pkg_dir),
                       ("ges.ges.scores", os.path.join(pkg_dir, "scores"))):
        shell = types.ModuleType(name)
        shell.__path__ = [path]
        sys.modules[name] = shell

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load("ges.ges.scores.decomposable_score",
         os.path.join(pkg_dir, "scores", "decomposable_score.py"))
    gauss = load("ges.ges.scores.gauss_obs_l0_pen",
                 os.path.join(pkg_dir, "scores", "gauss_obs_l0_pen.py"))
    utils = load("ges.ges.utils", os.path.join(pkg_dir, "utils.py"))
    main = load("ges.ges.main", os.path.join(pkg_dir, "main.py"))
    return main, utils, gauss.GaussObsL0Pen


def reference_provenance(repo_root):
    """Record commit, working-tree status and source hashes for the loaded modules."""
    root = Path(repo_root).resolve()
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    files = ["ges/ges/main.py", "ges/ges/utils.py",
             "ges/ges/scores/decomposable_score.py", "ges/ges/scores/gauss_obs_l0_pen.py"]
    return {"reference_commit": git("rev-parse", "HEAD"),
            "reference_dirty": bool(git("status", "--porcelain")),
            "reference_files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in files}}


def sample_data(dag, n, rng):
    """Linear-Gaussian data from `dag` (dag[i, j] != 0 => i -> j, topological)."""
    p = len(dag)
    W = dag * rng.uniform(0.5, 1.5, (p, p)) * rng.choice([-1.0, 1.0], (p, p))
    X = np.zeros((n, p))
    for j in range(p):
        X[:, j] = X @ W[:, j] + rng.normal(0, 1, n)
    return np.ascontiguousarray(X, dtype=np.float64)


def random_dag(p, density, rng):
    upper = np.triu(np.ones((p, p), dtype=int), 1)
    return ((rng.random((p, p)) < density) * upper).astype(int)


def build_cases(utils, rng):
    """(name, cpdag, dag_for_sampling, n_samples) tuples.

    The CPDAG a case is evaluated on is the CPDAG of a DAG, i.e. a state GES
    could actually be in. Hand-built cases pin down structures that random
    sparse graphs rarely produce: undirected cliques around the target (so the
    T enumeration has many valid subsets) and long semi-directed paths (so
    condition 3 actually rejects some T).
    """
    cases = []

    # Hand-built: a 4-clique of undirected edges among {1,2,3,4}, node 0 apart.
    clique = np.zeros((6, 6), dtype=int)
    for i in range(1, 5):
        for j in range(i + 1, 5):
            clique[i, j] = clique[j, i] = 1
    cases.append(("undirected_clique", clique, np.triu(clique, 1), 700))

    # Hand-built: a chain, whose CPDAG is fully undirected.
    chain = np.zeros((6, 6), dtype=int)
    for i in range(5):
        chain[i, i + 1] = chain[i + 1, i] = 1
    cases.append(("undirected_chain", chain, np.triu(chain, 1), 700))

    # Hand-built: a collider, so the CPDAG keeps directed edges.
    coll_dag = np.zeros((5, 5), dtype=int)
    coll_dag[0, 2] = coll_dag[1, 2] = coll_dag[2, 3] = coll_dag[3, 4] = 1
    cases.append(("collider", utils.dag_to_cpdag(coll_dag), coll_dag, 700))

    # Random CPDAGs across a range of sizes and densities. These are converged
    # states: nearly every INSERT is score-decreasing, so they exercise the
    # reject side of the gate and the discard side of CONSERVATIVE INSERT.
    for i, (p, density, n) in enumerate([(5, 0.4, 400), (6, 0.3, 500),
                                         (7, 0.25, 600), (8, 0.2, 800),
                                         (8, 0.45, 500), (9, 0.3, 700),
                                         (10, 0.2, 900), (10, 0.35, 600)]):
        dag = random_dag(p, density, rng)
        cases.append((f"random_{i}_p{p}", utils.dag_to_cpdag(dag), dag, n))

    # Mid-search states: the data comes from the full DAG but the CPDAG is that
    # of a subgraph, so score-increasing INSERTs exist. Without these, the
    # "keep the operators" branch of both strategies is never taken.
    for i, (p, density, n, keep) in enumerate([(6, 0.5, 600, 0.0), (6, 0.5, 600, 0.5),
                                               (7, 0.4, 700, 0.5), (8, 0.4, 800, 0.3),
                                               (9, 0.35, 700, 0.6)]):
        dag = random_dag(p, density, rng)
        edges = np.argwhere(dag != 0)
        partial = np.zeros_like(dag)
        for a, b in edges:
            if rng.random() < keep:
                partial[a, b] = 1
        cases.append((f"partial_{i}_p{p}", utils.dag_to_cpdag(partial), dag, n))

    return cases


def emit_matrix(out, key, M):
    out.append(key + " " + " ".join(str(int(v)) for v in np.asarray(M).ravel()))


def emit_ops(out, key, x, y, ops):
    """Serialize (score, new_A, x, y, T) tuples in deterministic subset order."""
    out.append(f"{key} {x} {y} {len(ops)}")
    # Sorted so the C++ side can compare without depending on enumeration order.
    for score, _, _, _, T in sorted(ops, key=lambda o: (sorted(o[4]), o[0])):
        members = " ".join(str(int(t)) for t in sorted(T))
        out.append(f"op {float(score)!r} {len(T)}" + (" " + members if members else ""))


def generate_search_regressions(out_dir):
    """Deterministic audit counterexamples for missed production candidates.

    The Safe case changes the chosen DAG parents at a target outside the local
    update set; the Conservative case removes a negative-operator veto.
    """
    recorded = {}
    for seed, name in [(12, "safe_refresh"), (4, "conservative_refresh")]:
        rng = np.random.default_rng(7000 + seed)
        p = [5, 8, 12, 16, 20][seed % 5]
        n = [50, 150, 500][seed % 3]
        dag = np.triu(rng.random((p, p)) < min(0.6, 2.5 / p), 1).astype(float)
        weights = dag * rng.uniform(0.2, 2, (p, p)) * rng.choice([-1.0, 1.0], size=(p, p))
        data = rng.normal(size=(n, p))
        for j in range(p):
            data[:, j] += data @ weights[:, j]
        data = data[:, rng.permutation(p)].copy()
        path = Path(out_dir) / f"{name}.npy"
        np.save(path, data)
        recorded[name] = {"seed": 7000 + seed, "p": p, "n": n,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(out_dir, "search_regressions.json").write_text(json.dumps(
        {"numpy": np.__version__, "cases": recorded}, indent=2, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--lges-repo",
                    default=default_lges_repo())
    ap.add_argument("--out", default=os.path.join(here, "fixtures"))
    ap.add_argument("--seed", type=int, default=20250812)
    args = ap.parse_args()

    lges_main, utils, GaussObsL0Pen = load_lges(args.lges_repo)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    provenance = reference_provenance(args.lges_repo)
    provenance.update(seed=args.seed, python=platform.python_version(), numpy=np.__version__)
    out = ["# generated by tests/gen_fixtures.py -- do not edit by hand",
           "# provenance " + json.dumps(provenance, sort_keys=True),
           f"alpha {ALPHA!r}"]
    n_cases = n_pairs = n_ops = 0

    for name, cpdag, sampling_dag, n in build_cases(utils, rng):
        p = len(cpdag)
        data = sample_data(sampling_dag, n, rng)
        data_file = f"{name}.npy"
        np.save(os.path.join(args.out, data_file), data)
        data_hash = hashlib.sha256(Path(args.out, data_file).read_bytes()).hexdigest()
        out.append(f"# data_sha256 {data_file} {data_hash}")

        cache = GaussObsL0Pen(data)
        # Supply the evaluation counter required by the gate function.
        metrics = {"inserts_eval": 0}
        G = utils.pdag_to_dag(cpdag)

        out.append(f"case {name}")
        out.append(f"p {p}")
        out.append(f"data {data_file}")
        emit_matrix(out, "cpdag", cpdag)
        emit_matrix(out, "dag", G)

        fro, to = np.where((cpdag + cpdag.T + np.eye(p)) == 0)
        pairs = sorted(zip(fro.tolist(), to.tolist()))

        safe_best, cons_best = [], []
        for x, y in pairs:
            gate = lges_main.check_legal_insert(cpdag, cache, metrics, x, y,
                                                G=G, score_based=True)
            out.append(f"gate {x} {y} {int(gate)}")

            valid = lges_main.score_valid_insert_operators(
                x, y, cpdag, cache, debug=0, prune=False)
            # Accept either an operator list or an (operators, found_lower) tuple.
            if isinstance(valid, tuple) and len(valid) == 2 and isinstance(valid[0], list):
                valid = valid[0]
            if not isinstance(valid, list):
                raise TypeError("Unexpected unpruned operator result from the reference")
            emit_ops(out, "valid", x, y, valid)

            pruned, found_lower = lges_main.score_valid_insert_operators(
                x, y, cpdag, cache, debug=0, prune=True)
            cons = [] if found_lower else pruned
            emit_ops(out, "cons", x, y, cons)

            n_pairs += 1
            n_ops += len(valid)
            if gate:
                safe_best += valid
                cons_best += cons

        for key, ops in (("best_safe", safe_best), ("best_cons", cons_best)):
            if not ops:
                out.append(f"{key} 0 0 1")
                continue
            scores = [o[0] for o in ops]
            best = max(scores)
            unique = sum(1 for s in scores if s == best) == 1
            out.append(f"{key} 1 {float(best)!r} {int(unique)}")

        out.append("endcase")
        n_cases += 1

    path = os.path.join(args.out, "lges_operators.txt")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    turn_lines = ["# Best positive undirected TURN gains.",
                  "# provenance " + json.dumps(provenance, sort_keys=True)]
    for line in out:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "p":
            p = int(tokens[1])
        elif tokens[0] == "data":
            data_file = tokens[1]
        elif tokens[0] == "cpdag":
            graph = np.array(list(map(int, tokens[1:]))).reshape(p, p)
            score = GaussObsL0Pen(np.load(Path(args.out) / data_file))
            operators = []
            for x, y in zip(*np.where((graph + graph.T) == 2)):
                operators.extend(lges_main.score_valid_turn_operators_undir(x, y, graph, score))
            best = max([0.0] + [float(op[0]) for op in operators])
            turn_lines.append(f"{p} {data_file} {best:.17g} " + " ".join(map(str, graph.ravel())))
    Path(args.out, "undirected_turns.txt").write_text("\n".join(turn_lines) + "\n")
    generate_search_regressions(args.out)
    print(f"wrote {path}: {n_cases} cases, {n_pairs} pairs, {n_ops} valid operators; 2 search regressions")


if __name__ == "__main__":
    main()
