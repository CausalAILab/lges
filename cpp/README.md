# XGES with LGES-inspired insertion strategies

A C++ implementation of score-based graph structure learning from observational,
continuous data. It fits a linear-Gaussian model using a BIC score and returns a
CPDAG representing a Markov equivalence class.

**This codebase is based on the original [XGES C++ code](https://github.com/ANazaret/XGES)
by Achille Nazaret, associated with the work of Achille Nazaret and David Blei.
Its SafeInsert and ConservativeInsert options are inspired by
[LGES](https://github.com/CausalAILab/lges) by Adiba Ejaz and Elias Bareinboim.**
This repository combines those insertion filters with XGES, phased LGES-0/LGES/LGES+,
and GES/OPS search modes and includes local changes to validation, candidate maintenance, scoring,
and tests. It does not implement the complete LGES algorithm or claim to
reproduce all results or guarantees of that work.

## Build

Required:

- A C++23-capable compiler and CMake 3.26 or newer.
- Eigen3, Boost headers, and a BLAS implementation discoverable by CMake.
- Network access on the first configure to fetch the pinned spdlog archive; `cxxopts` is vendored.
- Python 3 for the test runner when `BUILD_TESTING` is enabled (the default).

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The executable is `build/src-cpp/xges`. To build only the library and CLI without
the Python test dependency, add `-DBUILD_TESTING=OFF` when configuring.
If dependencies are installed in a nonstandard prefix, pass
`-DCMAKE_PREFIX_PATH=/path/to/prefix`. Use a fresh build directory after moving a
checkout; CMake caches absolute paths.

## Quick start

Run XGES-0 with the Conservative insertion filter on a bundled sample dataset:

```sh
./build/src-cpp/xges \
  --input tests/fixtures/undirected_chain.npy \
  --xges0 --insert conservative --alpha 1 \
  --output graph.csv --stats stats.csv
```

Replace the input path with your own dataset for actual use. View all options:

```sh
./build/src-cpp/xges --help
```

### Input and output

Input is a two-dimensional `.npy` array with rows as samples, columns as variables,
and dtype `float64`. Both C and Fortran layouts and either byte order are
supported. Data must be finite and have at least two samples and one variable.
Constant columns and singular or numerically degenerate regressions are rejected;
the scorer does not silently regularize them.

The graph output is a CSV with a node-ID header followed by a square binary
adjacency matrix. For matrix `A`:

- `A[i,j] = 1`, `A[j,i] = 0` represents `i → j`.
- Both entries equal to 1 represent `i — j`.
- Both entries equal to 0 mean no edge.

Node IDs correspond to input column indices, starting at zero. The statistics
output contains search timings, scores, and operation/cache counters. Search
timing excludes loading and covariance construction. Output directories must
already exist; existing output files are overwritten.

Use `--graph_truth truth.csv` to compare the learned CPDAG with a supplied
DAG/PDAG in the same CSV format, completed to a CPDAG before computing SHD.

## Search modes and insertion strategies

Search order and insertion filtering are separate options.

| Search selection | Behavior |
| --- | --- |
| Default | XGES-0 followed by extended search that tries forced deletions and accepts improving restarts. |
| `--xges0` or `-0` | Prefer deletions, then reversals, then insertions; reconsider that order after each accepted move. |
| `--baseline ges` | A forward insertion phase followed by a backward deletion phase. |
| `--baseline ges-t` | One forward, backward, and complete turning phase, in that order. |
| `--baseline ges-r` | Repeat insertion, deletion, and reversal phases until no move improves the score. |
| `--baseline ops` | Choose the largest gain across insertions and deletions. |
| `--baseline ops-r` | Choose the largest gain across insertions, deletions, and reversals. |
| `--baseline lges0` | One forward, turning, and backward phase, in that order. |
| `--baseline lges` | Exhaust deletions, exhaust reversals, apply one insertion, then repeat. |
| `--baseline lges+` | Run the LGES heuristic, then try forced deletions with restricted LGES restarts. |

Do not combine `--xges0` with `--baseline`. All CLI searches start from an empty graph.
The insertion strategy remains a separate option, including for LGES modes:

```sh
./build/src-cpp/xges --input data.npy --baseline lges --insert safe
./build/src-cpp/xges --input data.npy --baseline lges+ --insert conservative
```

Using `--insert standard` (the default) runs the selected schedule without the
additional insertion filters.

| `--insert` | Pair eligibility |
| --- | --- |
| `standard` (default) | No additional pair filter beyond operator validity and positive gain. |
| `safe` | Keep a pair if adding the source to the target's parents in a cached DAG extension does not decrease its local score. |
| `conservative` | Apply the same DAG gate, then reject the entire pair if any valid insertion for it has negative gain. |

Without `--strict-lges`, the DAG gate keeps exact score ties and does not test ancestry. Conservative's
pair veto is strictly negative; a zero-gain candidate does not veto the pair.
The search itself applies only positive-gain operators. Reversals are not
filtered by either insertion strategy.

XGES uses three persistent candidate heaps. Safe and Conservative rebuild the insertion heap
after graph changes because pair eligibility can change beyond the immediate
neighborhood. This adds work and does not guarantee a speedup. The chosen DAG
extension is reused while the graph is unchanged; its parent sets affect gate
scores.

LGES rebuilds candidates for each move within the active phase. Its temporary
vectors use the shared heap-based enumerators, but candidates do not persist
across graph changes. Reversals are exhausted before one insertion is attempted;
a further cycle also runs when reversals changed the graph but no insertion was
available. LGES+ blocks both insertion directions for the trial pair, discards
that restriction after the trial, accepts gains exceeding 1e-7 in compatibility mode
(or any positive gain with `--strict-lges`), and rebuilds its
deletion trial set immediately after acceptance.

`--alpha` controls the BIC penalty multiplier and defaults to **2**. Experiments
and recorded operator fixtures use **1**, so set it explicitly when comparing
runs. `--cache-size` limits cached local scores (default: 100000 entries;
0 disables caching). `--verbose` accepts 0 through 3.

### Scope relative to LGES

The LGES and LGES+ schedules are implemented alongside the original XGES
schedule. Prior-knowledge insertion groups, public initial-CPDAG input, and
interventional I-ORIENT are not implemented. Prior-knowledge groups would be soft
preferences, separate from the temporary pair restriction used by LGES+ trials.
`--baseline lges0` implements the forward–turning–backward schedule of Algorithm 8.

For the observational schedules in the published LGES paper, add `--strict-lges`
to `lges0`, `lges`, or `lges+`. This requires a strictly positive DAG insertion
score and a nondescendant source, considers both directed and undirected TURN
operators, and accepts any strictly positive LGES+ restart gain. Conservative
insertion still vetoes a pair if any valid insertion has negative gain. Each
chosen move is the best positive valid move in its active phase.

```sh
./build/src-cpp/xges --input data.npy --baseline lges+ \
  --insert conservative --strict-lges --alpha 1
```

`ges-t` uses complete TURN selection too, without an LGES insertion filter by
default. It differs from `ges-r`, which repeats phases and uses directed reversals.
The standard `ges` mode remains forward–backward. See the
[experiment audit](experiments/PAPER_MAIN_AUDIT.md) for the exact baseline mapping,
normalization ambiguity, and limits on reproducing the published results.

## Tests

For all five test suites, install the development dependencies and run CTest:

```sh
python3 -m venv ../xges-lges-venv
. ../xges-lges-venv/bin/activate
python -m pip install -r requirements-dev.txt
cmake -S . -B build -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The suites cover recorded insertion operators, exact tie semantics, graph/search
and scoring regressions (including LGES phase order and restart isolation), CLI input/output, and experiment checkpoint handling.
The experiment suite is skipped if NumPy is unavailable; matplotlib is needed
only to plot experiment results. The example keeps its virtual environment
outside the checkout.

Recorded fixtures are included; running tests needs no sibling repository.
Regenerating fixtures imports scoring and graph utilities from a separate LGES
checkout and records its source hashes with the output:

```sh
python tests/gen_fixtures.py --lges-repo /path/to/lges
```

## Experiments

`experiments/paper_main.py` runs the observational ER2 experiment with nine named
GES/XGES/LGES configurations, fixed 10,000 samples, BIC alpha 1, and 50 datasets
per graph size. The following restricts graph sizes to at most 50 variables:

```sh
python experiments/paper_main.py --binary ./build/src-cpp/xges \
  --output /path/to/results --nodes 5 10 15 20 25 35 50 --trials 50 --jobs 4
```

Install `requirements-experiments.txt` for the pinned scientific dependencies.
The runner discovers the containing LGES checkout when installed as `lges/cpp`,
or a sibling `../lges` in standalone use. Pass `--lges-repo /path/to/lges` to
override discovery. Each completed search and dataset
is saved separately with hashes; repeating the command resumes only if the
manifest matches. Outputs include `results.csv`, `summary.json`, and plots.
This runner excludes PC, NOTEARS, interventions, and prior-knowledge experiments.
Its local C++ runtime measurements do not reproduce the paper's shared Python
implementation timings. Read [PAPER_MAIN_AUDIT.md](experiments/PAPER_MAIN_AUDIT.md)
before interpreting the results as a replication.

The older experiment scripts compare nine combinations of GES, XGES-0, and XGES with
Standard, SafeInsert, and ConservativeInsert on shared synthetic datasets. They
require the development dependencies and an LGES checkout for graph utilities.

```sh
python experiments/insert_strategies.py \
  --lges-repo /path/to/lges --num-vars 10 --num-graphs 2 \
  --num-samples 1000 --out-dir /tmp/xges-strategy-results
```

Results are checkpointed to CSV with configuration, source/binary hashes, seeds,
and environment metadata. `--resume` continues an interrupted run only when its
configuration and provenance match. `--plot-only` uses saved results;
`--no-plot` avoids the plotting dependency. Use `--binary` for a different build.

`experiments/compare_to_paper.py` uses 10000 samples by default and also reports
precision, recall, and F1 over ordered edges. Its historical filename does not
mean it reproduces the complete LGES experiment: weight rows are normalized
unconditionally by default and labels follow topological order. `--raw-weights`
disables normalization. Generated outputs in `experiments/results/` are ignored. Obsolete benchmark
artifacts were removed; run a new experiment before making performance claims.

## Standalone and monorepo layouts

The project can be built from its own root, or as `cpp/` within an LGES checkout.
From the LGES root:

```sh
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
python3 -m pip install -r cpp/requirements-dev.txt
ctest --test-dir cpp/build --output-on-failure
./cpp/build/src-cpp/xges --input /path/to/data.npy --baseline lges \
  --insert conservative --strict-lges --alpha 1
```

All helper defaults are resolved relative to the scripts, not the current working
directory. Explicit input/output arguments remain relative to your working
directory. Use `--binary` when building outside the default `build/` directory.
Recorded regression tests require no Python LGES checkout. Only fixture generation
and synthetic experiments use its graph/score utilities.

CMake downloads an exact spdlog archive with a SHA-256 check into the build tree.
For offline builds, unpack that archive elsewhere and pass
`-DFETCHCONTENT_SOURCE_DIR_XGES_SPDLOG=/path/to/spdlog`. There are no nested Git
repositories or submodules to initialize. Source/dependency notices are listed in
[THIRD_PARTY.md](THIRD_PARTY.md).

## Known differences and experiment interpretation

The strict LGES schedules follow the intended observational, no-prior definitions,
with the pseudocode ambiguities documented in the experiment audit. This is not a
complete port of prior/interventional LGES, or a claim of identical trajectories.

The inherited XGES extended search differs from its paper pseudocode: after an
accepted restart it finishes the old deletion list before regenerating candidates,
and insertion bans persist in accepted graphs. It uses a `1e-7` restart threshold;
XGES-0 accepts only positive deletions rather than the pseudocode's nonnegative
boundary. These differences can affect search results, not just timing. The
September reduced-grid experiment did not reproduce the published accuracy
ranking. Its XGES comparison must not be labeled an exact paper-algorithm replication.

## Code map

| File | Responsibility |
| --- | --- |
| `src-cpp/main.cpp` | CLI options, loading, search dispatch, and outputs. |
| `src-cpp/XGES.{h,cpp}` | Search loops, candidate heaps, incremental updates, LGES phases/restarts, insertion filters, and cached DAG extension. |
| `src-cpp/PDAG.{h,cpp}` | Graph storage, operator validity/application, semi-directed path checks, CPDAG completion, and DAG extension. |
| `src-cpp/LGES.{h,cpp}` | Shared DAG score gate and exhaustive insertion selection used by tests. |
| `src-cpp/Operators.{h,cpp}` | Insert, delete, and reverse candidate records. |
| `src-cpp/EdgeQueueSet.{h,cpp}` | Completion work queue and net edge-change tracking. |
| `src-cpp/BICScorer.{h,cpp}` | Gaussian BIC scores, covariance, and bounded score cache. |
| `src-cpp/DataIO.{h,cpp}` | Validated float64 NumPy input. |
| `tests/` | Regression suites and fixture generation. |
| `experiments/` | Dataset generation, repeated runs, checkpoints, and plots. |

The `xges_core` CMake target exposes the C++ implementation to the CLI and tests.
A search borrows its scorer, which must outlive the search and any copies.
Graph/scorer caches and statistics are mutable and are not thread-safe.

## Attribution

- Achille Nazaret and David Blei, [Extremely Greedy Equivalence Search](https://proceedings.mlr.press/v244/nazaret24a.html), UAI 2024. [Original code](https://github.com/ANazaret/XGES).
- Adiba Ejaz and Elias Bareinboim, [Less Greedy Equivalence Search](https://arxiv.org/abs/2506.22331), NeurIPS 2025. [LGES code](https://github.com/CausalAILab/lges).

Existing source authorship notices are retained. Third-party dependencies retain
their own license notices in `lib/`.
