# Repository maintenance guide

This file records the implementation's contracts and scope for future changes.

## Scope and provenance

The C++ search is based on the original XGES code and adds LGES-inspired SafeInsert and
ConservativeInsert strategies. See README.md for upstream attribution and the supported scope.
Sibling repositories are reference sources, not runtime dependencies. Fixture regeneration and experiments
use the Gaussian score and graph utilities from an enclosing or sibling LGES checkout through explicit file imports, avoiding the
reference package's unrelated torch dependency.

This repository has substantial correctness, input-validation, testing and maintenance changes beyond
upstream. Do not assume `src-cpp` is byte-identical to XGES. The history records the initial import,
build repairs, strategy integration, simplified interfaces, and the audit fixes.

LGES-0, LGES and LGES+ phased search drivers are available through `fit_lges0`,
`fit_lges`, and `--baseline lges0/lges/lges+`. `--strict-lges` enables the
observational strict gate, complete TURN selection, and positive restart test.
`--baseline ges-t` runs forward–backward–full-TURN once. GET PRIORITY INSERTS, public initial-CPDAG input,
interventional scoring and I-ORIENT remain future features. Prior-knowledge
insertion groups are soft preferences, not hard bans on graph operations.

## Build and validation

Use C++23, CMake >= 3.26, Eigen, Boost headers, BLAS and the pinned spdlog archive fetched by CMake. The NumPy reader is
first-party `DataIO`; cnpy and its zlib/build dependency have been removed. `EIGEN_USE_BLAS` is a public
compile definition on `xges_core`, keeping all translation units consistent. `lib/cxxopts.hpp` is vendored.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
python3 -m pip install -r requirements-dev.txt
ctest --test-dir build --output-on-failure
```

`BUILD_TESTING=OFF` removes the Python test dependency. Otherwise CTest includes three C++ suites, a
standard-library Python CLI suite and a NumPy experiment suite. NumPy's absence produces an explicit
skip for the latter. Require all five when changing experiment plumbing.

For graph/search/numerical changes, also use a separate Debug build with
`-DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer'` and
`-DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'`. CMake caches absolute paths: after relocating a
checkout, configure a fresh build directory. Build outside synchronized folders when practical.

The September 2026 fix pass checked all five suites in Debug with sanitizers and in Release, and separately
validated 144 searches against the Python graph routines plus all 1024 topologically ordered five-node DAG
completions. The random runs checked score consistency, admissible insertion exhaustion, cached DAG state
and extended-trial counters. These checks are evidence for the exercised cases, not a general proof of
search completeness or statistical recovery.

Keep source comments focused on behavior, invariants and implementation choices. Put paper comparisons
and upstream attribution in README.md. Use the repository's `.clang-format` for C++. Keep first-party code clean under `-Wall -Wextra -Wpedantic`
and annotate intentional switch fallthroughs. Types logged via `operator<<` need an explicit
`fmt::ostream_formatter` specialization with the pinned spdlog/fmt version.

## Contracts that must survive refactoring

- **Scoring:** `ScorerInterface` requires decomposability, score equivalence, finite values and consistent
  variable indexing. `BICScorer` owns its covariance, requires finite data and positive finite alpha, and
  rejects constant columns or unsupported singular/numerically degenerate regressions. There is no implicit
  regularization. The default alpha is 2 in the CLI and 1 in experiments/reference fixtures.
- **Cache policy:** the BIC cache retains at most 100000 parent-set entries by default, configurable down to
  zero. At capacity it drops the previous generation. This is an entry budget, not a byte guarantee.
  Eviction must leave scores unchanged. Cache/statistics mutation is not thread-safe.
- **Ownership:** `XGES` borrows a scorer and holds a shared logger. Copies share these dependencies but own
  their graph state. An omitted logger uses spdlog's default or a silent fallback; callers need not register
  a logger named `stdout_logger`.
- **Graph storage:** preserve `FlatSet` (`flat_set<int>` over `small_vector<int,10>`). Parents and neighbors
  are often small, and contiguous sorted storage is intentional. Every edge mutation must maintain all five
  adjacency indexes and both edge counts. Invalid mutations throw before changing the graph.
- **Graph algorithms:** BFS enqueues each node once into a reusable vector. Completion checks v-structures
  and Meek rules 1–3; a stale worklist edge must be skipped. CPDAG validation also requires a consistent
  extension, catching directed cycles and nonchordal components that local checks miss.
- **Change tracking:** `EdgeModificationsMap` canonicalizes endpoints and stores net old/new edge types.
  It removes no-op transitions. `EdgeChange` names the seven cases in the update table.
- **Path witnesses:** `UnblockedPathsMap` is persistent dependency information used by incremental search,
  not disposable BFS scratch. Explicit `IntPairHash` parameters avoid invalid `std::hash<pair<int,int>>`
  specializations.
- **CLI boundary:** `DataIO` parses the supported float64 NPY schema without executing header content.
  Check shape, dtype, byte order and payload before mapping. Graph CSV has a unique node-ID header,
  exact square binary rows, a zero diagonal and a consistent extension. Output errors must fail the command.
  `--graph_truth` completes its DAG/PDAG and reports exact CPDAG SHD.

## Search correctness and strategy semantics

XGES-0 prioritizes deletes, then reversals, then inserts. OPS chooses the largest gain across its enabled
operator types. GES runs a forward and a backward phase; GES-r also runs reversals and repeats phases.
LGES exhausts deletions, then reversals, then applies one insertion per cycle. It regenerates
candidates for each selection and runs another cycle if any move changed the graph, including
when a reversal enabled a deletion but no insertion was found. All heuristic moves have positive gains.
XGES extended search forbids ordered reinsertion. LGES+ uses its own helper, blocks both directions
of a pair only during its trial, and refreshes all deletion trials immediately after acceptance.
Compatibility LGES+ and XGES accept net gains exceeding `1e-7`; strict LGES+
accepts any strictly positive net gain.

Candidate heaps can contain stale operators. Stored effective parents and graph validity must be checked
at pop time. Avoid score-based duplicate heuristics that discard an operator without checking its current
state: a regenerated operator with a similar score may be valid again.

The Standard strategy retains XGES's local candidate update table. **Safe and Conservative rebuild the
entire insertion heap after every graph change**, while deletes and reversals remain incremental. The
previous assumption that local updates were adequate was disproved by saved counterexamples:

- `safe_refresh.npy`: a changed DAG extension makes a previously rejected pair admissible outside the local
  update neighborhood.
- `conservative_refresh.npy`: deleting/orienting edges removes a negative valid operator, lifting the pair
  veto while a positive operator's own validity remains unchanged.

Pop-time rechecks cannot recover missing candidates. Do not restore local-only strategy updates based
solely on matching outputs on a small benchmark. A replacement needs complete dependencies for both the
chosen DAG's parents and the negative-operator veto, plus end-to-end exhaustion tests.

`current_dag` caches a consistent extension by graph version. Every applied operator invalidates that
version, and accepting an extended-search copy must invalidate the **parent's** cache as well. A copied
search starts with an empty cache. Counters from each trial are merged exactly once, including rejected
and unchanged trials; graph statistics are preserved when a trial's graph replaces the parent.

The implemented strategy semantics are:

- Compatibility SafeInsert keeps exact gate ties; strict LGES rejects them.
- ConservativeInsert discards only on a strictly negative valid gain.
- The chosen DAG member matters. Sink-elimination orders can differ across implementations.
- The compatibility DAG gate does not test ancestry; strict LGES rejects a source
  that is a descendant of the target in the chosen DAG.
- Reversals are not filtered by either insertion strategy. Strict LGES and ges-t
  also enumerate undirected TURN moves and compare their gains with directed reversals.

`LGES.cpp` retains the exhaustive reference enumeration. Production uses the same gate plus XGES's clique
subset enumeration and lazy path checks for negative veto candidates. `lges_operators` compares these
paths on recorded states. `XGESTestAccess` is a narrow test-only adapter for candidate/cache regressions;
do not expose mutable graph state through the public search API to facilitate tests.

## Tests, counters and experiments

Fixture generation writes data and expectations together. Its provenance records the reference commit,
source hashes (including local changes), seed, Python/NumPy versions and data hashes. It supports old and
current unpruned reference return types and converts NumPy scalars to ordinary numeric text. The fixture
comparator rejects nonfinite scores and checks maximum scores even when the best operator is tied.

`XGES::statistics`, `PDAG::statistics` and scorer statistics are the built-in profiler. CLI search counters
exclude covariance setup, initial empty-graph scoring and the final `score check`. Extended-trial work is
included. API counters accumulate until explicitly cleared. Timings use a monotonic clock. Interpret cache
peak entries as a gauge, not as an additive count.

Experiment scripts use one seed sequence per `(seed, p, graph)` and share each dataset across all nine
configurations. They validate graph/stat outputs, enforce a configurable subprocess timeout, clean up
owned temporary directories, and checkpoint each completed row. Resume requires identical settings,
binary/source/reference hashes and environment; it truncates only an uncommitted partial-write suffix.
Completed CSVs have checksums in their manifests. Plot-only/report-only reads saved metadata, and flags
legacy data whose settings cannot be reconstructed. Error bands use sample standard errors (`ddof=1`).

The weight generator's unconditional row-l1 normalization and topological labels are recorded experimental
choices. `compare_to_paper.py` uses the larger sample size but does not claim to reproduce the full LGES
algorithm or conditional normalization procedure. F1 counts ordered edges; empty/empty is defined as 1,
and one empty set against a nonempty set as 0.

Generated files in `experiments/results/` are ignored. Obsolete historical benchmark outputs
were removed from source control. Re-run into a new directory before drawing conclusions.

The separate `paper_main.py` runner uses the published ER2 expected-edge count,
conditional normalization and explicit strict schedules. Its source/binary and
reference-utility hashes are frozen before a run. Parameters and unresolved
publication ambiguities are recorded in `experiments/PAPER_MAIN_AUDIT.md`.
Keep the older strategy experiment's normalization and configuration descriptions
separate from those of this runner.

## Packaging and known limits

Keep the project portable as both a standalone checkout and `lges/cpp`.
`python_support.py` resolves enclosing/sibling reference repositories; every helper
accepts `--lges-repo`. Do not hardcode developer paths or temporary directories.
The executable target remains `xges` even under a parent CMake project. spdlog is
fetched from a checksum-pinned archive, so no Git submodule state is needed.

The inherited XGES restart implementation refreshes deletions only after exhausting
the previous list, persists accepted insertion bans, and uses a 1e-7 threshold.
These are differences from its paper pseudocode, not merely performance choices.
Do not claim exact paper equivalence for that baseline or explain the observed
accuracy mismatch entirely as a runtime implementation issue.
