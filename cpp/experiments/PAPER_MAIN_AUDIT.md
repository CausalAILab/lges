# Observational LGES experiment audit

This run targets the published LGES definitions. The post-run audit found
XGES baseline differences listed below; the complete comparison is not an exact
paper-algorithm replication. The requested reduced
scope is the main observational ER2 experiment, restricted to at most 50
**variables**. The sample count remains 10,000. PC, NOTEARS, prior information,
interventions, and supplementary ER3/large-graph experiments are excluded.

## Parameters

| Parameter | Value |
| --- | --- |
| Variables p | 5, 10, 15, 20, 25, 35, 50 |
| Graphs per p | 50 |
| Samples per graph | 10,000 |
| Graph family | ER2: 2p expected edges |
| sempler average-degree argument | 4, since expected edges = kp/2 |
| Edge weights | Uniform signed magnitudes in [0.5, 2] |
| Noise means | Independent N(0, 1) |
| Noise variances | Independent U(0.1, 0.5) |
| Score | Gaussian BIC, alpha 1: penalty 0.5 log(n) per parent |
| Initial graph | Empty |
| Prior constraints | None |
| Seed | 42 + trial index, for graph, weight/noise RNG and sampling |
| Repetitions | 350 shared datasets × 9 configurations = 3,150 searches |
| Concurrency | Four independent datasets; one BLAS thread per process |
| Per-search safety timeout | 10,000 seconds; failures retained explicitly |

## Algorithm dispatch

I denotes INSERT, D denotes DELETE, R denotes directed reversal, and T denotes
complete TURN selection over both directed and undirected edges. All heuristic
moves maximize score gain over valid eligible operators in the active phase.

| Result label | CLI selection | Operator filter and schedule |
| --- | --- | --- |
| GES | `--baseline ges-t --insert standard` | Exhaust I, then D, then T, once |
| XGES-0 | `--xges0 --insert standard` | Prefer D, R, I; reconsider after each single move |
| XGES | `--insert standard` | XGES-0 plus original forced-deletion extended search |
| LGES-0 (Safe) | `--baseline lges0 --insert safe --strict-lges` | Exhaust I, then T, then D, once |
| LGES-0 (Conservative) | `--baseline lges0 --insert conservative --strict-lges` | Same LGES-0 schedule |
| LGES (Safe) | `--baseline lges --insert safe --strict-lges` | Exhaust D, exhaust T, one I; repeat until no move |
| LGES (Conservative) | `--baseline lges --insert conservative --strict-lges` | Same LGES schedule |
| LGES+ (Safe) | `--baseline lges+ --insert safe --strict-lges` | LGES, followed by pair-blocked forced-deletion trials |
| LGES+ (Conservative) | `--baseline lges+ --insert conservative --strict-lges` | Same LGES+ schedule |

Safe checks that the source is a nondescendant of the target in a chosen
consistent DAG extension and requires strictly positive DAG insertion gain.
Conservative applies that same gate and rejects the pair if any valid CPDAG
insertion for it has negative gain. With no prior knowledge there is one
candidate-priority group. Reversals/TURN are not screened by these insertion
filters. LGES+ blocks both insertion directions of the perturbed pair during a
trial, accepts a strictly positive total gain, and refreshes deletion trials
immediately after acceptance.

XGES retains its persistent candidate heaps. LGES regenerates the active phase's
candidates after every move; temporary heap ordering is only an implementation
of maximum selection. The GES turning phase is optional in the paper's general
GES description. Here it is enabled, matching the released experiment's default
forward–backward–turning call with `iterate=False`. The original forward–backward
`--baseline ges` is not mislabeled as that experimental baseline.

## Resolved discrepancies and remaining limits

- The inherited XGES baseline finishes its old deletion trial list after accepting
  a restart; its paper Algorithm 3 refreshes immediately. Accepted graphs also
  retain prior trial insertion bans, whereas the pseudocode passes the current
  trial's restriction separately. The restart threshold is 1e-7 rather than zero.
  XGES-0 discards zero-gain deletions, unlike the pseudocode's nonnegative test.
  These can alter trajectories; they were not corrected for the recorded run.
  The resulting accuracy ranking did not match the paper, and the effects of
  these differences have not been isolated.

- The released script passes `dag_avg_deg(p, 2, ...)`, yielding p expected edges.
  This run passes 4 to implement the publication's stated **2p** expected edges.
- The released script's LGES labels do not dispatch all published schedules.
  Algorithms 1, 8 and 9 determine the LGES-family schedules here.
- Algorithm 9 constructs a deleted graph E′ but writes E0 in the subsequent
  restart call. This run restarts from E′, consistent with its forced-deletion
  description; otherwise the constructed perturbation would be unused.
- Algorithm 4's `else if` structure can miss a negative veto when it is the first
  encountered candidate. This run follows the prose definition: **any** negative
  valid insertion vetoes the pair.
- The paper does not specify an almost-singular threshold. This run first samples
  with the original weights. If sempler sampling emits a numerical RuntimeWarning
  or raises a linear-algebra error, it scales each nonzero row of that same weight
  matrix to unit L1 norm and retries, preserving noise parameters. It neither
  normalizes all graphs nor redraws a different graph. Each activation is saved.
  This is a documented operational interpretation, not a recovered original threshold.
- Exact original datasets/sample seeds and tie-breaking rules are not published.
  This run uses deterministic fresh datasets and the implementation's consistent
  DAG extension and maximum-selection ordering. It does not claim identical
  estimates or identical numerical values to the published plots.
- The paper used related Python implementations on a shared Xeon cluster. These
  are local C++ implementations with different candidate-maintenance costs.
  Runtime excludes input/covariance setup and final rescoring, includes restart
  work, and is measured while four independent datasets run concurrently. It
  must not be used as a reproduction of the paper's relative Python speed claims.

## Validation and output

The run stores data, generating DAGs, weights/noise parameters, true CPDAGs,
estimated CPDAGs, raw statistics, per-search logs, and SHA-256 identities.
Every search must finish successfully, produce finite statistics and agree with
an independent C++ whole-graph rescore. Strict LGES dispatch is checked from the
binary's reported configuration flag. Failures remain in the results and are
excluded from means with explicit completion/failure counts.

The source regressions cover gate ties and ancestry, LGES phase order and restart
isolation, and exhaustive undirected TURN gains on 16 recorded CPDAG states.
All five normal and sanitizer test suites are required for graph/search changes.
A separate 5/50-variable, 18-search smoke run validated the final nine labels
before the main reduced grid was launched.

Results use CPDAG SHD, counting an absent/extra adjacency or a wrong orientation
once per unordered pair. Precision/recall/F1 use ordered adjacency entries.
Curves show means and sample standard errors (ddof=1), not standard deviations.

## Sources

- [Published paper, Algorithms 1 and 4–9; Section 5.1 and Appendix D.1](https://arxiv.org/abs/2506.22331)
- [Released observational experiment](https://github.com/CausalAILab/lges/blob/main/src/experiments/obs/obs_no_prior.py)
- [sempler graph generator](https://github.com/juangamella/sempler/blob/master/sempler/generators.py)
- [Original XGES implementation](https://github.com/ANazaret/XGES)

The output manifest pins the exact local source, binary, runner and conversion
utility hashes and the installed scientific package versions. Runtime environments
are local prerequisites; recorded dataset hashes are the definitive identity of
this run.
