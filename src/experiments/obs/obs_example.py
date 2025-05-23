"""
Quick-start demo: GES vs. LGES (SafeInsert) vs. LGES (ConservativeInsert) on a 10-node DAG
==================================================================

• Generates a random DAG with average degree 2
• Simulates 1000 linear-Gaussian samples
• Fits:
    1.   GES                  (score_based = False, prune = False)
    2.   LGES-SafeInsert      (score_based = True , prune = False)
    3.   LGES-Conservative    (score_based = True , prune = True)
• Prints Structural Hamming Distance (SHD) and runtime for each estimate.
"""

import numpy as np
from sempler.generators import dag_avg_deg
import ges.ges as ges
import ges.ges.utils as utils
from pprint import pprint
from src.experiments.utils import cpdag_shd, get_random_data, RANDOM_SEED

np.random.seed(RANDOM_SEED)

# Generate a random 10-node DAG with average degree 2
p = 10
avg_deg = 2.0
G_true = dag_avg_deg(p, avg_deg, 1, 1, random_state=RANDOM_SEED)
cpdag_true = utils.dag_to_cpdag(G_true)

# Simulate linear-Gaussian data (1000 samples)
n_samples = 1000
from src.experiments.utils import get_random_data   # <- your helper
data  = get_random_data(G_true, n_samples, model="linear-gaussian",
                            random_state=RANDOM_SEED)[0]

# Initialize score (Gaussian BIC)
score = ges.scores.GaussObsL0Pen(data)

estimates = {}
runtimes = {}

estimates["GES"], metrics = ges.fit(score,
                              score_based=False,   # use CI tests
                              prune=False)
runtimes["GES"] = metrics["time"]

estimates["LGES-Safe"], metrics = ges.fit(score,
                                    score_based=True,   # score-based insertions
                                    prune=False)        # SafeInsert by default
runtimes["LGES-Safe"] = metrics["time"]

estimates["LGES-Cons"], metrics = ges.fit(score,
                                    score_based=True,
                                    prune=True)         # ConservativeInsert
runtimes["LGES-Cons"] = metrics["time"]



results = {}
for name, est in estimates.items():
    shd = cpdag_shd(cpdag_true, est)[0]
    results[name] = int(shd)

print("\nSHD w.r.t. ground-truth CPDAG:")
for name, shd in results.items():
    print(f"{name}: {shd}")

print("\nRuntimes (in seconds):")
for name, time in runtimes.items():
    print(f"{name}: {time:.3f} sec")
