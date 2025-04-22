# Random graphs of n <= 10 nodes
# Compare quality of output graph
# No background knowledge

import random
import sempler
from sempler.generators import dag_avg_deg
import numpy as np
import ges.ges as ges
from src.experiments.utils import METRICS_KEYS, MetricsEvaluator, CIOracle
import ges.ges.utils as utils
from causallearn.utils.cit import CIT
import argparse


RANDOM_SEED = 42
NUM_GRAPHS = 50  # number of graphs
NUM_VARS = 20
NUM_SAMPLES = 10000


# Set a random seed for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

if __name__ == "__main__":
    num_vars = 15  # number of variables
    num_samples = 1000  # size of the observational sample
    num_graphs = 20  # number of graphs

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run GES experiments with different parameters.")
    parser.add_argument("-p", "--num_vars", type=int, default=NUM_VARS, help=f"Number of variables (default: {NUM_VARS})")
    parser.add_argument("-n", "--num_samples", type=int, default=NUM_SAMPLES, help=f"Number of samples (default: {NUM_SAMPLES})")
    parser.add_argument("-g", "--num_graphs", type=int, default=NUM_GRAPHS, help=f"Number of graphs (default: {NUM_GRAPHS})")
    args = parser.parse_args()

    # Update variables with CLI arguments
    num_vars = args.num_vars
    num_samples = args.num_samples
    num_graphs = args.num_graphs

    print("Running GES experiments with the following parameters:")
    print(f"Number of variables: {num_vars}")
    print(f"Number of samples: {num_samples}")
    print(f"Number of graphs: {num_graphs}")

    settings = ["Without CIs", "With CI Oracle", "With CI Fisher"]
    m = MetricsEvaluator(settings)

    for i in range(num_graphs):
        print("Checking SCM %d" % (i))
        # Generate DAG and data
        A = dag_avg_deg(num_vars, 3, 1, 1, random_state=RANDOM_SEED + i)
        cpA = utils.dag_to_cpdag(A)
        W = A * np.random.uniform(1, 2, A.shape)
        data = sempler.LGANM(W, (1, 10), (0.5, 1), random_state=RANDOM_SEED+i).sample(n=num_samples)
        ci_oracle = CIOracle(A)
        ci_fisher = CIT(data, method="fisherz")

        ci = {"Without CIs": None, "With CI Oracle": ci_oracle, "With CI Fisher": ci_fisher}

        score_class = ges.scores.GaussObsL0Pen(data)

        for setting in settings:
            estimate, metrics = ges.fit(score_class, ci=ci[setting])
            m.eval_ground_truth(cpA, estimate)
            m.append_metrics(setting, metrics)
    
    m.summarize()

