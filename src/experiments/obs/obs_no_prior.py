# Random graphs of n <= 10 nodes
# Compare quality of output graph
# No background knowledge

import random
import numpy as np
import ges.ges as ges
from src.experiments.utils import METRICS_KEYS, MetricsEvaluator, CIOracle, RANDOM_SEED, get_random_dag, get_random_data
import ges.ges.utils as utils
from causallearn.utils.cit import CIT
import argparse
import os
import json

NUM_GRAPHS = 50  # number of graphs
NUM_VARS = 20
NUM_SAMPLES = 10000


# Set a random seed for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# Experimental parameters
num_vars_list = {"small": [5, 10, 15, 20],
                 "medium": [25, 35, 50],
                 "large": [75, 100, 150],
                 "very_large": [250, 500, 750, 1000]
                }
num_samples_list = {"small": [500,1000],
                    "medium": [10000, 50000],
                    "large": [100000,500000],
                    "very_large": [1000000]
                   }
avg_degree_list = {"sparse": [0.01, 0.05],
                    "medium": [0.1, 0.2],
                    "dense": [0.3, 0.4]
                   }
                    
NUM_GRAPHS = 50

# Convert RuntimeWarning to an exception
# warnings.filterwarnings("error", category=RuntimeWarning)


if __name__ == "__main__":

    # Parse command-line arguments for experimental parameters
    parser = argparse.ArgumentParser(description="Run experiments with specified parameters.")
    parser.add_argument("-p", "--num_vars", type=str, help=f"Size of graphs for the experiment (options: {', '.join(num_vars_list.keys())} or a single integer). Default: medium.")
    parser.add_argument("-n", "--num_samples", type=str, help=f"Size of sample for the experiment (options: {', '.join(num_samples_list.keys())} or a single integer). Default: medium.")
    parser.add_argument("-d", "--avg_degree", type=str, help=f"Graph density for the experiment (options: {', '.join(avg_degree_list.keys())} or a single float). Default: medium.")
    parser.add_argument("-g", "--num_graphs", type=int, default=NUM_GRAPHS, help=f"Number of graphs for the experiment (default: {NUM_GRAPHS}).")
    parser.add_argument("-m", "--model", choices=["multinomial", "linear-gaussian"], default="linear-gaussian", help="Model type for data generation (options: multinomial, linear-gaussian). Default: linear-gaussian.")
    args = parser.parse_args()

    try:
        num_vars = [int(args.num_vars)] if args.num_vars.isdigit() else num_vars_list[args.num_vars]
    except KeyError:
        raise ValueError(f"Invalid value for num_vars: {args.num_vars}. Must be one of {list(num_vars_list.keys())} or a single integer.")

    try:
        num_samples = [int(args.num_samples)] if args.num_samples.isdigit() else num_samples_list[args.num_samples]
    except KeyError:
        raise ValueError(f"Invalid value for num_samples: {args.num_samples}. Must be one of {list(num_samples_list.keys())} or a single integer.")

    try:
        avg_degree = [float(args.avg_degree)] if args.avg_degree.replace('.', '', 1).isdigit() else avg_degree_list[args.avg_degree]
    except KeyError:
        raise ValueError(f"Invalid value for avg_degree: {args.avg_degree}. Must be one of {list(avg_degree_list.keys())} or a single float.")
    
    num_graphs = args.num_graphs
    model = args.model

    settings = ["Without CIs", "With CI Oracle", "With CI Test"]
    ci_test_type = {"linear-gaussian": "fisherz", "multinomial": "chisq"}
    

    # Create a results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    # Change the working directory to the results directory
    os.chdir(results_dir)

    for p in num_vars:
        for n in num_samples:
            for d in avg_degree:
                key = f"vars_{p}_samples_{n}_degree_{d}_model_{model}"

                # Check if results.json already exists for the current experiment
                results_file_path = os.path.join(key, "results.json")
                if os.path.exists(results_file_path):
                    print(f"Results already exist for key: {key}. Skipping experiment...")
                    continue

                # m = MetricsEvaluator(settings)

                # Create a directory for the experiment key
                
                os.makedirs(key, exist_ok=True)

                print("Starting experiment for:", key)

                for i in range(num_graphs):

                    
                    print(f"Running graph {i}/{num_graphs-1} for key: {key}")
                    # Create a directory for this trial
                    trial_dir = os.path.join(key, f"trial_{i}")
                    os.makedirs(trial_dir, exist_ok=True)
                    # Check if data already exists for the current experiment
                    dag_file_path = os.path.join(trial_dir, f"dag_{i}.npy")
                    data_file_path = os.path.join(trial_dir, f"data_{i}.npy")

                    if os.path.exists(dag_file_path):
                        print(f"DAG already exists for key: {key}. Loading DAG...")
                        # Load the DAG from the existing file
                        G = np.load(dag_file_path)
                    else:
                        # Generate a random DAG
                        G = get_random_dag(p, d, random_state=RANDOM_SEED + i)
                        np.save(dag_file_path, G)

                    if os.path.exists(data_file_path):
                        print(f"Data already exists for key: {key}. Loading data...")
                        # Load the data from the existing file
                        data = np.load(data_file_path)
                    else:
                        # Generate random data
                        data = get_random_data(G, n, model, random_state=RANDOM_SEED + i)[0]
                        if len(data) == 0:
                            print(f"Failed to generate data for key: {key}. Skipping experiment...")
                            # Log the error to an error file
                            error_file_path = os.path.join(trial_dir, "error.log")
                            with open(error_file_path, "a") as error_file:
                                error_file.write(f"Failed to generate data for key: {key}, graph: {i}\n")
                            continue
                        np.save(data_file_path, data)

                    
                    # Check if a lock file exists for the current experiment
                    lock_file_path = os.path.join(trial_dir, "lock")
                    if os.path.exists(lock_file_path):
                        print(f"Lock file exists for key: {trial_dir}. Skipping experiment...")
                        continue

                    # Create a lock file to indicate the experiment is in progress
                    with open(lock_file_path, "w") as lock_file:
                        lock_file.write("Experiment in progress")
                    try:
                        # Run GES
                        if model == "linear-gaussian":
                            score_class = ges.scores.GaussObsL0Pen(data)
                        elif model == "multinomial":
                            score_class = ges.scores.DiscreteObsL0Pen(data)
                        ci_oracle = CIOracle(G)
                        ci_test = CIT(data, ci_test_type[model])
                        ci = {"Without CIs": None, "With CI Oracle": ci_oracle, "With CI Test": ci_test}

                        metrics = {setting: dict() for setting in settings}
                        metrics_file_path = os.path.join(trial_dir, f"metrics__{i}.json")

                        for setting in settings:

                            estimate_file_path = os.path.join(trial_dir, f"estimate_{setting.replace(' ', '_').lower()}_{i}.npy")
                            if os.path.exists(estimate_file_path) and os.path.exists(metrics_file_path):
                                print(f"Estimate and metrics already exist for setting: {setting}, graph: {i}. Skipping...")
                                estimate = np.load(estimate_file_path)
                                with open(metrics_file_path, "r") as metrics_file:
                                    metrics[setting] = json.load(metrics_file)[setting]
                            else:
                                # Fit the model with the specified CI setting
                                print(f"Fitting model for setting: {setting}, graph: {i}")
                                estimate, metrics[setting] = ges.fit(score_class, ci=ci[setting])
                                # Save the estimate to a file
                                np.save(estimate_file_path, estimate)

                            # m.eval_ground_truth(utils.dag_to_cpdag(G), estimate, setting)
                            # m.append_metrics(setting, metrics[setting])


                        if not os.path.exists(metrics_file_path):
                            with open(metrics_file_path, "w") as metrics_file:
                                json.dump(metrics, metrics_file, indent=4)


                        
                    except KeyboardInterrupt:
                        # Remove the lock file if a keyboard interrupt occurs
                        if os.path.exists(lock_file_path):
                            os.remove(lock_file_path)
                        print("Experiment interrupted by user. Exiting...")
                        raise
                    finally:
                        # Remove the lock file once the experiment is complete
                        if os.path.exists(lock_file_path):
                            os.remove(lock_file_path)
                    

                    
                
                # results = m.summarize()
                # # Save the summary to a file
                # with open(results_file_path, "w") as results_file:
                #     json.dump(results, results_file, indent=4)
                 
                print("Completed experiment for:", key)


