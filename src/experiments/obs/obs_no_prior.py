# Experiments with random graphs and observational data
# No prior asssumptions

import random
import numpy as np
import ges.ges as ges
from src.experiments.utils import METRICS_KEYS, RANDOM_SEED, get_random_dag, get_random_data
import ges.ges.utils as utils
from causallearn.utils.cit import CIT
import argparse
import os
import json
from sempler.generators import dag_avg_deg
from causallearn.search.ConstraintBased.PC import pc
import time
from causalnex.structure.notears import from_numpy
import networkx as nx


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

avg_degree_list = {"sparse": [1.0],
                    "medium": [2.0],
                    "dense": [3.0],
                    "verydense": [4.0]
                   }
                    
NUM_GRAPHS = 50


def run_obs_baselines(data, trial_dir, trial_num):

    baselines = ["pc", "notears"]
    # baselines = ["pc", "notears_threshold"]
    #  baselines = ["pc"]

    for baseline in baselines:
        # Check if the baseline already exists
        estimate_file_path = os.path.join(trial_dir, f"estimate_{baseline}_{trial_num}.npy")
        metrics_file_path = os.path.join(trial_dir, f"metrics_{baseline}_{trial_num}.json")
        
        if os.path.exists(estimate_file_path) != os.path.exists(metrics_file_path):
            if os.path.exists(estimate_file_path):
                os.remove(estimate_file_path)
            if os.path.exists(metrics_file_path):
                os.remove(metrics_file_path)
        
        if os.path.exists(estimate_file_path) and os.path.exists(metrics_file_path):
            print(f"{baseline} estimate and metrics already exists for trial {trial_num}. Skipping...")
            continue

        # Run the baseline method
        if baseline == "pc":
            # Run PC
            print(f"Running PC for {trial_dir}...")
            start = time.time()
            estimate = pc(data, indep_test="fisherz", alpha=ges.ALPHA)
            end = time.time()
            metrics = {"time": end - start}
        elif baseline == "notears":
            print(f"Running Notears for {trial_dir}...")
            start = time.time()
            sm = from_numpy(data)
            sm.threshold_till_dag()
            end = time.time()
            G = nx.to_numpy_array(sm)
            G[G == 0] = 0.0
            G[G != 0] = 1.0
            G = G.astype(int)
            estimate = utils.dag_to_cpdag(G)
            metrics = {"time": end - start}
        elif baseline == "notears_threshold":
            print(f"Running Notears Threshold 0.1 for {trial_dir}...")
            start = time.time()
            sm = from_numpy(data, w_threshold=0.1)
            sm.threshold_till_dag()
            end = time.time()
            G = nx.to_numpy_array(sm)
            G[G == 0] = 0.0
            G[G != 0] = 1.0
            G = G.astype(int)
            estimate = utils.dag_to_cpdag(G)
            metrics = {"time": end - start}

        np.save(estimate_file_path, estimate)
        with open(metrics_file_path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)
 


if __name__ == "__main__":

    # Parse command-line arguments for experimental parameters
    parser = argparse.ArgumentParser(description="Run experiments with specified parameters.")
    parser.add_argument("-p", "--num_vars", type=str, help=f"Size of graphs for the experiment (options: {', '.join(num_vars_list.keys())} or a single integer). Default: medium.")
    parser.add_argument("-n", "--num_samples", type=str, help=f"Size of sample for the experiment (options: {', '.join(num_samples_list.keys())} or a single integer). Default: medium.")
    parser.add_argument("-d", "--avg_degree", type=str, help=f"Graph degree for the experiment (options: {', '.join(avg_degree_list.keys())} or a single float). Default: 2.0.")
    parser.add_argument("-g", "--num_graphs", type=int, default=NUM_GRAPHS, help=f"Number of graphs for the experiment (default: {NUM_GRAPHS}).")
    parser.add_argument("-m", "--model", choices=["multinomial", "linear-gaussian"], default="linear-gaussian", help="Model type for data generation (options: multinomial, linear-gaussian). Default: linear-gaussian.")
    parser.add_argument("-b", "--run_baselines", action="store_true", help="Flag to indicate whether to run baseline methods (default: False).")
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
    run_baselines = args.run_baselines


    if run_baselines:
        settings = ["Without CIs", # Standard GES 
                    "LGES With CI Score", # LGES with SafeInsert
                    "LGES Prune With CI Score", # LGES with ConservativeInsert
        ]
    else:
        settings = ["LGES With CI Score", # LGES with SafeInsert
                    "LGES Prune With CI Score", # LGES with ConservativeInsert
        ]

    

    results_dir = "results_no_prior"
    os.makedirs(results_dir, exist_ok=True)
    os.chdir(results_dir)

    for p in num_vars:
        for n in num_samples:
            for d in avg_degree:
                key = f"vars_{p}_samples_{n}_degree_{d}_model_{model}"

                
                os.makedirs(key, exist_ok=True)

                print("Starting experiment for:", key)

                for i in range(num_graphs):

                    
                    print(f"Running graph {i}/{num_graphs-1} for key: {key}")
                    trial_dir = os.path.join(key, f"trial_{i}")
                    os.makedirs(trial_dir, exist_ok=True)

                    
                    dag_file_path = os.path.join(trial_dir, f"dag_{i}.npy")
                    data_file_path = os.path.join(trial_dir, f"data_{i}.npy")

                    # Check if DAG already exists for the current experiment
                    if os.path.exists(dag_file_path):
                        print(f"DAG already exists for key: {key}. Loading DAG...")
                        G = np.load(dag_file_path)
                    else:
                        # Generate a random DAG
                        G = dag_avg_deg(p, d, 1, 1, random_state=RANDOM_SEED + i)
                        np.save(dag_file_path, G)

                    # Check if data already exists for the current experiment
                    if os.path.exists(data_file_path):
                        print(f"Data already exists for key: {key}. Loading data...")
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

                        metrics_file_path = os.path.join(trial_dir, f"metrics__{i}.json")
                        if os.path.exists(metrics_file_path):
                            print(f"Metrics already exist for key: {key}. Skipping experiment...")
                            continue

                        for setting in settings:
                            metrics_file_path = os.path.join(trial_dir, f"metrics_{setting.replace(' ', '_').lower()}_{i}.json")
                            estimate_file_path = os.path.join(trial_dir, f"estimate_{setting.replace(' ', '_').lower()}_{i}.npy")
                            
                            if os.path.exists(estimate_file_path) != os.path.exists(metrics_file_path):
                                if os.path.exists(estimate_file_path):
                                    os.remove(estimate_file_path)
                                if os.path.exists(metrics_file_path):
                                    os.remove(metrics_file_path)
            
                            
                            if os.path.exists(estimate_file_path) and os.path.exists(metrics_file_path):
                                print(f"Estimate and metrics already exist for setting: {setting}, graph: {i}. Skipping...")
                                continue

                            # Fit the model with the specified CI setting
                            print(f"Fitting model for setting: {setting}, graph: {i}")
                            prune = True if "Prune" in setting else False
                            score_based = True if "Score" in setting else False
                            if prune:
                                print("pruning is enabled in obs_no_prior")
                            estimate, metrics = ges.fit(score_class, 
                                                        prune=prune,
                                                        score_based=score_based,)
                            
                            # Save the estimate and metrics to a file
                            np.save(estimate_file_path, estimate)
                            with open(metrics_file_path, "w") as metrics_file:
                                json.dump(metrics, metrics_file, indent=4)

                        # Run baselines if the flag is set
                        if run_baselines:
                            run_obs_baselines(data, trial_dir, i)
                        
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
                    
                 
                print("Completed experiment for:", key)


