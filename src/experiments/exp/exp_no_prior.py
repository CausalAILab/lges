# Experiments with random graphs and observational + interventional data
# No background knowledge

import random
import numpy as np
import ges.ges as ges
from src.experiments.utils import RANDOM_SEED, get_random_dag, get_random_data
import ges.ges.utils as utils
import argparse
import os
import json
import gies
import time
import pickle
from sempler.generators import dag_avg_deg

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

# Convert RuntimeWarning to an exception
# warnings.filterwarnings("error", category=RuntimeWarning)


def get_random_interventions(num_nodes, num_targets, max_int_per_target = 3, random_state=RANDOM_SEED):

    rand = random.Random(random_state)
    targets = [[]]
    interventions = [{}]  # include observational data by default?
    for _ in range(num_targets):
        num_int = rand.randint(1, max_int_per_target)
        target = rand.sample(range(num_nodes), num_int)
        target.sort()
        intervention = {t: (rand.choice([-1, 1]), 0.001) for t in target}
        targets.append(target)
        interventions.append(intervention)

    return targets, interventions

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
                    targets_file_path = os.path.join(trial_dir, f"targets_{i}.npy")

                     # Check if DAG already exists for the current experiment
                    if os.path.exists(dag_file_path):
                        print(f"DAG already exists for key: {key}. Loading DAG...")
                        # Load the DAG from the existing file
                        G = np.load(dag_file_path)
                    else:
                        # Generate a random DAG
                        G = dag_avg_deg(p, d, 1, 1, random_state=RANDOM_SEED + i)
                        np.save(dag_file_path, G)

                    # Check if data and targets already exists for the current experiment
                    if os.path.exists(data_file_path) and os.path.exists(targets_file_path):
                        print(f"Data already exists for key: {key}. Loading data...")
                        # Load the data from the existing file
                        with open(data_file_path, 'rb') as f:
                            data = pickle.load(f)
                        with open(targets_file_path, 'rb') as f:
                            targets = pickle.load(f)
                    else:
                        # Delete existing data and targets files if they exist
                        # Generate random data and targets
                        if os.path.exists(data_file_path):
                            os.remove(data_file_path)
                        if os.path.exists(targets_file_path):
                            os.remove(targets_file_path)

                        # Generate random targets and data
                        targets, interventions = get_random_interventions(num_nodes=p, 
                                                                      num_targets=int(0.10*p), # intervene on 10% of nodes
                                                                      max_int_per_target=3, # at most 3 variables in one intervention
                                                                      random_state=RANDOM_SEED + i)
                        if isinstance(targets, np.ndarray):
                            targets = targets.tolist()
                        data = get_random_data(G, 
                                               num_samples=n, 
                                               model=model, 
                                               interventions=interventions, 
                                               random_state=RANDOM_SEED + i)
                        
                        if len(data) == 0:
                            print(f"Failed to generate data for key: {key}. Skipping experiment...")
                            # Log the error to an error file
                            error_file_path = os.path.join(trial_dir, "error.log")
                            with open(error_file_path, "a") as error_file:
                                error_file.write(f"Failed to generate data for key: {key}, graph: {i}\n")
                            continue

                        with open(data_file_path, 'wb') as f:
                            pickle.dump(data, f)
                        with open(targets_file_path, 'wb') as f:
                            pickle.dump(targets, f)

                    
                    # Check if a lock file exists for the current experiment
                    lock_file_path = os.path.join(trial_dir, "lock")
                    if os.path.exists(lock_file_path):
                        print(f"Lock file exists for key: {trial_dir}. Skipping experiment...")
                        continue

                    # Create a lock file to indicate the experiment is in progress
                    with open(lock_file_path, "w") as lock_file:
                        lock_file.write("Experiment in progress")
                    try:
                        if model == "linear-gaussian":
                            score_class = [ges.scores.GaussObsL0Pen(data[i]) for i in range(len(data))]
                        elif model == "multinomial":
                            score_class = [ges.scores.GaussObsL0Pen(data[i]) for i in range(len(data))]
                    
                        algs = ["lges_prune", "lges", "gies"]
                        for alg in algs:

                            metrics_file_path = os.path.join(trial_dir, f"metrics_{alg}_{i}.json")
                            estimate_file_path = os.path.join(trial_dir, f"estimate_{alg}_{i}.npy")
                        
                            if os.path.exists(estimate_file_path) != os.path.exists(metrics_file_path):
                                if os.path.exists(estimate_file_path):
                                    os.remove(estimate_file_path)
                                if os.path.exists(metrics_file_path):
                                    os.remove(metrics_file_path)
            
                            
                            if os.path.exists(estimate_file_path) and os.path.exists(metrics_file_path):
                                print(f"Estimate and metrics already exist for algorithm: {alg}, graph: {i}. Skipping...")
                                continue


                            if alg == "gies":
                                start = time.time()
                                estimate, score = gies.fit_bic(data, targets)
                                end = time.time()
                                metrics = {"time": end - start}
                            else: # lges
                                # fit ges with SafeInsert Heuristic
                                if "prune" in alg:
                                    estimate, metrics = ges.fit_interventional(score_class=score_class,
                                                                            score_based = True, # LGES  
                                                                            prune=True, # ConservativeInsert
                                                                            targets=targets)
                                else:
                                    estimate, metrics = ges.fit_interventional(score_class=score_class,
                                                                            score_based = True, # LGES 
                                                                            prune=False, # SafeInsert
                                                                            targets=targets)

                            
                            # Save the estimate and metrics to a file
                            np.save(estimate_file_path, estimate)
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
                    

                 
                print("Completed experiment for:", key)


