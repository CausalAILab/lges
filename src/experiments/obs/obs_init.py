# Experiments with random graphs and observational data
# Including prior assumptions

import ges.ges as ges
import ges.ges.utils as utils
import numpy as np
import os
import json
import random
from src.experiments.utils import RANDOM_SEED, get_random_data, cpdag_shd
from sempler.generators import dag_avg_deg


np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

NUM_GRAPHS = 50
NUM_NODES = [10, 20, 30, 40, 50]
NUM_SAMPLES = [1000]
FRACTION_CORRECT = [0.0, 0.25, 0.5, 0.75, 1.0]
DEGREE = 2
DIRECTED_PROB = 1 # probability of keeping the edge directed



def construct_init_dag(required, forbidden=None):
    """
    Given a set of required and forbidden edges, return a DAG
    consistent with the set.

    Parameters
    ----------
    required : np.array
        the adjacency matrix of a PDAG, where A[i,j] != 0 => i -> j
        and A[i,j] != 0 & A[j,i] != 0 => i - j.

    forbidden : np.array
        the adjacency matrix of a PDAG, where A[i,j] != 0 => i -> j
        and A[i,j] != 0 & A[j,i] != 0 => i - j.

    Returns
    -------
    np.array
        the adjacency matrix of a DAG, where A[i,j] != 0 => i -> j
    """
    
    G = np.zeros_like(required)

    # Copy directed edges first
    to, fro = np.where((required == 1) & (required.T == 0))
    for i, j in zip(to, fro):
        G[i][j] = 1


    # Then, copy undirected edges and orient per forbidden
    to, fro = np.where((required == 1) & (required.T == 1) & (np.triu(required) == 1))
    for i, j in zip(to, fro):
        if forbidden is not None:
            if forbidden[i][j] == 1 and forbidden[j][i] == 1:
                raise ValueError("The background knowledge is inconsistent. Edge %d - %d is forbidden and required" % (i, j))
            elif forbidden[i][j] == 0 and forbidden[j][i] == 1:
                G[i][j] = 1
                G[j][i] = 0
                continue
            elif forbidden[i][j] == 1 and forbidden[j][i] == 0:
                G[i][j] = 0
                G[j][i] = 1

    # Finally, orient remaining edges in required 
    for i, j in zip(to, fro):
        if G[i][j] == 1 or G[j][i] == 1:
            continue
        ordering = utils.topological_ordering(G)
        if ordering.index(i) < ordering.index(j):
            G[i][j] = 1
            G[j][i] = 0
        else:
            G[i][j] = 0
            G[j][i] = 1
    
    return G
  


def get_init_mec(true_dag, fraction_correct=1, directed_prob=DIRECTED_PROB, num_edges=None):
    """
    Get the initial MEC (Maximal Equivalence Class) from the true DAG.

    Parameters:
    true_dag (np.ndarray): The true DAG represented as an adjacency matrix.
    num_edges (int): The number of edges in background knowledge.
    fraction_correct (float): Fraction of background knowledge edges that are correct.
    directed_prob (float): Probability of keeping the correct edge directed.
    """

    # Select a random subset of edges from the true DAG to create the initial DAG
    required = np.zeros_like(true_dag, dtype=int)  # Initialize the required edges
    num_true = np.sum(true_dag)  # Total number of edges in the true DAG
    if num_edges is None:
        # Set num_edges in background knowledge to half of the number of edges in the true DAG
        num_edges = int(num_true // 2)
        # Set num_edges in background knowledge to 3/4of the number of edges in the true DAG
        # num_edges = int(3 * num_true // 4)

    if num_true == 0:
        print("True DAG is empty. No edges to select.")
        return np.zeros_like(true_dag, dtype=int)
    num_correct = int(np.floor(num_edges * fraction_correct))  # Number of correct edges to keep
    if num_correct > num_true:
        # DAG is too sparse to achieved desired fraction and total number
        num_correct = int(num_true)
        num_edges = int(np.ceil(num_correct // fraction_correct))
    num_incorrect = int(num_edges - num_correct)  # Number of incorrect edges to add

    if num_correct > 0:
        edges = np.transpose(np.nonzero(true_dag))  # Get all edges as (i, j) pairs
        correct_edges = random.sample(list(edges), num_correct)  # Randomly select a subset of edges

        for i, j in correct_edges:
            required[i, j] = 1
            # make some of them undirected
            if random.random() > directed_prob:  # Randomly make some edges in the background knowledge undirected
                required[j, i] = 1

    if num_incorrect > 0:
        non_adjacencies = np.where((true_dag == 0) & (true_dag.T == 0) & ~np.eye(true_dag.shape[0], dtype=bool))
        incorrect_edges = random.sample(list(zip(*non_adjacencies)), num_incorrect)
        for i, j in incorrect_edges:
            # make all incorrect edges undirected for now
            required[i, j] = 1
            required[j, i] = 1


    init_dag = construct_init_dag(required)
    return utils.dag_to_cpdag(init_dag), required

if __name__ == "__main__":
    

    results_dir = "results_init"
    os.makedirs(results_dir, exist_ok=True)
    # Change the working directory to the results directory
    os.chdir(results_dir)

    d = DEGREE
    model = "linear-gaussian"
   
    
    configs = [("Without CIs", "no_init"), # GES without initialization
                ("Without CIs", "init"),    # GES with initialization
                ("LGES with CI Score", "no_init"),
                ("LGES Prune With CI Score", "no_init"),
                ("LGES With CI Score", "init"),    # LGES with safe insert and initialization
               ("LGES With CI Score", "req"),      # LGES with safe insert and priority insertion
               ("LGES Prune With CI Score", "init"),    # LGES with safe insert and initialization
               ("LGES Prune With CI Score", "req")      # LGES with safe insert and priority insertion
               ]
   
    

    # Loop over different configurations
    for p in NUM_NODES:
        for n in NUM_SAMPLES:
            key = f"vars_{p}_samples_{n}_degree_{d}_model_{model}"
            os.makedirs(key, exist_ok=True)
            print("Starting experiment for:", key)
            for i in range(NUM_GRAPHS):
                print(f"Running graph {i}/{NUM_GRAPHS-1} for key: {key}")
                trial_dir = os.path.join(key, f"trial_{i}")
                os.makedirs(trial_dir, exist_ok=True)

                # Check if dag and/or data already exists for the current experiment
                dag_file_path = os.path.join(trial_dir, f"dag_{i}.npy")
                data_file_path = os.path.join(trial_dir, f"data_{i}.npy")

                if os.path.exists(dag_file_path):
                    print(f"DAG already exists for key: {key}. Loading DAG...")
                    # Load the DAG from the existing file
                    true_dag = np.load(dag_file_path)
                else:
                    # Generate a random DAG
                    true_dag = dag_avg_deg(p, DEGREE, 1, 1, random_state=RANDOM_SEED + i)
                    np.save(dag_file_path, true_dag)

                true_cpdag = utils.dag_to_cpdag(true_dag)

                if os.path.exists(data_file_path):
                    print(f"Data already exists for key: {key}. Loading data...")
                    # Load the data from the existing file
                    data = np.load(data_file_path)
                else:
                    # Generate random data
                    data = get_random_data(true_dag, n, model="linear-gaussian", random_state=RANDOM_SEED + i)[0]
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

                    for ci_type, init_type in configs:
                        init_dir = os.path.join(trial_dir, init_type)
                        os.makedirs(init_dir, exist_ok=True)

                        prune = True if "Prune" in ci_type else False
                        score_based = True if "Score" in ci_type else False


                        if init_type == "no_init":

                            # assume that with cis => prune
                            metrics_file_path = os.path.join(init_dir, f"metrics_{ci_type.replace(' ', '_').lower()}_{i}.json")
                            estimate_file_path = os.path.join(init_dir, f"estimate_{ci_type.replace(' ', '_').lower()}_{i}.npy")
                            
                            if os.path.exists(estimate_file_path) != os.path.exists(metrics_file_path):
                                if os.path.exists(estimate_file_path):
                                    os.remove(estimate_file_path)
                                if os.path.exists(metrics_file_path):
                                    os.remove(metrics_file_path)
            
                            
                            if os.path.exists(estimate_file_path) and os.path.exists(metrics_file_path):
                                print(f"Estimate and metrics already exist for init {init_type}, ci setting {ci_type}, graph: {i}. Skipping...")
                                continue

                            # Fit the model with the specified CI setting
                            print(f"Running GES for init {init_type}, ci_type {ci_type}, graph: {i}")
                            estimate, metrics = ges.fit(score_class, 
                                                        prune=prune,
                                                        score_based=score_based,)

                            # Save the estimate and metrics to a file
                            np.save(estimate_file_path, estimate)
                            with open(metrics_file_path, "w") as metrics_file:
                                json.dump(metrics, metrics_file, indent=4)

                        else:
                            for fc in FRACTION_CORRECT:

                                metrics_file_path = os.path.join(init_dir, f"metrics_{ci_type.replace(' ', '_').lower()}_fc_{fc}_{i}.json")
                                estimate_file_path = os.path.join(init_dir, f"estimate_{ci_type.replace(' ', '_').lower()}_fc_{fc}_{i}.npy")
                                
                                if os.path.exists(estimate_file_path) != os.path.exists(metrics_file_path):
                                    if os.path.exists(estimate_file_path):
                                        os.remove(estimate_file_path)
                                    if os.path.exists(metrics_file_path):
                                        os.remove(metrics_file_path)
                
                                
                                if os.path.exists(estimate_file_path) and os.path.exists(metrics_file_path):
                                    print(f"Estimate and metrics already exist for init {init_type}, ci setting {ci_type}, graph: {i}. Skipping...")
                                    continue

                                # num
                                A0, required = get_init_mec(true_dag, fraction_correct=fc)
                                print(f"Running GES for init {init_type}, ci_type {ci_type}, graph: {i} with initialization fc {fc}")

                                # Fit the model with the specified CI setting
                                estimate, metrics = ges.fit(score_class, 
                                                            prune=prune,
                                                            score_based=score_based,
                                                            A0=A0 if init_type == "init" else None,
                                                            required=required if init_type == "req" else None,
                                                            )

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

            

