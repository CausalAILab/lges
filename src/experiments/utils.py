import numpy as np
import networkx as nx
import os
import json
import matplotlib.pyplot as plt
import warnings
import random
from sempler.generators import dag_avg_deg
import sempler
from src.scm.ctm import CTM
import time 

# Set a random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

METRICS_KEYS = [
        "edit_distance", "false_positives", "false_negatives",
        "inserts_eval", "deletes_eval", "turns_eval", "ops_eval",
        "inserts_actual", "deletes_actual", "turns_actual", "ops_actual",
        "cis_eval", "score", "time"
    ]


def get_avg_degree_from_density(num_vars, density):
    """
    Convert density to average degree.
    """
    # Calculate the maximum number of edges
    max_edges = num_vars * (num_vars - 1) / 2

    # Calculate the average degree from density
    avg_degree = density * max_edges / num_vars

    return avg_degree

def get_random_dag(num_vars, avg_degree, random_state=RANDOM_SEED):
    """
    Generate a random directed acyclic graph (DAG) with the specified number of variables and average degree.
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    # Generate a random DAG
    avg_degree = get_avg_degree_from_density(num_vars, avg_degree)
    G = dag_avg_deg(num_vars, avg_degree, 1, 1,random_state=random_state)

    return G

def make_l1_row_normalized_W(G, row_sum_target=0.9):
    W = G * np.random.choice([-1, 1], size=G.shape) * np.random.uniform(0.5, 2.0, size=G.shape)
    np.fill_diagonal(W, 0)

    # L1 normalize each row
    row_sums = np.sum(np.abs(W), axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)  # avoid division by 0
    W = W / row_sums * row_sum_target

    return W

def get_random_data(G, num_samples, model="linear-gaussian", interventions=[{}], random_state=RANDOM_SEED):
    """
    Generate random data based on the given DAG and data type.

    Parameters:
    - G: The adjacency matrix of the DAG.
    - num_samples: Number of samples to generate.
    - data_type: Type of data to generate ("multinomial", "linear-gaussian", "continuous").
    - random_state: Random seed for reproducibility.

    Returns:
    - Generated data as a numpy array.
    """

    if model == "linear-gaussian":
        try:
            # Convert warnings to exceptions to catch them
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                
                # Generate random weights and sample data
                # Weights chosen uniformly in range [-2, -0.5] U [0.5, 2]
                W = G * np.random.choice([-1, 1], size=G.shape) * np.random.uniform(0.5, 2, G.shape)
                # sample noise means from N(0, 1)
                noise_means = np.random.normal(0, 1, size=G.shape[0])
                # sample noise variances from U(0.1, 0.5)
                noise_variances = np.random.uniform(0.1, 0.5, size=G.shape[0])
                sampler = sempler.LGANM(W, noise_means, noise_variances, random_state=random_state)
                # Check if any diagonal elements of W are nonzero
                data = [sampler.sample(n=num_samples if len(i)==0 else int(num_samples//10), do_interventions=i) for i in interventions]
                return data 
        except Exception as e:
            print(f"Error generating data: {e}. Trying with normalized matrix.")
            W = make_l1_row_normalized_W(G)
            noise_means = np.random.normal(0, 1, size=G.shape[0])
            # sample noise variances from U(0.1, 0.5)
            noise_variances = np.random.uniform(0.1, 0.5, size=G.shape[0])
            sampler = sempler.LGANM(W, noise_means, noise_variances, random_state=random_state)
            data = [sampler.sample(n=num_samples if len(i)==0 else int(num_samples//10), do_interventions=i) for i in interventions]
            return data 
    elif model == "multinomial":
        # Generate random multinomial data
        ctm = CTM(G, random_state=random_state)
        data = ctm.sample(num_samples)
        return data
    elif model == "continuous":
       raise NotImplementedError("Continuous data generation is not implemented yet.")
    else:
        raise ValueError(f"Invalid model: {model}. Must be 'multinomial', 'linear-gaussian', or 'continuous'.")

    return data


def cpdag_shd(target, estimate):
    # 1. Skeletons (undirected structure)

    estimate_copy = estimate.copy()

    skeleton_estimate = estimate + estimate.T
    skeleton_target = target + target.T
    skeleton_estimate[skeleton_estimate >= 1] = 1
    skeleton_target[skeleton_target >= 1] = 1
    
    # Difference of skeletons
    diff_skeletons = skeleton_estimate - skeleton_target

    # False positives: extra edges in estimate not in cpA
    superfluous = np.where(diff_skeletons > 0)
    false_positives = len(superfluous[0]) // 2

    estimate_copy[superfluous] = 0  # Remove extra edges from estimate

    # False negatives: missing edges
    missing = np.where(diff_skeletons < 0)
    false_negatives = len(missing[0]) // 2

    estimate_copy[missing] = target[missing]  # Add missing edges to estimate

    # Now skeletons match, compare orientations
    d = np.abs(estimate_copy - target)

    wrong_orientations = np.sum((d + d.T) > 0) // 2


    # Total SHD
    edit_distance = false_positives + false_negatives + wrong_orientations
    
    return edit_distance, false_positives, false_negatives, wrong_orientations

def compute_f1(target, estimate):

    def contains_edge(G, i, j):
        return G[i, j] == 1 # may contain i -> j or i - j, but not j -> i

    t = target.copy()
    e = estimate.copy()

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            if contains_edge(t, i, j) and not contains_edge(e, i, j):
                fp += 1
            elif contains_edge(e, i, j) and not contains_edge(t, i, j):
                fn += 1
            elif contains_edge(t, i, j) and contains_edge(e, i, j):
                tp += 1
            elif not contains_edge(t, i, j) and not contains_edge(e, i, j):
                tn += 1

    
    # Calculate precision, recall, and F1 score
    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
    f1_score = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


class MetricsEvaluator:
    def __init__(self, settings):
        self.settings = settings
        self.metrics_keys = METRICS_KEYS
        self.total_metrics = {key: {setting: [] for setting in self.settings} for key in self.metrics_keys}

    def eval_ground_truth(self, cpA, estimate, setting):
        """
        Evaluate the ground truth by calculating various metrics such as edit distance,
        false positives, and false negatives for the given settings.
        Args:
            cpA (numpy.ndarray): The ground truth cpdag array .
            estimate (numpy.ndarray): The estimated cpdag array to compare against the ground truth.
        Updates:
            self.total_metrics (dict): A dictionary where metrics such as "edit_distance",
                           "false_positives", and "false_negatives" are updated
                           for each setting in `self.settings`.
        Metrics:
            - Edit Distance: The total number of differing elements between `cpA` and `estimate`.
            - False Positives: The count of elements where `cpA` is 0 and `estimate` is 1.
            - False Negatives: The count of elements where `cpA` is 1 and `estimate` is 0.
        """
        
        n = estimate.shape[0]
    
        # 1. Skeletons (undirected structure)
        skeleton_estimate = estimate + estimate.T
        skeleton_cpA = cpA + cpA.T
        skeleton_estimate[skeleton_estimate == 2] = 1
        skeleton_cpA[skeleton_cpA == 2] = 1
        
        # Difference of skeletons
        diff_skeletons = skeleton_estimate - skeleton_cpA

        # False positives: extra edges in estimate not in cpA
        superfluous = np.where(diff_skeletons > 0)
        false_positives = len(superfluous[0]) // 2

        estimate[superfluous] = 0  # Remove extra edges from estimate

        # False negatives: missing edges
        missing = np.where(diff_skeletons < 0)
        false_negatives = len(missing[0]) // 2

        estimate[missing] = cpA[missing]  # Add missing edges to estimate

        # Now skeletons match, compare orientations
        d = np.abs(estimate - cpA)

        wrong_orientations = np.sum((d + d.T) > 0) // 2


        # Total SHD
        edit_distance = false_positives + false_negatives + wrong_orientations
        print(f"False positives: {false_positives}, False negatives: {false_negatives}, Wrong orientations: {wrong_orientations}, Edit distance: {edit_distance}")


        self.total_metrics["edit_distance"][setting].append(edit_distance)
        self.total_metrics["false_positives"][setting].append(false_positives)
        self.total_metrics["false_negatives"][setting].append(false_negatives)

    def append_metrics(self, setting, metrics):
        """
        Append the metrics for a specific setting to the total metrics dictionary.
        Args:
            setting (str): The name of the setting (e.g., "Without CIs", "With CI Oracle").
            metrics (dict): A dictionary containing various metrics such as "inserts_eval",
                            "deletes_eval", "turns_eval", etc.
        Updates:
            self.total_metrics (dict): The dictionary where metrics are appended for the specified setting.
        """
        print()
        for metric, val in metrics.items():
            self.total_metrics[metric][setting].append(val) 

    def summarize(self, display=True):            
        
        summary = {key: {setting:  {"mean": 0, "std": 0} for setting in self.settings} for key in self.metrics_keys}

        for setting in self.settings:
            if display:
                print(f"{setting}:")
            for metric in self.metrics_keys:
                avg_value = np.mean(self.total_metrics[metric][setting])
                std_value = np.std(self.total_metrics[metric][setting])
                summary[metric][setting]["mean"] = avg_value
                summary[metric][setting]["std"] = std_value
                if display:
                    print(f"  {metric.replace('_', ' ').title()}: {avg_value:.2f} ± {std_value:.2f}")

        return summary
    

def extract_params_from_dir(dir):
    """
    Extract experimental parameters from a formatted string.
    Args:
        format_string (str): The formatted string to extract information from.
                             Example: "vars_5_samples_500_degree_0.01_model_linear-gaussian"
    Returns:
        dict: A dictionary containing the extracted key-value pairs.
              Example: {"vars": 5, "samples": 500, "degree": 0.01, "model": "linear-gaussian"}
    """
    parts = dir.split("_")
    params = {}
    for i in range(0, len(parts), 2):
        key = parts[i]
        value = parts[i + 1]
        try:
            value = float(value) if "." in value else int(value)
        except ValueError:
            pass
        params[key] = value
    return params


def plot_obs_no_prior(settings):

    save_dir = "obs/plots"
    os.makedirs(save_dir, exist_ok=True)
    
    for metric in METRICS_KEYS:
        metrics = {setting: [] for setting in settings}
        for dir in os.listdir("obs/results"):
            params = extract_params_from_dir(dir)

            # Skip the iteration if the condition is met
            if not (params["samples"] == 500 and params["degree"] == 0.01):
                continue
            
            results_path = os.path.join("obs", "results", dir, "results.json")
            if not os.path.exists(results_path):
                print(f"File {results_path} does not exist.")
                continue

            with open(results_path, "r") as f:
                results = json.load(f)

            for setting in settings:
                metrics[setting].append((params["vars"], results[metric][setting]))
        
        # Plot the metrics with a different line for each setting
        for setting in settings:
            sorted_points = sorted(metrics[setting], key=lambda x: x[0])
            x_vals = [v for v, _ in sorted_points]
            y_vals = [res["mean"] for _, res in sorted_points]
            y_errs = [res["std"] for _, res in sorted_points]

            plt.errorbar(
                x_vals, y_vals, yerr=y_errs,
                label=setting,
                marker='o', capsize=5
            )

        plt.xlabel("Number of Variables")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} vs Number of Variables")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Save the figure
        save_path = os.path.join(save_dir, f"{metric}.png")
        plt.savefig(save_path)

        plt.close()


class CIOracle:

    def __init__(self, A):
        self.A = nx.DiGraph(A)
        self.dsep_total_time = 0

        
    def __call__(self, X, Y, S=[]):
        """
        Test if X and Y are separated given Z in the DAG represented by
        the adjacency matrix A. Makes the instance callable.
        """
        start = time.time()
        sep = nx.d_separated(self.A, {X}, {Y}, set(S))
        end = time.time()
        self.dsep_total_time += end - start
        return sep

if __name__ == '__main__':
    settings = ["Without CIs", "With CI Oracle", "With CI Test"]
    plot_obs_no_prior(settings)