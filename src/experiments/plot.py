import matplotlib.pyplot as plt
from src.experiments.utils import METRICS_KEYS
import os
import json
from collections import defaultdict
import re
import numpy as np
from src.experiments.utils import cpdag_shd
import ges.ges.utils as utils
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
import ges.ges as ges
from causallearn.graph.GraphClass import CausalGraph



pretty_metric_names = {
    "SHD": "SHD",
    "ops_eval": "Number of scoring operations",
    "time": "Time (s)",
    "BIC": "BIC",
    "BIC difference from true DAG": "BIC Difference",
    "False positives": "Excess Adjacencies",
    "False negatives": "Missing Adjacencies",
    "Wrong orientations": "Incorrect Orientations",
}

method_names = {
    "Without CIs": "GES",
    "With CI Score": "LGES with SafeInsert",
    "Prune With CI Score": "LGES with ConservativeInsert",
    "XGES": "XGES",
    "PC": "PC",
}



def process_pc_estimate(estimate):

    # PC alg returns cg : a CausalGraph object, where 
    # cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicate i –> j; 
    # cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicate i — j; 
    # cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

   
    G = estimate.G.graph

    cpdag = np.zeros_like(np.asarray(G))

    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            if G[i, j] == -1 and G[j, i] == -1:
                cpdag[i, j] = 1
                cpdag[j, i] = 1
            elif G[j, i] == 1 and G[i, j] == -1:
                cpdag[i, j] = 1

    return cpdag




def plot_obs_no_prior():
    base_dir = "./obs/results_no_prior"

    # Structure: metrics_data[metric][ci_type][num_nodes] = list of values
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ci_types = ["Without CIs", 
                # "With CI Test", 
                # "Prune With CI Test",
                "With CI Score",
                "Prune With CI Score",
                # "XGES",
                "PC",
                ]

    for folder in os.listdir(base_dir):
        match = re.match(r"vars_(\d+)_samples_(10000)_degree_2.0_model_linear-gaussian", folder)
        if not match:
            continue
        num_nodes = int(match.group(1))
        folder_path = os.path.join(base_dir, folder)

        for trial in os.listdir(folder_path):
            trial_match = re.match(r"trial_(\d+)", trial)
            if not trial_match:
                continue
            trial_number = trial_match.group(1)
            trial_path = os.path.join(folder_path, trial)
            if not os.path.isdir(trial_path):
                continue

            dag_file = os.path.join(trial_path, f"dag_{trial_number}.npy")
            if not os.path.exists(dag_file):
                continue

            data_file = os.path.join(trial_path, f"data_{trial_number}.npy")
            if not os.path.exists(data_file):
                continue

            dag = np.load(dag_file)
            data = np.load(data_file)
            score_class = ges.scores.GaussObsL0Pen(data)
            true_dag_score = score_class.full_score(dag)

            for ci_type in ci_types:
                metrics_file_path = os.path.join(trial_path, f"metrics_{ci_type.replace(' ', '_').lower()}_{trial_number}.json")
                estimate_file_path = os.path.join(trial_path, f"estimate_{ci_type.replace(' ', '_').lower()}_{trial_number}.npy")

                if not os.path.exists(metrics_file_path) or not os.path.exists(estimate_file_path):
                    continue


                estimate = np.load(estimate_file_path, allow_pickle=True)
                if ci_type == "PC":
                    estimate = estimate.item()
                    estimate = process_pc_estimate(estimate)
                            
                shd, false_positives, false_negatives, wrong_orientations = cpdag_shd(utils.dag_to_cpdag(dag), estimate)
                metrics_data["SHD"][ci_type][num_nodes].append(shd)
                metrics_data["False positives"][ci_type][num_nodes].append(false_positives)
                metrics_data["False negatives"][ci_type][num_nodes].append(false_negatives)
                metrics_data["Wrong orientations"][ci_type][num_nodes].append(wrong_orientations)
                try:
                    estimate_dag = utils.pdag_to_dag(estimate)
                    metrics_data["BIC"][ci_type][num_nodes].append(score_class.full_score(estimate_dag))
                    metrics_data["BIC difference from true DAG"][ci_type][num_nodes].append(true_dag_score - score_class.full_score(estimate_dag))
                except Exception as e:
                    print(f"Error converting estimate to DAG: {e} for CI type {ci_type} and trial {trial_number}")
                    continue
               

                with open(metrics_file_path, "r") as f:
                    metrics = json.load(f)
                    for metric, value in metrics.items():
                        metrics_data[metric][ci_type][num_nodes].append(value)

    # Plotting
    output_dir = "./obs/plots_no_prior"
    os.makedirs(output_dir, exist_ok=True)

    # Style maps
    styles = {
        "Without CIs": {
            "linestyle": ":", "color": "tab:blue", "marker": "s", "label": "Without CIs"
        },
        "With CI Score": {
            "linestyle": "-", "color": "tab:orange", "marker": "o", "label": "With CI Score"
        },
        "Prune With CI Score": {
            "linestyle": "--", "color": "mediumpurple", "marker": "^", "label": "Prune (CI Score)"
        },
        "XGES": {
            "linestyle": "-", "color": "dimgray", "marker": "v", "label": "XGES"
        },
        "PC": {
            "linestyle": "-.", "color": "tab:green", "marker": "D", "label": "PC"
        },
    }

    rcParams.update({'font.size': 14})

    for metric, ci_data in metrics_data.items():
        plt.figure(figsize=(10, 6))

        for ci_type, node_dict in sorted(ci_data.items()):
            x_vals = sorted(node_dict.keys())
            means = [np.mean(node_dict[n]) for n in x_vals]
            stds = [np.std(node_dict[n]) for n in x_vals]
            s = styles[ci_type]

            plt.errorbar(
                x_vals, means, yerr=stds,
                label=method_names[ci_type],
                linestyle=s["linestyle"],
                color=s["color"],
                marker=s["marker"],
                capsize=4,
                linewidth=2
            )

        if metric == "time" or metric == "ops_eval":
            plt.yscale("log")
            
        desired_order = ["PC", "Without CIs",  "XGES", "With CI Score", "Prune With CI Score"]
        desired_order = [method_names[ci] for ci in desired_order if ci in method_names]

        # Reorder
        handles, labels = plt.gca().get_legend_handles_labels()
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]))
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        plt.errorbar(
            x_vals,
            means,
            yerr=stds,
            label=ci_type,
            linestyle=s["linestyle"],
            color=s["color"],
            marker=s["marker"],
            linewidth=2,
            elinewidth=1,        # thinner error bars
            ecolor=s["color"],   # match line color
            alpha=0.5,           # slightly transparent error bars
            capsize=2,           # small caps (optional)
            zorder=1             # draw error bars below lines
        )
        plt.xlabel("Number of Nodes")
        plt.ylabel(pretty_metric_names[metric] if metric in pretty_metric_names.keys() else metric)
        # plt.yscale("log")
        #  plt.title(f"{metric.capitalize()} vs Number of Nodes")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.legend(
            sorted_handles,
            sorted_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=12,
            title="Method"
        )       
        plt.savefig(os.path.join(output_dir, f"{metric}.png"), dpi=300, bbox_inches="tight")
        plt.close()


def plot_obs_init_fc():
    base_dir = "./obs/results_init"

    # Structure: metrics_data[metric][(init_type, ci_type, fc_value)][num_nodes] = list of values
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    fc_values_found = set()

    for folder in os.listdir(base_dir):
        match = re.match(r"vars_(\d+)_samples_1000_degree_3_model_linear-gaussian", folder)
        if not match:
            continue
        num_nodes = int(match.group(1))
        folder_path = os.path.join(base_dir, folder)

        for trial in os.listdir(folder_path):
            trial_match = re.match(r"trial_(\d+)", trial)
            if not trial_match:
                continue
            trial_number = trial_match.group(1)
            trial_path = os.path.join(folder_path, trial)
            if not os.path.isdir(trial_path):
                continue

            for init_type in ["init", "no_init", "req"]:
                init_path = os.path.join(trial_path, init_type)
                if not os.path.exists(init_path):
                    continue

                if init_type == "init" or init_type == "req":
                    # Handle multiple fc values
                    for file in os.listdir(init_path):
                        fc_match = re.match(r"metrics_with_ci_test_fc_([0-9.]+)_([0-9]+)\.json", file)
                        if not fc_match:
                            continue
                        fc_str = fc_match.group(1)  # e.g., "0", "0.25", "1"
                        fc_value = float(fc_str)
                        fc_values_found.add(fc_value)

                        metrics_filepath = os.path.join(init_path, file)
                        estimate_file = os.path.join(init_path, f"estimate_with_ci_test_fc_{fc_value}_{trial_number}.npy")
                        dag_file = os.path.join(trial_path, f"dag_{trial_number}.npy")

                        if not os.path.exists(dag_file) or not os.path.exists(estimate_file):
                            continue

                        dag = np.load(dag_file)
                        estimate = np.load(estimate_file)
                        shd = cpdag_shd(utils.dag_to_cpdag(dag), estimate)
                        metrics_data["SHD"][(f"{init_type}", "with_ci", fc_value)][num_nodes].append(shd)

                        with open(metrics_filepath, "r") as f:
                            metrics = json.load(f)
                            for metric, value in metrics.items():
                                metrics_data[metric][(f"{init_type}", "with_ci", fc_value)][num_nodes].append(value)

                else:
                    # Only one version for no_init
                    for ci_type, ci_setting in [("with_ci", "_with_ci_test_"), ("without_ci", "_without_cis_")]:
                        metrics_filename = f"metrics{ci_setting}{trial_number}.json"
                        metrics_filepath = os.path.join(init_path, metrics_filename)
                        estimate_file = os.path.join(init_path, f"estimate{ci_setting}{trial_number}.npy")
                        dag_file = os.path.join(trial_path, f"dag_{trial_number}.npy")

                        if not os.path.exists(metrics_filepath) or not os.path.exists(dag_file) or not os.path.exists(estimate_file):
                            continue

                        dag = np.load(dag_file)
                        estimate = np.load(estimate_file)
                        shd = cpdag_shd(utils.dag_to_cpdag(dag), estimate)
                        metrics_data["SHD"][("no_init", ci_type, None)][num_nodes].append(shd)

                        with open(metrics_filepath, "r") as f:
                            metrics = json.load(f)
                            for metric, value in metrics.items():
                                metrics_data[metric][("no_init", ci_type, None)][num_nodes].append(value)

    # Plotting
    output_dir = "./obs/plots_init"
    os.makedirs(output_dir, exist_ok=True)

    fc_colors = {
        0.0: 'tab:blue',
        0.25: 'tab:orange',
        0.5: 'tab:green',
        0.75: 'tab:red',
        1.0: 'tab:purple'
    }


    for metric, config_data in metrics_data.items():
        plt.figure()


        for (init_type, ci_type, fc_value), node_dict in sorted(config_data.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or -1)):
            x_vals = sorted(node_dict.keys())
            means = [np.mean(node_dict[n]) for n in x_vals]
            stds = [np.std(node_dict[n]) for n in x_vals]

            # Determine line style and color
            if init_type in ["init", "req"]:
                linestyle = '-' if init_type == "init" else '--'
                color = fc_colors.get(fc_value, 'black')  # fallback in case
                label = f"{init_type}, fc={fc_value}"
            else:
                linestyle = '-'  # solid for no_init
                color = 'gray' if ci_type == "without_ci" else 'black'
                label = f"{init_type}, {ci_type}"

            plt.errorbar(x_vals, means, yerr=stds, label=label, marker='o', capsize=5, linestyle=linestyle)

        plt.xlabel("Number of Nodes")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Number of Nodes")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}.png"))
        plt.close()


def plot_obs_init_fc_seaborn():
    base_dir = "./obs/results_init"

    # Structure: metrics_data[metric][(init_type, ci_type, fc_value)][num_nodes] = list of values
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    fc_values_found = set()

    for folder in os.listdir(base_dir):
        match = re.match(r"vars_(\d+)_samples_1000_degree_3_model_linear-gaussian", folder)
        if not match:
            continue
        num_nodes = int(match.group(1))
        folder_path = os.path.join(base_dir, folder)

        for trial in os.listdir(folder_path):
            trial_match = re.match(r"trial_(\d+)", trial)
            if not trial_match:
                continue
            trial_number = trial_match.group(1)
            trial_path = os.path.join(folder_path, trial)
            if not os.path.isdir(trial_path):
                continue

            for init_type in ["init", "no_init", "req"]:
                init_path = os.path.join(trial_path, init_type)
                if not os.path.exists(init_path):
                    continue

                if init_type == "init" or init_type == "req":
                    # Handle multiple fc values
                    for file in os.listdir(init_path):
                        fc_match = re.match(r"metrics_with_ci_test_fc_([0-9.]+)_([0-9]+)\.json", file)
                        if not fc_match:
                            continue
                        fc_str = fc_match.group(1)  # e.g., "0", "0.25", "1"
                        fc_value = float(fc_str)
                        fc_values_found.add(fc_value)

                        metrics_filepath = os.path.join(init_path, file)
                        estimate_file = os.path.join(init_path, f"estimate_with_ci_test_fc_{fc_value}_{trial_number}.npy")
                        dag_file = os.path.join(trial_path, f"dag_{trial_number}.npy")

                        if not os.path.exists(dag_file) or not os.path.exists(estimate_file):
                            continue

                        dag = np.load(dag_file)
                        estimate = np.load(estimate_file)
                        shd = cpdag_shd(utils.dag_to_cpdag(dag), estimate)
                        metrics_data["SHD"][(f"{init_type}", "with_ci", fc_value)][num_nodes].append(shd)

                        with open(metrics_filepath, "r") as f:
                            metrics = json.load(f)
                            for metric, value in metrics.items():
                                metrics_data[metric][(f"{init_type}", "with_ci", fc_value)][num_nodes].append(value)

                else:
                    # Only one version for no_init
                    for ci_type, ci_setting in [("with_ci", "_with_ci_test_"), ("without_ci", "_without_cis_")]:
                        metrics_filename = f"metrics{ci_setting}{trial_number}.json"
                        metrics_filepath = os.path.join(init_path, metrics_filename)
                        estimate_file = os.path.join(init_path, f"estimate{ci_setting}{trial_number}.npy")
                        dag_file = os.path.join(trial_path, f"dag_{trial_number}.npy")

                        if not os.path.exists(metrics_filepath) or not os.path.exists(dag_file) or not os.path.exists(estimate_file):
                            continue

                        dag = np.load(dag_file)
                        estimate = np.load(estimate_file)
                        shd = cpdag_shd(utils.dag_to_cpdag(dag), estimate)
                        fc_values = [0.0, 0.25, 0.5, 0.75, 1.0]
                        for val in fc_values:
                            metrics_data["SHD"][("no_init", ci_type, val)][num_nodes].append(shd)

                        with open(metrics_filepath, "r") as f:
                            metrics = json.load(f)
                            for metric, value in metrics.items():
                                for val in fc_values:
                                    metrics_data[metric][("no_init", ci_type, val)][num_nodes].append(value)

    # Plotting
    output_dir = "./obs/plots_init_seaborn"
    os.makedirs(output_dir, exist_ok=True)

    fc_colors = {
        0.0: 'tab:blue',
        0.25: 'tab:orange',
        0.5: 'tab:green',
        0.75: 'tab:red',
        1.0: 'tab:purple'
    }

    records = []

    for metric, config_data in metrics_data.items():
        for (init_type, ci_type, fc_value), node_dict in config_data.items():
            for num_nodes, values in node_dict.items():
                for v in values:
                    records.append({
                        "Metric": metric,
                        "Init Type": init_type,
                        "CI Type": ci_type,
                        "FC": fc_value,
                        "Num Nodes": num_nodes,
                        "Value": v,
                    })

    df = pd.DataFrame.from_records(records)

    for metric in df["Metric"].unique():
        metric_df = df[df["Metric"] == metric].copy()

        # Convert FC to string for better categorical handling
        metric_df["FC"] = metric_df["FC"].astype(str)

        g = sns.catplot(
            data=metric_df,
            x="Num Nodes",
            y="Value",
            hue="Init Type",
            col="FC",  # Facet by FC
            kind="point",
            ci="sd",
            dodge=True,
            height=4,
            aspect=1,
            palette="Set2",
            markers=["o", "s", "D"]
        )

        g.figure.subplots_adjust(top=0.85)
        g.figure.suptitle(f"{metric} vs Number of Nodes")
        g.set_axis_labels("Number of Nodes", metric)
        g.add_legend(title="Init Type")

        g.savefig(os.path.join(output_dir, f"{metric}_seaborn.png"))
        plt.close()




if __name__ == "__main__":
    # plot_obs_no_prior_seaborn()
    plot_obs_no_prior()
    # plot_obs_init_fc()
    # plot_obs_init_fc_seaborn()