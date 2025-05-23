import matplotlib.pyplot as plt
import argparse

from src.experiments.utils import METRICS_KEYS
import os
import json
from collections import defaultdict
import re
import numpy as np
from src.experiments.utils import cpdag_shd, compute_f1
import ges.ges.utils as utils
import matplotlib.cm as cm
from matplotlib import rcParams
import ges.ges as ges
import pickle
from gies.utils import replace_unprotected
import pandas as pd
import seaborn as sns


# --- 1) set once, before any figures are created --------------------------
rcParams.update({'axes.edgecolor': 'black',   # dark frame
                 'axes.linewidth': 1.0})      # (optional) thicker frame


pretty_metric_names = {
    "SHD": "SHD",
    "ops_eval": "Number of scoring operations",
    "time": "Time (s)",
    "BIC": "BIC",
    "BIC difference from true DAG": "BIC Difference",
    "False positives": "Excess adjacencies",
    "False negatives": "Missing adjacencies",
    "Wrong orientations": "Incorrect orientations",
    "F1": "F1 Score",
    "Precision": "Precision",
    "Recall": "Recall",
}

method_names = {
    "Without CIs": "GES",
    "With CI Score": "Old LGES (Safe)",
    "Prune With CI Score": "Old LGES (Cons)",
    "LGES With CI Score": "LGES (Safe)",
    "LGES Prune With CI Score": "LGES (Cons)",
    "lges_prune": "LGIES (Cons)",
    "lges": "LGIES (Safe)",
    "PC": "PC",
    "ges": "LGIES (Safe)", # for interventional data
    "gies": "GIES",
    "NoTears": "NoTears",
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

def generate_heatmaps(metrics_data, output_dir, n=1000):

    os.makedirs(output_dir, exist_ok=True)

    metrics = ["SHD", "False positives", "False negatives", "Wrong orientations", "F1"]
    methods = ["PC","Without CIs", "LGES With CI Score", "LGES Prune With CI Score"]
    p_values = [10, 25, 50, 100]

    for metric in metrics:
        data = []
        display_labels = []
        for method in methods:
            row = []
            for p in p_values:
                vals = metrics_data[metric][method][p]
                if vals:
                    row.append(np.mean(vals))
                else:
                    row.append(np.nan)
            data.append(row)
            display_labels.append(method_names.get(method, method))

        df = pd.DataFrame(data, index=display_labels, columns=[f"p={p}" for p in p_values])
        plt.figure(figsize=(8, 4))
        plt.rcParams.update({
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12
        })
        if metric != "F1":
            sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis_r", cbar_kws={"label": pretty_metric_names[metric]}, annot_kws={"size": 12})
        else:
            sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": pretty_metric_names[metric]}, annot_kws={"size": 12})
        # plt.title(f"{pretty_metric_names[metric]}")
        plt.ylabel("Method")
        plt.xlabel("Number of variables (p)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_{metric.replace(' ', '_').lower()}_n_samples_{n}.png"))
        plt.close()


def plot_obs_no_prior():
    base_dir = "./obs/results_no_prior"

    # Structure: metrics_data[metric][ci_type][num_nodes] = list of values
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ci_types = ["Without CIs", 
                "LGES With CI Score",
                "LGES Prune With CI Score",
                "PC",
                "NoTears"
                ]

    for folder in os.listdir(base_dir):
        match = re.match(r"vars_(\d+)_samples_(10000)_degree_2.0_model_linear-gaussian", folder)
        if not match:
            continue
        num_nodes = int(match.group(1))
        if num_nodes > 150:
            continue
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
                precision, recall, f1 = compute_f1(utils.dag_to_cpdag(dag), estimate)
                metrics_data["SHD"][ci_type][num_nodes].append(shd)
                metrics_data["False positives"][ci_type][num_nodes].append(false_positives)
                metrics_data["False negatives"][ci_type][num_nodes].append(false_negatives)
                metrics_data["Wrong orientations"][ci_type][num_nodes].append(wrong_orientations)
                metrics_data["F1"][ci_type][num_nodes].append(f1)
                metrics_data["Precision"][ci_type][num_nodes].append(precision)
                metrics_data["Recall"][ci_type][num_nodes].append(recall)


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
            "linestyle": "-", "color": "#FFA500", "marker": "o", "label": "With CI Score"  
        },
        "Prune With CI Score": {
            "linestyle": "--", "color": "rebeccapurple", "marker": "^", "label": "Prune (CI Score)"
        },
        "LGES With CI Score": {
            "linestyle": "-", "color": "tab:orange", "marker": "o", "label": "With CI Score"
        },
        "LGES Prune With CI Score": {
            "linestyle": "--", "color": "mediumpurple", "marker": "^", "label": "Prune (CI Score)"
        },
        "PC": {
            "linestyle": "-.", "color": "tab:green", "marker": "D", "label": "PC"
        },
        "NoTears": {
            "linestyle": "--", "color": "tab:red", "marker": "X", "label": "NoTears"
        },
    }

    # uncomment this line to generate heatmaps
    # generate_heatmaps(metrics_data, output_dir="./obs/plots_no_prior", n=1000)


    for metric, ci_data in metrics_data.items():

        if metric == "Precision" or metric == "Recall" or metric == "F1":
            plt.figure(figsize=(10, 10))
            rcParams.update({'font.size': 16})
        else:
            plt.figure(figsize=(10, 6))
            rcParams.update({'font.size': 14})

        for ci_type, node_dict in sorted(ci_data.items()):
            x_vals = sorted(node_dict.keys())
            means = [np.mean(node_dict[n]) for n in x_vals]
            stds = [np.std(node_dict[n]) for n in x_vals]
            if metric in ["SHD", "False positives", "False negatives", "Wrong orientations", "F1"]:
                print(f"\nMetric: {metric}")
                print("Method:", method_names[ci_type])
                print("Number of Variables (p):", x_vals)
                print("Means:", [f"{mean:.2f}" for mean in means])
                print("Standard Deviations:", [f"{std:.2f}" for std in stds])
            s = styles[ci_type]

            plt.errorbar(
                x_vals, means, 
                # yerr=[np.zeros_like(stds), stds],  # asymmetric error bars: [lower, upper]
                yerr=stds,
                label=method_names[ci_type],
                linestyle=s["linestyle"],
                color=s["color"],
                marker=s["marker"],
                capsize=4,
                linewidth=2
            )

        if metric == "time":
            plt.yscale("log")
            
        desired_order = ["NoTears", "PC", "Without CIs", "With CI Score", "Prune With CI Score", "LGES With CI Score", "LGES Prune With CI Score"]
        desired_order = [method_names[ci] for ci in desired_order if ci in method_names]

        # Reorder
        handles, labels = plt.gca().get_legend_handles_labels()
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]))
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        plt.errorbar(
            x_vals,
            means,
            yerr=[np.zeros_like(stds), stds],
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
        plt.xlabel("Number of variables (p)")
        plt.ylabel(pretty_metric_names[metric] if metric in pretty_metric_names.keys() else metric)
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



def plot_fc_vs_metric_for_fixed_nvars(base_dir="./obs/results_init",
                                      output_dir="./obs/plots_init",
                                      fixed_nvars=50, max_std=20):
    """
    One figure per metric (SHD, time)
    x-axis  : fraction-correct (fc)
    y-axis  : metric value
    Only data for `fixed_nvars` (#variables) are used.
    """
    import os, re, json, numpy as np, matplotlib.pyplot as plt
    from collections import defaultdict

    # ------------------------------------------------------------------ #
    # ------------ LOAD DATA  ----------------------------------------- #
    # metrics_data[metric][method_key][fc] = list of values
    # method_key = (ci_type, init_type)
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ci_tags = {
        "Without CIs": "without_cis",
        "LGES With CI Score": "lges_with_ci_score",
        "LGES Prune With CI Score": "lges_prune_with_ci_score",
    }
    init_types = ["init", "req", "no_init"]
    fc_values_found = set()
    rcParams.update({'font.size': 14})

    for folder in os.listdir(base_dir):
        m = re.match(r"vars_(\d+)_samples_1000_degree_2_model_linear-gaussian", folder)
        if not m or int(m.group(1)) != fixed_nvars:
            continue
        for trial in os.listdir(os.path.join(base_dir, folder)):
            tmatch = re.match(r"trial_(\d+)", trial)
            if not tmatch:
                continue
            trial_path = os.path.join(base_dir, folder, trial)
            dag_path   = os.path.join(trial_path, f"dag_{tmatch.group(1)}.npy")
            if not os.path.exists(dag_path):
                continue
            dag = np.load(dag_path)

            for init_type in init_types:
                subdir = os.path.join(trial_path, init_type)
                if not os.path.isdir(subdir):
                    continue
                for ci_type, tag in ci_tags.items():

                    # -------- no-init (only one file, no fc) -------- #
                    if init_type == "no_init":
                        # if ci_type != "Without CIs":    # only this combo exists
                        #     continue
                        est = os.path.join(subdir, f"estimate_{tag}_{tmatch.group(1)}.npy")
                        met = os.path.join(subdir, f"metrics_{tag}_{tmatch.group(1)}.json")
                        if not (os.path.exists(est) and os.path.exists(met)):
                            continue
                        fc = None
                        metrics_data["SHD"][(ci_type, init_type)][fc].append(
                            cpdag_shd(utils.dag_to_cpdag(dag), np.load(est)))
                        with open(met) as f:
                            j = json.load(f)
                            if "time" in j:
                                metrics_data["time"][(ci_type, init_type)][fc].append(j["time"])
                        continue

                    # -------- init / req  : many fc-tagged files ---- #
                    pattern = re.compile(rf"estimate_{tag}_fc_([0-9.]+)_{tmatch.group(1)}\.npy")
                    for fname in os.listdir(subdir):
                        mm = pattern.match(fname)
                        if not mm:
                            continue
                        fc = float(mm.group(1))
                        fc_values_found.add(fc)
                        est = os.path.join(subdir, fname)
                        met = os.path.join(
                            subdir, f"metrics_{tag}_fc_{fc}_{tmatch.group(1)}.json")
                        if not (os.path.exists(est) and os.path.exists(met)):
                            continue
                        metrics_data["SHD"][(ci_type, init_type)][fc].append(
                            cpdag_shd(utils.dag_to_cpdag(dag), np.load(est)))
                        with open(met) as f:
                            j = json.load(f)
                            if "time" in j:
                                metrics_data["time"][(ci_type, init_type)][fc].append(j["time"])

    fc_values = sorted(fc_values_found)

    # ------------------------------------------------------------------ #
    # ------------ PLOTTING  ------------------------------------------ #
    style_map = {
        ("Without CIs", "no_init"): "GES-0",
        ("Without CIs", "init")   : "GES-Init",
        ("With CI Score", "init") : "Old LGES-Init (Safe)",
        ("With CI Score", "req")  : "Old LGES (Safe)",
        ("Prune With CI Score", "init"): "Old LGES-Init (Cons)",
        ("Prune With CI Score", "req") : "Old LGES (Cons)",
        ("LGES With CI Score", "no_init") : "LGES-0 (Safe)",
        ("LGES With CI Score", "init") : "LGES-Init (Safe)",
        ("LGES With CI Score", "req")  : "LGES (Safe)",
        ("LGES Prune With CI Score", "no_init"): "LGES-0 (Cons)",
        ("LGES Prune With CI Score", "init"): "LGES-Init (Cons)",
        ("LGES Prune With CI Score", "req") : "LGES (Cons)",
    }
    colour = {"GES-0":"tab:blue","GES-Init":"tab:blue",
              "LGES-0 (Safe)":"tab:orange", "LGES-Init (Safe)":"tab:orange","LGES (Safe)":"darkorange",
              "LGES-0 (Cons)":"tab:purple", "LGES-Init (Cons)":"tab:purple","LGES (Cons)":"mediumpurple",
              "Old LGES-Init (Safe)":"tab:orange","Old LGES (Safe)":"darkorange",
              "Old LGES-Init (Cons)":"tab:purple","Old LGES (Cons)":"mediumpurple"}
    
    ls  = {"GES-0":":", "LGES-0 (Safe)":":", "LGES-0 (Cons)":":",
            "GES-Init":"-","LGES-Init (Safe)":"-","LGES (Safe)":"--",
           "LGES-Init (Cons)":"-","LGES (Cons)":"--", "Old LGES-Init (Safe)":"-",
           "Old LGES (Safe)":"--","Old LGES-Init (Cons)":"-","Old LGES (Cons)":"--"}
    mk  = {"GES-0":"s", "LGES-0 (Safe)":"s", "LGES-0 (Cons)":"s",
            "GES-Init":"o","LGES-Init (Safe)":"o","LGES (Safe)":"D",
           "LGES-Init (Cons)":"o","LGES (Cons)":"D", "Old LGES-Init (Safe)":"o",
           "Old LGES (Safe)":"D","Old LGES-Init (Cons)":"o","Old LGES (Cons)":"D"}

    os.makedirs(output_dir, exist_ok=True)

    for metric in ["SHD", "time"]:
        plt.figure(figsize=(8,6))
        for method_key, fc_dict in metrics_data[metric].items():
            label = style_map.get(method_key)
            if not label:                     # skip unknown combos
                continue

            # --- horizontal line for no_init (fc=None) ------------- #
            if None in fc_dict:
                mean_val = np.mean(fc_dict[None])
                plt.axhline(mean_val, color=colour[label], linestyle=ls[label],
                            linewidth=2, label=label)
                continue

            # --- line with error bars over fc ---------------------- #
            xs, ys, es = [], [], []
            for fc in fc_values:
                vals = fc_dict.get(fc, [])
                if vals:
                    xs.append(fc)
                    ys.append(np.mean(vals))
                    es.append(min(np.std(vals), max_std))
            if xs:
                plt.errorbar(xs, ys, 
                            yerr=es, 
                             marker=mk[label],
                             color=colour[label], linestyle=ls[label],
                             linewidth=2, capsize=4, elinewidth=1, label=label)
                

        plt.xlabel("Fraction Correct (fc)")
        plt.ylabel("Time (s)" if metric=="time" else metric)
        if metric == "time":
            plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=.6)
        plt.legend(fontsize=12, title="Method", loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                     f"{metric}_vs_fc_nvars_{fixed_nvars}.png"), dpi=300)
        plt.close()


def plot_exp_no_prior():
    base_dir = "./exp/results_no_prior"

    # Structure: metrics_data[metric][alg][num_nodes] = list of values
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    #algs = ["ges", "gies", "lges_prune", "lges"]
    algs = ["gies", "lges_prune", "lges"]

    rcParams.update({'font.size': 14})

    for folder in os.listdir(base_dir):
        match = re.match(r"vars_(\d+)_samples_10000_degree_2.0_model_linear-gaussian", folder)
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
            targets_file_path = os.path.join(trial_path, f"targets_{trial_number}.npy")

            if not os.path.exists(dag_file) or not os.path.exists(targets_file_path):
                continue

            G = np.load(dag_file)
            with open(targets_file_path, 'rb') as f:
                        targets = pickle.load(f)
            imec = replace_unprotected(G, targets)

            for alg in algs:
                metrics_file_path = os.path.join(trial_path, f"metrics_{alg}_{trial_number}.json")
                estimate_file_path = os.path.join(trial_path, f"estimate_{alg}_{trial_number}.npy")

                if not os.path.exists(metrics_file_path) or not os.path.exists(estimate_file_path):
                    continue

                estimate = np.load(estimate_file_path)
                shd = cpdag_shd(imec, estimate)
                metrics_data["SHD"][alg][num_nodes].append(shd)

                with open(metrics_file_path, "r") as f:
                    metrics = json.load(f)
                    for metric, value in metrics.items():
                        metrics_data[metric][alg][num_nodes].append(value)

    # Plotting
    output_dir = "./exp/plots_no_prior"
    os.makedirs(output_dir, exist_ok=True)

    # Style maps
    styles = {
        # "ges":   {"linestyle": ":",  "color": "tab:orange",   "marker": "s"},
        "gies": {"linestyle": "-",  "color": "tab:blue", "marker": "s"},
        "lges_prune": {"linestyle": "--", "color": "tab:orange", "marker": "^"},
        "lges": {"linestyle": "-.", "color": "tab:red", "marker": "o"},
    }

    rcParams.update({'font.size': 14})
    max_error = 5 

    for metric, ci_data in metrics_data.items():
        plt.figure(figsize=(10, 6))

        for alg, node_dict in sorted(ci_data.items()):
            x_vals = sorted(node_dict.keys())
            means = [np.mean(node_dict[n]) for n in x_vals]
            stds = [np.std(node_dict[n]) if n != 35 else min(np.std(node_dict[n]), max_error) for n in x_vals ]
            # stds = [min(np.std(node_dict[n]), max_error) for n in x_vals]
            s = styles[alg]

            plt.errorbar(
                x_vals, means, 
                yerr=stds,
                label=alg,
                linestyle=s["linestyle"],
                color=s["color"],
                marker=s["marker"],
                capsize=4,
                linewidth=2
            )


        plt.xlabel("Number of variables (p)")
        plt.ylabel(pretty_metric_names[metric])
        if metric == "time":
            plt.yscale("log")
        # plt.yscale("log")
        #  plt.title(f"{metric.capitalize()} vs Number of Nodes")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        # Build legend entries sorted by final mean
        legend_items = []
        for alg, node_dict in ci_data.items():
            x_vals = sorted(node_dict.keys())
            means = [np.mean(node_dict[n]) for n in x_vals]
            if means:
                final_mean = means[-1]
                legend_items.append((final_mean, alg))

        # Sort from highest to lowest final mean
        legend_items.sort(reverse=True)

        # Build dummy handles in the same sorted order
        handles, labels = [], []
        for _, alg in legend_items:
            s = styles[alg]
            h, = plt.plot([], [], label=alg, color=s["color"], linestyle=s["linestyle"], marker=s["marker"])
            handles.append(h)
            labels.append(method_names.get(alg, alg))  # ← use pretty label

        # Draw sorted legend
        plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0))

        plt.savefig(os.path.join(output_dir, f"{metric}.png"), dpi=300, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific plotting functions.")
    parser.add_argument("--plot", type=str, choices=["obs_no_prior", "exp_no_prior", "obs_init"], 
                        required=True, help="Specify which plot function to run.")
    args = parser.parse_args()

    if args.plot == "obs_no_prior":
        plot_obs_no_prior()
    elif args.plot == "exp_no_prior":
        plot_exp_no_prior()
    elif args.plot == "obs_init":
        plot_fc_vs_metric_for_fixed_nvars()
