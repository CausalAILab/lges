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
import pickle
from gies.utils import replace_unprotected

# --- 1) set once, before any figures are created --------------------------
rcParams.update({'axes.edgecolor': 'black',   # dark frame
                 'axes.linewidth': 1.0})      # (optional) thicker frame


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
    "With CI Score": "LGES (Safe)",
    "Prune With CI Score": "LGES (Cons)",
    "XGES": "XGES",
    "PC": "PC",
    "ges": "LGIES (Safe)", # for interventional data
    "gies": "GIES",
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
                x_vals, means, 
                yerr=[np.zeros_like(stds), stds],  # asymmetric error bars: [lower, upper]
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


def plot_fc_vs_metric_for_fixed_nvars(base_dir="./obs/results_init",
                                      output_dir="./obs/plots_fc_fixed",
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
        "With CI Score": "with_ci_score",
        "Prune With CI Score": "prune_with_ci_score",
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
                        if ci_type != "Without CIs":    # only this combo exists
                            continue
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
        ("Without CIs", "no_init"): "GES",
        ("Without CIs", "init")   : "GES-Init",
        ("With CI Score", "init") : "LGES-Init (Safe)",
        ("With CI Score", "req")  : "LGES (Safe)",
        ("Prune With CI Score", "init"): "LGES-Init (Cons)",
        ("Prune With CI Score", "req") : "LGES (Cons)",
    }
    colour = {"GES":"tab:blue","GES-Init":"tab:blue",
              "LGES-Init (Safe)":"tab:orange","LGES (Safe)":"darkorange",
              "LGES-Init (Cons)":"tab:purple","LGES (Cons)":"mediumpurple"}
    ls  = {"GES":":","GES-Init":"-","LGES-Init (Safe)":"-","LGES (Safe)":"--",
           "LGES-Init (Cons)":"-","LGES (Cons)":"--"}
    mk  = {"GES":"s","GES-Init":"o","LGES-Init (Safe)":"o","LGES (Safe)":"D",
           "LGES-Init (Cons)":"o","LGES (Cons)":"D"}

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
                            #  yerr=es, 
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

def plot_obs_init_fc(base_dir="./obs/results_init", output_dir="./obs/plots_init"):
    os.makedirs(output_dir, exist_ok=True)
    rcParams.update({'font.size': 14})

    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # metric → (ci_type, init_type, fc) → num_nodes → list
    fc_values_found = set()

    ci_tags = {
        "Without CIs": "without_cis",
        "With CI Score": "with_ci_score",
        "Prune With CI Score": "prune_with_ci_score"
    }

    init_types = ["init", "req", "no_init"]

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    for folder in os.listdir(base_dir):
        match = re.match(r"vars_(\d+)_samples_1000_degree_2_model_linear-gaussian", folder)
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

            dag_path = os.path.join(trial_path, f"dag_{trial_number}.npy")
            if not os.path.exists(dag_path):
                continue
            dag = np.load(dag_path)

            for init_type in init_types:
                subdir = os.path.join(trial_path, init_type)
                if not os.path.isdir(subdir):
                    continue

                for ci_type, tag in ci_tags.items():
                    if init_type == "no_init":
                        if ci_type != "Without CIs":
                            continue
                        fc = None
                        estimate_file = os.path.join(subdir, f"estimate_{tag}_{trial_number}.npy")
                        metrics_file = os.path.join(subdir, f"metrics_{tag}_{trial_number}.json")
                        if not os.path.exists(estimate_file) or not os.path.exists(metrics_file):
                            continue

                        estimate = np.load(estimate_file)
                        shd = cpdag_shd(utils.dag_to_cpdag(dag), estimate)
                        metrics_data["SHD"][(ci_type, init_type, fc)][num_nodes].append(shd)

                        with open(metrics_file) as f:
                            metrics = json.load(f)
                            if "time" in metrics:
                                metrics_data["time"][(ci_type, init_type, fc)][num_nodes].append(metrics["time"])
                        continue

                    # init or req, look for fc-tagged files
                    pattern = re.compile(rf"estimate_{tag}_fc_([0-9.]+)_{trial_number}.npy")
                    for fname in os.listdir(subdir):
                        match = pattern.match(fname)
                        if not match:
                            continue
                        fc = float(match.group(1))
                        fc_values_found.add(fc)

                        estimate_file = os.path.join(subdir, fname)
                        metrics_file = os.path.join(subdir, f"metrics_{tag}_fc_{fc}_{trial_number}.json")
                        if not os.path.exists(estimate_file) or not os.path.exists(metrics_file):
                            continue

                        estimate = np.load(estimate_file)
                        shd = cpdag_shd(utils.dag_to_cpdag(dag), estimate)
                        metrics_data["SHD"][(ci_type, init_type, fc)][num_nodes].append(shd)

                        with open(metrics_file) as f:
                            metrics = json.load(f)
                            if "time" in metrics:
                                metrics_data["time"][(ci_type, init_type, fc)][num_nodes].append(metrics["time"])

    fc_values = sorted(fc_values_found)

    # Style map
    style_map = {
        ("Without CIs", "no_init"): "GES",
        ("Without CIs", "init"): "GES-Init",
        ("With CI Score", "init"): "LGES-Init (SafeInsert)",
        ("With CI Score", "req"): "LGES (SafeInsert)",
        ("Prune With CI Score", "init"): "LGES-Init (ConservativeInsert)",
        ("Prune With CI Score", "req"): "LGES (ConservativeInsert)",
    }

    for metric in ["SHD", "time"]:
        fig, axs = plt.subplots(
            1, len(fc_values), figsize=(6 * len(fc_values), 5), sharey=True
        )
        axs = axs if len(fc_values) > 1 else [axs]

        for i, fc in enumerate(fc_values):
            ax = axs[i]

            legend_rows = []      # (final_mean, handle, label)

            for (ci_type, init_type, m_fc), node_dict in metrics_data[metric].items():
                # pick the curves that belong in this subplot
                if m_fc is not None and not np.isclose(m_fc, fc):
                    continue

                x_vals = sorted(node_dict.keys())
                if not x_vals:
                    continue
                means = [np.mean(node_dict[n]) for n in x_vals]
                stds  = [np.std(node_dict[n])  for n in x_vals]

                style   = style_map.get((ci_type, init_type), {})
                line,   = ax.plot(                       # bold mean line
                    x_vals, means,
                    linestyle=style.get("linestyle", "-"),
                    color=style.get("color",   "black"),
                    marker=style.get("marker", "o"),
                    linewidth=2,
                )

                # upper-only error bars (faint)
                ax.errorbar(
                    x_vals, means,
                    yerr=[np.zeros_like(stds), stds],    # [lower, upper]
                    uplims=True,                         # cap marker
                    linestyle="none",
                    ecolor=style.get("color", "black"),
                    elinewidth=1,
                    alpha=0.25,
                    capsize=3,
                )

                # store for legend ordering
                label = style.get("label", f"{init_type} ({ci_type})")
                final_mean = means[-1]                   # mean at largest #nodes
                legend_rows.append((final_mean, line, label))

            # -- subplot cosmetics
            ax.set_title(f"FC = {fc}")
            ax.set_xlabel("Number of variables (p)")
            if i == 0:
                ax.set_ylabel(pretty_metric_names.get(metric, metric))
            ax.grid(True, linestyle="--", alpha=0.6)

            # order legend: lowest final mean first
            legend_rows.sort(key=lambda t: t[0])
            handles = [r[1] for r in legend_rows]
            labels  = [r[2] for r in legend_rows]
            ax.legend(handles, labels, title="Method", fontsize=11,
                    loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_by_fc.png"), dpi=300)
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
        g.set_axis_labels("Number of variables (p)", metric)
        g.add_legend(title="Init Type")

        g.savefig(os.path.join(output_dir, f"{metric}_seaborn.png"))
        plt.close()


def plot_exp_no_prior():
    base_dir = "./exp/results_no_prior"

    # Structure: metrics_data[metric][alg][num_nodes] = list of values
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    algs = ["ges", "gies"]

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
        "ges":   {"linestyle": ":",  "color": "tab:orange",   "marker": "s"},
        "gies": {"linestyle": "-",  "color": "tab:blue", "marker": "o"},
    }

    rcParams.update({'font.size': 14})

    for metric, ci_data in metrics_data.items():
        plt.figure(figsize=(10, 6))

        for alg, node_dict in sorted(ci_data.items()):
            x_vals = sorted(node_dict.keys())
            means = [np.mean(node_dict[n]) for n in x_vals]
            stds = [np.std(node_dict[n]) for n in x_vals]
            s = styles[alg]

            plt.errorbar(
                x_vals, means, 
                yerr=[np.zeros_like(stds), stds],  # asymmetric error bars: [lower, upper]
                # yerr=stds,
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
    # plot_obs_no_prior_seaborn()
    plot_obs_no_prior()
    # plot_obs_init_fc()
    plot_exp_no_prior()
    plot_fc_vs_metric_for_fixed_nvars()
    # plot_obs_init_fc_seaborn()
    # plot_obs_no_prior_sns()
    # plot_exp_no_prior_sns()
    # plot_fc_vs_metric_sns()
    a = 0
    