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
from gies.gies.utils import replace_unprotected
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


# --- 1) set once, before any figures are created --------------------------
rcParams.update({'axes.edgecolor': 'black',   # dark frame
                 'axes.linewidth': 1.0})      # (optional) thicker frame


pretty_metric_names = {
    "SHD": "Structural Hamming Distance",
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


#"lges_prune",  #lges (cons) + iorient
                                # "lges", # lges (safe) + iorient
                                # "gies",  # gies
                                # "lgies_prune", # gies + cons insert
                                # "lgies"] # gies + safeinsert
method_names = {
    "Without CIs": "GES",
    "LGES With CI Score": "LGES-0 (Safe)",
    "LGES Prune With CI Score": "LGES-0 (Cons)",
    "lges_prune": "LGIES (Cons)",
    "lges": "LGIES (Safe)",
    "PC": "PC",
    "NoTears": "NoTears",
    "Iterate Without CIs": "GES (Repeats)",
    "XGES Without CIs": "XGES-0",
    "XGES With CI Score": "LGES (Safe)",
    "XGES Prune With CI Score": "LGES (Cons)",
    "Full XGES Without CIs": "XGES",
    "Full XGES With CI Score": "LGES+ (Safe)",
    "Full XGES Prune With CI Score": "LGE (Cons)", 
    "Iterate LGES With CI Score": "LGES (Safe, Repeats)",
    "Iterate LGES Prune With CI Score": "LGES (Cons, Repeats)",
    # Interventional data
    "gies": "GIES",
    "lges": "LGES (Safe) + I-Orient",
    "lges_prune": "LGES (Cons) + I-Orient",
    "lgies": "LGIES (Safe)",
    "lges_prune": "LGIES (Cons)"
    
}

# Styles (reuse your mapping for consistency)
styles = {
    "Without CIs":                {"ls": "-",  "color": "tab:blue",     "marker": "s", "label": "GES"},
    "XGES With CI Score":         {"ls": "-", "color": "tab:orange",   "marker": "o", "label": "LGES (Safe)"},
    "XGES Prune With CI Score":   {"ls": "-", "color": "mediumpurple", "marker": "^", "label": "LGES (Cons)"},
    "PC":                         {"ls": "-",  "color": "tab:green",    "marker": "D", "label": "PC"},
    "XGES Without CIs":           {"ls": "-",  "color": "chocolate",     "marker": "s", "label": "XGES-0"},
    "Full XGES Without CIs":      {"ls": "-",  "color": "tab:cyan",    "marker": "s", "label": "XGES"},
    # APPENDIX
    "NoTears":                    {"ls": ":", "color": "tab:red", "marker": "X", "label": "NoTears"},
    "LGES With CI Score":         {"ls": "--", "color": "tab:orange", "marker": "o", "label": "LGES-0 (Safe)"},
    "LGES Prune With CI Score":   { "ls": "--", "color": "mediumpurple", "marker": "^", "label": "LGES-0 (Cons)"},
    "Iterate Without CIs":        {"ls": "-.", "color": "tab:blue", "marker": "s", "label": "Without CIs"},
    "Iterate LGES With CI Score": {"ls": "-.", "color": "tab:orange", "marker": "o", "label": "With CI Score"},
    "Iterate LGES Prune With CI Score": {"ls": "-.", "color": "mediumpurple", "marker": "^", "label": "Prune (CI Score)"},
    "Full XGES With CI Score": {"ls": "-", "color": "mediumpurple", "marker": "o", "label": "XGES (Safe)"},
    "Full XGES Prune With CI Score": {"ls": "--", "color": "mediumpurple", "marker": "^", "label": "XGES (Cons)"},
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
   
    # which methods to plot
    ci_types = {
        "Without CIs",
        "LGES With CI Score",
        "LGES Prune With CI Score",
        "PC",
        # "NoTears",
        # "Iterate Without CIs",
        # "XGES Without CIs",
        # "XGES With CI Score",
        # "XGES Prune With CI Score",
        "Full XGES Without CIs",
        # "Full XGES With CI Score",
        # "Full XGES Prune With CI Score",
        # "Iterate LGES With CI Score",
        # "Iterate LGES Prune With CI Score",
    }

    for folder in os.listdir(base_dir):
        match = re.match(r"vars_(\d+)_samples_(10000)_degree_2.0_model_linear-gaussian", folder)
        if not match:
            continue
        num_nodes = int(match.group(1))
        if num_nodes < 50 or num_nodes > 150:
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
            if metric in ["SHD", "False positives", "False negatives", "Wrong orientations", "F1", "time"]:
                print(f"\nMetric: {metric}")
                print("Method:", method_names[ci_type])
                print("Number of Variables (p):", x_vals)
                print("Means:", [f"{mean:.2f}" for mean in means])
                print("Standard Deviations:", [f"{std:.2f}" for std in stds])
            s = styles[ci_type]

            label = method_names.get(ci_type, ci_type)  # <-- here’s 'label'

            plt.errorbar(
                x_vals, means, 
                # yerr=[np.zeros_like(stds), stds],  # asymmetric error bars: [lower, upper]
                yerr=stds,
                label=method_names[ci_type],
                linestyle=s["ls"],
                color=s["color"],
                marker=s["marker"],
                markersize=5,
                capsize=2, elinewidth=1, alpha=0.7
            )

        if metric == "time":
            plt.yscale("log")
        
        if metric == "time":
            desired_order = ["NoTears", 
                            "PC", 
                            "Without CIs", 
                            "Iterate Without CIs",
                            "With CI Score", 
                            "Prune With CI Score", 
                            "LGES With CI Score",
                            "Iterate LGES With CI Score", 
                            "LGES Prune With CI Score",
                            "Iterate LGES Prune With CI Score",
                            "XGES Without CIs",
                            "XGES With CI Score",
                            "XGES Prune With CI Score",
                            "Full XGES Without CIs",
                            "Full XGES With CI Score",
                            "Full XGES Prune With CI Score",
                            ]
        else:
             desired_order = ["NoTears", 
                            "PC", 
                            "Without CIs", 
                            "Iterate Without CIs",
                            "With CI Score", 
                            "Prune With CI Score", 
                            "LGES With CI Score",
                            "Iterate LGES With CI Score", 
                            "LGES Prune With CI Score",
                            "Iterate LGES Prune With CI Score",
                            "XGES Without CIs",
                            "XGES With CI Score",
                            "XGES Prune With CI Score",
                            "Full XGES Without CIs",
                            "Full XGES With CI Score",
                            "Full XGES Prune With CI Score",
                            ]

        desired_order = [method_names[ci] for ci in desired_order if ci in method_names]

        # Reorder
        handles, labels = plt.gca().get_legend_handles_labels()
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]))
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)

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


def plot_obs_no_prior_seaborn(
    base_dir="./obs/results_no_prior",
    output_dir="./obs/plots_no_prior",
    xticks=(0, 25, 50, 75, 100, 125, 150),
    square=True
):
    # ---------- INGEST ----------
    # metrics_data[metric][ci_type][num_nodes] = list of values
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    ci_types = {
        # main paper
        "Without CIs",
        "XGES With CI Score",
        "XGES Prune With CI Score",
        "Full XGES Without CIs",
        "PC",
        # "LGES With CI Score",
        # "LGES Prune With CI Score",
        # "XGES Without CIs",
        # "Full XGES With CI Score",
        # "Full XGES Prune With CI Score",
    }

    for folder in os.listdir(base_dir):
        m = re.match(r"vars_(\d+)_samples_(10000)_degree_3.0_model_linear-gaussian", folder)
        if not m:
            continue
        num_nodes = int(m.group(1))
        if not (num_nodes <= 150):
            continue

        folder_path = os.path.join(base_dir, folder)
        for trial in os.listdir(folder_path):
            tm = re.match(r"trial_(\d+)", trial)
            if not tm:
                continue
            tnum = tm.group(1)
            tpath = os.path.join(folder_path, trial)
            if not os.path.isdir(tpath):
                continue

            dag_file  = os.path.join(tpath, f"dag_{tnum}.npy")
            data_file = os.path.join(tpath, f"data_{tnum}.npy")
            if not (os.path.exists(dag_file) and os.path.exists(data_file)):
                continue

            dag = np.load(dag_file)

            for ci_type in ci_types:
                metrics_fp  = os.path.join(tpath, f"metrics_{ci_type.replace(' ','_').lower()}_{tnum}.json")
                estimate_fp = os.path.join(tpath, f"estimate_{ci_type.replace(' ','_').lower()}_{tnum}.npy")
                if not (os.path.exists(metrics_fp) and os.path.exists(estimate_fp)):
                    continue

                estimate = np.load(estimate_fp, allow_pickle=True)
                if ci_type == "PC":
                    estimate = estimate.item()
                    estimate = process_pc_estimate(estimate)

                shd, fp, fn, wo = cpdag_shd(utils.dag_to_cpdag(dag), estimate)
                precision, recall, f1 = compute_f1(utils.dag_to_cpdag(dag), estimate)

                metrics_data["SHD"][ci_type][num_nodes].append(shd)
                metrics_data["False positives"][ci_type][num_nodes].append(fp)
                metrics_data["False negatives"][ci_type][num_nodes].append(fn)
                metrics_data["Wrong orientations"][ci_type][num_nodes].append(wo)
                metrics_data["F1"][ci_type][num_nodes].append(f1)
                metrics_data["Precision"][ci_type][num_nodes].append(precision)
                metrics_data["Recall"][ci_type][num_nodes].append(recall)

                with open(metrics_fp, "r") as f:
                    other = json.load(f)
                for met, val in other.items():
                    metrics_data[met][ci_type][num_nodes].append(val)

    # ---------- PLOTTING ----------


    
    os.makedirs(output_dir, exist_ok=True)


    # Pretty seaborn background (grey), light grid, etc.
    sns.set_theme(style="darkgrid", context="notebook")
    rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "legend.frameon": False,
        "font.size": 13,
        "axes.labelsize": 16,   # axis labels
        "xtick.labelsize": 14,  # x tick labels
        "ytick.labelsize": 14,  # y tick labels
        "legend.fontsize": 13,  # legend text
        "legend.title_fontsize": 14,
    })

    



    def _legend_order_keys(metric):
        # base order template; filtered to keys present in `method_names`
        base = ["NoTears", "PC", "Without CIs", "Iterate Without CIs",
                "With CI Score", "Prune With CI Score",
                "LGES With CI Score", "Iterate LGES With CI Score",
                "LGES Prune With CI Score", "Iterate LGES Prune With CI Score",
                "XGES Without CIs", "XGES With CI Score", "XGES Prune With CI Score",
                "Full XGES Without CIs", "Full XGES With CI Score", "Full XGES Prune With CI Score"]
        return [k for k in base if k in method_names]

    for metric, ci_data in metrics_data.items():
        # Square canvas (independent x/y scales) if requested
        fig_size = (8, 8) if square else ((10, 10) if metric in {"Precision", "Recall", "F1"} else (10, 6))
        fig, ax = plt.subplots(figsize=fig_size)
        if square:
            try:
                ax.set_box_aspect(1)
            except Exception:
                pass  # older Matplotlib

        legend_handles = {}
        last_y = {}  # for legend ordering by last x

        for ci_type, node_dict in sorted(ci_data.items()):
            x_vals = sorted(node_dict.keys())
            y_mean = np.array([np.mean(node_dict[n]) for n in x_vals], dtype=float)
            y_std  = np.array([np.std(node_dict[n])  for n in x_vals], dtype=float)

            # style + label
            s = styles[ci_type]
            label = method_names.get(ci_type, s.get("label", ci_type))

            # seaborn line (no CI fill), keep markers white-faced for legibility
            h = sns.lineplot(
                x=x_vals, y=y_mean, ax=ax, label=label,
                color=s["color"], linestyle=s["ls"],
                marker=s["marker"], markers=True,
                markerfacecolor=s["color"],
                markeredgecolor=s["color"], 
                dashes=False, linewidth=2,
                markeredgewidth=1.1, markersize=5
            )

            # Full error bars at every point
            ax.errorbar(
                x_vals, y_mean, yerr=y_std,
                fmt="none", ecolor=s["color"],
                elinewidth=2, capsize=2, 
            )

            legend_handles[label] = h.lines[-1] if h.lines else None
            last_y[label] = float(y_mean[-1]) if len(y_mean) else np.nan

        # Axes formatting
        if metric == "time":
            ax.set_yscale("log")

        ax.set_xlabel("Number of variables (p)")
        ax.set_ylabel(pretty_metric_names.get(metric, metric))

        # subtle y padding
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.05)

        sns.despine(ax=ax)

        # -------- Legend ordered by last x value --------
        # Build list of (label, style, last_y) only for plotted labels
        legend_items = []
        for ci_type, s in styles.items():
            lab = method_names.get(ci_type, s.get("label", ci_type))
            if lab in legend_handles:
                legend_items.append((lab, s["color"], s["ls"], s["marker"], last_y.get(lab, np.nan)))

        # Sort (NaNs last). Desc if higher_better; otherwise asc.
        legend_items.sort(key=lambda t: (np.isnan(t[4]), t[4]), reverse=(metric in {"SHD"}))

        # Custom handles to show color + dash + marker
        handles, labels = [], []
        for (lab, color, ls, marker, _) in legend_items:
            handles.append(Line2D([0], [0],
                color=color, linestyle=ls, linewidth=2,
                marker=marker, markeredgewidth=1.1, markersize=6,
                label=lab
            ))
            labels.append(lab)

        if handles:
            ax.legend(handles, labels, title="Method",
                      loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      fontsize=12, title_fontsize=12, borderaxespad=0.0)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{metric}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)



def plot_lges_time_shd(
    base_dirs=["./obs/results_no_prior", "./exp/results_no_prior"],
    output_dir="./obs/plots_no_prior",
    nodes=[150],                     # e.g. [50, 100, 150]; None = all p
    boxplot=False,                   # False -> bars (mean±std)
    drop_nonpositive_time=True,     # remove 0/neg times for log scale
    time_eps=1e-9,                  # small floor if you’d rather shift than drop
    include_strip=True,             # overlay points (faint)
):
    """
    Two panels: time (log y) and SHD. Shows Baseline/SafeInsert/ConsInsert
    for XGES and Full XGES only.
    """

    # ---------- INGEST ----------
    # metrics_data[metric][ci_type][num_nodes] = list of values
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    ci_types = {
        "Without CIs",
        "LGES With CI Score",
        "LGES Prune With CI Score",
        "XGES Without CIs",
        "XGES With CI Score",
        "XGES Prune With CI Score",
        "Full XGES Without CIs",
        "Full XGES With CI Score",
        "Full XGES Prune With CI Score",
        "gies",
        "lgies",
        "lgies_prune"
    }

    for base_dir in base_dirs:
        for folder in os.listdir(base_dir):
            m = re.match(r"vars_(\d+)_samples_(10000)_degree_3.0_model_linear-gaussian", folder)
            if not m:
                continue
            num_nodes = int(m.group(1))
            if not (num_nodes <= 150):
                continue

            folder_path = os.path.join(base_dir, folder)
            for trial in os.listdir(folder_path):
                tm = re.match(r"trial_(\d+)", trial)
                if not tm:
                    continue
                tnum = tm.group(1)
                tpath = os.path.join(folder_path, trial)
                if not os.path.isdir(tpath):
                    continue

                dag_file  = os.path.join(tpath, f"dag_{tnum}.npy")
                data_file = os.path.join(tpath, f"data_{tnum}.npy")
                if not (os.path.exists(dag_file) and os.path.exists(data_file)):
                    continue

                dag = np.load(dag_file)

                for ci_type in ci_types:
                    metrics_fp  = os.path.join(tpath, f"metrics_{ci_type.replace(' ','_').lower()}_{tnum}.json")
                    estimate_fp = os.path.join(tpath, f"estimate_{ci_type.replace(' ','_').lower()}_{tnum}.npy")
                    if not (os.path.exists(metrics_fp) and os.path.exists(estimate_fp)):
                        continue

                    estimate = np.load(estimate_fp, allow_pickle=True)

                    shd, fp, fn, wo = cpdag_shd(utils.dag_to_cpdag(dag), estimate)

                    metrics_data["SHD"][ci_type][num_nodes].append(shd)
                    
                    with open(metrics_fp, "r") as f:
                        other = json.load(f)
                    for met, val in other.items():
                        metrics_data[met][ci_type][num_nodes].append(val)

        # ---------- PLOTTING ----------

    
    os.makedirs(output_dir, exist_ok=True)

    # ---- mapping from your ci_type keys to (Family, Variant) ----
    mapping = [
        ("GES",      "Baseline",   "Without CIs"),
        ("GES",      "SafeInsert",   "LGES With CI Score"),
        ("GES",      "ConsInsert",   "LGES Prune With CI Score"),
        ("XGES-0",      "Baseline",   "XGES Without CIs"),
        ("XGES-0",      "SafeInsert", "XGES With CI Score"),
        ("XGES-0",      "ConsInsert", "XGES Prune With CI Score"),
        ("XGES", "Baseline",   "Full XGES Without CIs"),
        ("XGES", "SafeInsert", "Full XGES With CI Score"),
        ("XGES", "ConsInsert", "Full XGES Prune With CI Score"),
        ("GIES", "Baseline",   "gies"),
        ("GIES", "SafeInsert", "lgies"),
        ("GIES", "ConsInsert", "lgies_prune"),
    ]
    families      = ("GES", "XGES-0", "XGES", "GIES")
    variant_order = ("Baseline", "SafeInsert", "ConsInsert")
    palette = {"Baseline": "tab:blue", "SafeInsert": "tab:orange", "ConsInsert": "mediumpurple"}

    # ---- extract only the needed metrics ----
    required_metrics = ["time", "SHD"]
    missing = [m for m in required_metrics if m not in metrics_data]
    if missing:
        raise ValueError(f"Missing metrics in metrics_data: {missing}")

    def melt_metric(metric_name):
        ci_data = metrics_data[metric_name]

        # decide which p to include
        if nodes is None:
            p_all = set()
            for _, _, key in mapping:
                if key in ci_data:
                    p_all |= set(ci_data[key].keys())
            p_list = sorted(p_all)
        else:
            p_list = list(sorted(nodes))

        rows = []
        for fam, variant, key in mapping:
            if key not in ci_data:
                continue
            for p in p_list:
                vals = ci_data[key].get(p, [])
                for v in vals:
                    rows.append({"Family": fam, "Variant": variant, "p": p, "Value": float(v)})
        df = pd.DataFrame(rows)
        df["Metric"] = metric_name
        return df

    df_time = melt_metric("time")
    df_shd  = melt_metric("SHD")

    # Clean: drop NaNs/infs
    df_time = df_time.replace([np.inf, -np.inf], np.nan).dropna(subset=["Value"])
    df_shd  = df_shd.replace([np.inf, -np.inf], np.nan).dropna(subset=["Value"])

    # Handle log-scale requirements for time
    if drop_nonpositive_time:
        df_time = df_time[df_time["Value"] > 0]
    else:
        # floor tiny/zero to epsilon so they show on log scale
        df_time.loc[df_time["Value"] <= 0, "Value"] = time_eps

    # Combine for iteration
    plots = [("time", df_time, True), ("SHD", df_shd, False)]

    # ---- style ----
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="darkgrid", context="notebook")
    rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "legend.frameon": False,
        "font.size": 13,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "legend.title_fontsize": 14,
    })

    # ---- build the figure with two panels ----
    fig, axes = plt.subplots(ncols=2, figsize=(12, 5.5), sharex=True)
    for ax, (metric_name, df, make_log) in zip(axes, plots):
        if df.empty:
            ax.set_visible(False)
            continue

        title = (
            f"{metric_name} — p={int(df['p'].iloc[0])}"
            if len(set(df["p"])) == 1
            else f"{metric_name} — aggregated over p ∈ {{{', '.join(map(str, sorted(set(df['p']))) )}}}"
        )

        if boxplot:
            sns.boxplot(
                data=df, x="Family", y="Value",
                order=families, hue="Variant", hue_order=variant_order,
                palette=palette, width=0.7, fliersize=1.5, linewidth=1.2, ax=ax
            )
            if include_strip:
                sns.stripplot(
                    data=df, x="Family", y="Value",
                    order=families, hue="Variant", hue_order=variant_order,
                    palette=palette, dodge=True, alpha=0.25, size=3,
                    ax=ax, legend=False
                )
        else:
            # bars: mean ± std (across selected p and trials)
            # g = (df.groupby(["Family", "Variant"], as_index=False)["Value"]
            #     .agg(mean="mean", std="std"))  # <-- flat columns, no MultiIndex issues
            g = (
                    df.groupby(["Family", "Variant"], as_index=False)
                    .agg(mean=("Value", "mean"), std=("Value", "std"))
                )


            # keep hue positions consistent and aligned even if some combos are missing
            families = ("GES", "XGES-0", "XGES", "GIES")
            variant_order = ("Baseline", "SafeInsert", "ConsInsert")
            pos = {fam: i for i, fam in enumerate(families)}
            width = 0.23
            offsets = {"Baseline": -width, "SafeInsert": 0.0, "ConsInsert": +width}

            for variant in variant_order:
                sub = g[g["Variant"] == variant]
                if sub.empty:
                    continue
                xs = [pos[f] + offsets[variant] for f in sub["Family"]]
                ax.bar(xs, sub["mean"], width=width, label=variant, color=palette[variant])
                ax.errorbar(xs, sub["mean"], yerr=sub["std"], fmt="none",
                            elinewidth=2, capsize=3, ecolor=palette[variant])

            ax.set_xticks(range(len(families)))
            ax.set_xticklabels(families)
        if make_log:
            ax.set_yscale("log")

        ax.set_xlabel("")
        ax.set_ylabel(metric_name)
        ax.set_title(title)
        sns.despine(ax=ax)

        # put legend once (right panel)
        handles, labels = ax.get_legend_handles_labels()
        # seaborn may duplicate handles when stacking box+strip; take first 3
        if ax is axes[1]:
            if handles:
                ax.legend(handles[:3], labels[:3], title="Variant",
                          loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        else:
            ax.get_legend().remove() if ax.get_legend() else None

    fig.tight_layout()
    out = os.path.join(output_dir, f"lges_benefit_time_shd_{'box' if boxplot else 'bar'}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_fc_vs_metric_for_fixed_nvars(base_dir="./obs/results_init/0.75m",
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
        "Iterate Without CIs":  "iterate_without_cis",
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
        ("Without CIs", "init")   : "GES-Init (Repeats)",
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
    colour = {"GES-0":"tab:blue","GES-Init":"tab:blue","GES-Init (Repeats)":"tab:green",
              "LGES-0 (Safe)":"tab:orange", "LGES-Init (Safe)":"tab:orange","LGES (Safe)":"darkorange",
              "LGES-0 (Cons)":"tab:purple", "LGES-Init (Cons)":"tab:purple","LGES (Cons)":"mediumpurple",
              "Old LGES-Init (Safe)":"tab:orange","Old LGES (Safe)":"darkorange",
              "Old LGES-Init (Cons)":"tab:purple","Old LGES (Cons)":"mediumpurple"}
    
    ls  = {"GES-0":":", "LGES-0 (Safe)":":", "LGES-0 (Cons)":":",
            "GES-Init":"-", "GES-Init (Repeats)":":", "LGES-Init (Safe)":"-","LGES (Safe)":"--",
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
                            # yerr=es, 
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
        match = re.match(r"vars_(\d+)_samples_10000_degree_3.0_model_linear-gaussian", folder)
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
        "lges_prune": {"linestyle": "--", "color": "tab:purple", "marker": "^"},
        "lges": {"linestyle": "-.", "color": "tab:orange", "marker": "o"},
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
        plt.legend(handles, labels, loc="upper left", title="Method", bbox_to_anchor=(1.02, 1.0))

        plt.savefig(os.path.join(output_dir, f"{metric}.png"), dpi=300, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific plotting functions.")
    parser.add_argument("--plot", type=str, choices=["obs_no_prior", "sns_obs_no_prior", "exp_no_prior", "obs_init"], 
                        required=True, help="Specify which plot function to run.")
    args = parser.parse_args()

    if args.plot == "obs_no_prior":
        plot_obs_no_prior()
    if args.plot == "sns_obs_no_prior":
        # plot_obs_no_prior_seaborn()
        plot_lges_time_shd()
    elif args.plot == "exp_no_prior":
        plot_exp_no_prior()
    elif args.plot == "obs_init":
        plot_fc_vs_metric_for_fixed_nvars()
