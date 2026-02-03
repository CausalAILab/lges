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
from matplotlib.patches import Patch


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
    "Full XGES Prune With CI Score": "LGES+ (Cons)", 
    "Iterate LGES With CI Score": "LGES (Safe, Repeats)",
    "Iterate LGES Prune With CI Score": "LGES (Cons, Repeats)",
    # Interventional data
    "gies": "GIES",
    "lges": "LGES-0 (Safe) + I-Orient",
    "lges_prune": "LGES-0 (Cons) + I-Orient",
    "lgies": "LGIES-0 (Safe)",
    "lgies_prune": "LGIES-0 (Cons)",
    "xlges": "LGES (Safe) + I-Orient",
    "xlges_prune":  "LGES (Cons) + I-Orient",
    "xlgies": "LGIES (Safe)",
    "xlgies_prune": "LGIES (Cons)",
}

# Styles (reuse your mapping for consistency)
styles = {
    "Without CIs":                {"ls": "-",  "color": "tab:blue",     "marker": "o", "label": "GES"},
    "XGES With CI Score":         {"ls": "-", "color": "tab:orange",   "marker": "o", "label": "LGES (Safe)"},
    "XGES Prune With CI Score":   {"ls": "-", "color": "mediumpurple", "marker": "o", "label": "LGES (Cons)"},
    "PC":                         {"ls": "-",  "color": "tab:green",    "marker": "o", "label": "PC"},
    "XGES Without CIs":           {"ls": "-",  "color": "chocolate",     "marker": "o", "label": "XGES-0"},
    "Full XGES Without CIs":      {"ls": "-",  "color": "tab:cyan",    "marker": "o", "label": "XGES"},

    # Interventional data
    "gies":                       {"ls": "-",  "color": "tab:blue",     "marker": "o", "label": "GIES"},
    "lgies":                      {"ls": ":", "color": "tab:orange",   "marker": "o", "label": "LGIES-0 (Safe)"},
    "lgies_prune":                {"ls": ":", "color": "mediumpurple", "marker": "o", "label": "LGIES-0 (Cons)"},
    "lges":                       {"ls": ":",  "color": "tab:orange",     "marker": "^", "label": "LGES-0 (Safe) + I-Orient"},
    "lges_prune":                 {"ls": ":",  "color": "mediumpurple",    "marker": "^", "label": "LGES-0 (Safe) + I-Orient"},
    "xlgies":                      {"ls": "-", "color": "tab:orange",   "marker": "o", "label": "LGIES-0 (Safe)"},
    "xlgies_prune":                {"ls": "-", "color": "mediumpurple", "marker": "o", "label": "LGIES-0 (Cons)"},
    "xlges":                       {"ls": "--",  "color": "tab:orange",     "marker": "o", "label": "LGES-0 (Safe) + I-Orient"},
    "xlges_prune":                 {"ls": "--",  "color": "mediumpurple",    "marker": "o", "label": "LGES-0 (Safe) + I-Orient"},

    # APPENDIX
    "NoTears":                    {"ls": "-", "color": "tab:red", "marker": "o", "label": "NoTears"},
    "LGES With CI Score":         {"ls": ":", "color": "tab:orange", "marker": "o", "label": "LGES-0 (Safe)"},
    "LGES Prune With CI Score":   { "ls": ":", "color": "mediumpurple", "marker": "o", "label": "LGES-0 (Cons)"},
    "Iterate Without CIs":        {"ls": "-.", "color": "tab:blue", "marker": "o", "label": "Without CIs"},
    "Iterate LGES With CI Score": {"ls": "-.", "color": "tab:orange", "marker": "o", "label": "With CI Score"},
    "Iterate LGES Prune With CI Score": {"ls": "-.", "color": "mediumpurple", "marker": "o", "label": "Prune (CI Score)"},
    "Full XGES With CI Score": {"ls": "--", "color": "tab:orange", "marker": "o", "label": "XGES (Safe)"},
    "Full XGES Prune With CI Score": {"ls": "--", "color": "mediumpurple", "marker": "o", "label": "XGES (Cons)"},
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


def plot_obs_no_prior(
    base_dir="./obs/results_no_prior",
    output_dir="./obs/plots_no_prior",
    # xticks=(0, 25, 50, 75, 100, 125, 150),
    xticks=(0, 25, 50, 75, 100),
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
        # "Full XGES Without CIs",
        "PC",
        # Appendix
        # "NoTears",
        # "LGES With CI Score",
        # "LGES Prune With CI Score",
        # "XGES Without CIs",
        # "Full XGES With CI Score",
        # "Full XGES Prune With CI Score",
    }

    for folder in os.listdir(base_dir):
        m = re.match(r"vars_(\d+)_samples_(1000)_degree_2.0_model_linear-gaussian", folder)
        if not m:
            continue
        num_nodes = int(m.group(1))
        if not (10 <= num_nodes <= 150):
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
                    metrics_fp  = os.path.join(tpath, f"old_metrics_{ci_type.replace(' ','_').lower()}_{tnum}.json")
                    estimate_fp = os.path.join(tpath, f"old_estimate_{ci_type.replace(' ','_').lower()}_{tnum}.npy")
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



    # print("\n=== Timeout summary (time == 0 and SHD == 0) ===")
    # for ci_type in ci_types:
    #     # collect across all node counts for that method
    #     times, shds = [], []
    #     for num_nodes in metrics_data["time"][ci_type]:
    #         times.extend(metrics_data["time"][ci_type][num_nodes])
    #         shds.extend(metrics_data["SHD"][ci_type][num_nodes])
    #     times = np.array(times)
    #     shds = np.array(shds)

    #     if len(times) == 0:
    #         continue

    #     timeouts = (times == 0) & (shds == 0)
    #     n_total = len(times)
    #     n_timeouts = np.sum(timeouts)
    #     frac_timeout = n_timeouts / n_total if n_total > 0 else np.nan

    #     valid_times = times[~timeouts]
    #     valid_shds = shds[~timeouts]

    #     mean_time = np.mean(valid_times) if len(valid_times) > 0 else np.nan
    #     std_time  = np.std(valid_times)  if len(valid_times) > 0 else np.nan
    #     mean_shd  = np.mean(valid_shds)  if len(valid_shds) > 0 else np.nan
    #     std_shd   = np.std(valid_shds)   if len(valid_shds) > 0 else np.nan

    #     print(
    #         f"{ci_type:30s}  timeout_frac={frac_timeout:6.2%}  "
    #         f"mean_time={mean_time:9.3f}  std_time={std_time:9.3f}  "
    #         f"mean_shd={mean_shd:9.3f}  std_shd={std_shd:9.3f}  (N={n_total})"
    #     )

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
        "font.size": 16,
        "axes.labelsize": 16,   # axis labels
        "xtick.labelsize": 16,  # x tick labels
        "ytick.labelsize": 16,  # y tick labels
        "legend.fontsize": 16,  # legend text
        "legend.title_fontsize": 16,
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

            if metric.lower() == "time":
                factor = 10.0  # 1 order of magnitude
                lower_bound = np.maximum(y_mean / factor, 1e-12)  # endpoint min
                upper_bound = y_mean * factor                     # endpoint max

                # clamp the endpoints
                lower_endpoint = np.maximum(y_mean - y_std, lower_bound)
                upper_endpoint = np.minimum(y_mean + y_std, upper_bound)

                # convert endpoints -> asymmetric error deltas
                lower_err = y_mean - lower_endpoint
                upper_err = upper_endpoint - y_mean
                y_err = np.vstack([lower_err, upper_err])  # shape (2, N)
            else:
                y_err = y_std

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
                x_vals, y_mean, yerr=y_err,
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

        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])

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
            # handles.append(Patch(facecolor=color, edgecolor="black", label=lab))  # <— solid block
            handles.append(Line2D(
                [0], [0],
                color=color,
                linestyle=ls,
                linewidth=2,
                marker=marker,
                markeredgewidth=1.1,
                markersize=6,
                label=lab
            ))
            labels.append(lab)

        if handles:
            ax.legend(
                handles, labels, title="Method",
                loc="upper left", bbox_to_anchor=(1.02, 1.0),
                fontsize=12, title_fontsize=12, borderaxespad=0.0
            )

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{metric}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

def plot_lges_time_shd(
    base_dirs=("./obs/results_no_prior", "./exp/results_no_prior"),
    output_dir="./obs/plots_no_prior",
    nodes=(150,),
    drop_nonpositive_time=True,
    time_eps=1e-9,
):
    """
    Plot time (log-y) and SHD for GES, XGES, GIES families.
    Each family shows two clusters: degree 2.0 (solid) and degree 3.0 (hatched).
    Inside each cluster, three variant bars: Baseline / SafeInsert / ConsInsert (colored).
    """

    import os, re, json
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.patches import Patch

    # ---------- INGEST ----------
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

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
        "lgies_prune",
    }

    folder_re = re.compile(r"vars_(\d+)_samples_(10000)_degree_(2\.0|3\.0)_model_linear-gaussian")
    nodes_set = set(nodes) if nodes else None

    for base_dir in base_dirs:
        if not os.path.isdir(base_dir):
            continue
        for folder in os.listdir(base_dir):
            m = folder_re.match(folder)
            if not m:
                continue
            num_nodes = int(m.group(1))
            degree = float(m.group(3))
            if nodes_set and num_nodes not in nodes_set:
                continue

            folder_path = os.path.join(base_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            for trial in os.listdir(folder_path):
                tm = re.match(r"trial_(\d+)", trial)
                if not tm:
                    continue
                tnum = tm.group(1)
                tpath = os.path.join(folder_path, trial)
                if not os.path.isdir(tpath):
                    continue

                dag_file = os.path.join(tpath, f"dag_{tnum}.npy")
                data_file = os.path.join(tpath, f"data_{tnum}.npy")
                if not (os.path.exists(dag_file) and os.path.exists(data_file)):
                    continue

                dag = np.load(dag_file, allow_pickle=False)
                for ci_type in ci_types:
                    key = ci_type.replace(" ", "_").lower()
                    metrics_fp = os.path.join(tpath, f"metrics_{key}_{tnum}.json")
                    estimate_fp = os.path.join(tpath, f"estimate_{key}_{tnum}.npy")
                    if not (os.path.exists(metrics_fp) and os.path.exists(estimate_fp)):
                        continue

                    estimate = np.load(estimate_fp, allow_pickle=True)
                    shd, fp, fn, wo = cpdag_shd(utils.dag_to_cpdag(dag), estimate)
                    metrics_data["SHD"][ci_type][degree][num_nodes].append(shd)

                    with open(metrics_fp, "r") as f:
                        other = json.load(f)
                    for met, val in other.items():
                        metrics_data[met][ci_type][degree][num_nodes].append(val)

    # ---------- PLOTTING ----------
    os.makedirs(output_dir, exist_ok=True)

    mapping = [
        ("GES", "Baseline", "Without CIs"),
        ("GES", "Safe", "LGES With CI Score"),
        ("GES", "Cons", "LGES Prune With CI Score"),
        ("XGES-0", "Baseline", "XGES Without CIs"),
        ("XGES-0", "Safe", "XGES With CI Score"),
        ("XGES-0", "Cons", "XGES Prune With CI Score"),
        ("XGES", "Baseline", "Full XGES Without CIs"),
        ("XGES", "Safe", "Full XGES With CI Score"),
        ("XGES", "Cons", "Full XGES Prune With CI Score"),
        ("GIES", "Baseline", "gies"),
        ("GIES", "Safe", "lgies"),
        ("GIES", "Cons", "lgies_prune"),
    ]

    families = ("GES", "XGES-0", "XGES", "GIES")
    variant_order = ("Baseline", "Safe", "Cons")
    degree_order = (2.0, 3.0)

    palette = {"Baseline": "tab:blue", "Safe": "tab:orange", "Cons": "mediumpurple"}
    hatch_map = {2.0: "", 3.0: "///"}

    def melt_metric(metric_name):
        ci_data = metrics_data[metric_name]
        rows = []
        for fam, variant, key in mapping:
            if key not in ci_data:
                continue
            deg_dict = ci_data[key]
            for deg in degree_order:
                if deg not in deg_dict:
                    continue
                for p in nodes or deg_dict[deg].keys():
                    vals = deg_dict[deg].get(p, [])
                    for v in vals:
                        rows.append(
                            {
                                "Family": fam,
                                "Variant": variant,
                                "Degree": deg,
                                "p": p,
                                "Value": float(v),
                            }
                        )
        return pd.DataFrame(rows)

    df_time = melt_metric("time")
    df_shd = melt_metric("SHD")

    for df in (df_time, df_shd):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["Value"], inplace=True)
    if not df_time.empty and drop_nonpositive_time:
        df_time = df_time[df_time["Value"] > 0]

    plots = [("time", df_time, True), ("SHD", df_shd, False)]

    sns.set_theme(style="darkgrid", context="notebook")
    rcParams.update(
        {
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
        }
    )

    fig, axes = plt.subplots(ncols=2, figsize=(13.5, 5.8), sharex=True)
    for ax, (metric_name, df, make_log) in zip(axes, plots):
        if df.empty:
            ax.set_xticks(range(len(families)))
            ax.set_xticklabels(families)
            ax.set_xlabel("")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} — no data")
            sns.despine(ax=ax)
            continue

        g = (
            df.groupby(["Family", "Variant", "Degree"], observed=True, as_index=False)
            .agg(mean=("Value", "mean"), std=("Value", "std"))
        )

        fam_pos = {fam: i for i, fam in enumerate(families)}
        cluster_gap = 0.22
        bar_width = 0.12
        deg_off = {2.0: -cluster_gap, 3.0: +cluster_gap}
        var_off = {"Baseline": -bar_width, "Safe": 0.0, "Cons": +bar_width}

        # draw bars (degree outer, variant inner)
        for deg in degree_order:
            sub_deg = g[g["Degree"] == deg]
            if sub_deg.empty:
                continue
            for variant in variant_order:
                sub = sub_deg[sub_deg["Variant"] == variant]
                if sub.empty:
                    continue
                xs = [fam_pos[f] + deg_off[deg] + var_off[variant] for f in sub["Family"]]
                ax.bar(
                    xs,
                    sub["mean"],
                    width=bar_width,
                    color=palette[variant],
                    edgecolor="black",
                    linewidth=0.7,
                    hatch=hatch_map[deg],
                )
                ax.errorbar(
                    xs,
                    sub["mean"],
                    yerr=sub["std"],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=3,
                )

        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families)
        if make_log:
            ax.set_yscale("log")
        ax.set_xlabel("")
        ax.set_ylabel(metric_name)
        title = (
            f"{metric_name} — p={int(df['p'].iloc[0])}"
            if df["p"].nunique() == 1
            else f"{metric_name} — aggregated over p ∈ {{{', '.join(map(str, sorted(df['p'].unique())))}}}"
        )
        ax.set_title(title)
        sns.despine(ax=ax)

    # ---- Legends ----
    # color legend (variant)
    variant_handles = [
        Patch(facecolor=palette[v], edgecolor="black", label=v) for v in variant_order
    ]
    # pattern legend (degree)
    degree_handles = [
        Patch(
            facecolor="lightgray",
            edgecolor="black",
            hatch=hatch_map[d],
            label=f"ER{int(d)}",
        )
        for d in degree_order
    ]

    legend1 = axes[1].legend(
        handles=variant_handles,
        title="Variant",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    axes[1].add_artist(legend1)
    legend2 = axes[1].legend(
            handles=degree_handles,
            title="Degree",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            borderaxespad=0.0,
        )
    
    axes[1].set_xlabel("Family")
    axes[1].set_xlabel("Family")


    fig.tight_layout()
    out = os.path.join(output_dir, "lges_benefit_time_shd_deg2_vs_deg3.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", bbox_extra_artists=(legend1, legend2),   # <-- ensures legends are included
    pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {out}")

def plot_fc_vs_metric_for_fixed_nvars(base_dir="./obs/results_init/0.5m",
                                      output_dir="./obs/plots_init",
                                      fixed_nvars=50, max_std=20,
                                      show_errorbars=True):
    """
    One figure per metric (SHD, time)
    x-axis  : fraction-correct (fc)
    y-axis  : metric value
    Only data for `fixed_nvars` (#variables) are used.

    Seaborn-styled. Styling rules:
      - Line style encodes knowledge: dotted (no_init), dashed (init), solid (req=priority insertion).
      - Color encodes family: blue (GES), orange (LGES Safe), purple (LGES Cons).
      - Two legends on the right: colors (family) and line styles (knowledge).
    """
    import os, re, json
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from collections import defaultdict

    # ----------------- Load -----------------
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ci_tags = {
        "Without CIs": "without_cis",
        # "LGES With CI Score": "lges_with_ci_score",
        # "LGES Prune With CI Score": "lges_prune_with_ci_score",
        # "Iterate Without CIs":  "iterate_without_cis",
        "XGES With CI Score": "lges_with_ci_score",
        "XGES Prune With CI Score": "lges_prune_with_ci_score",
    }
    init_types = ["init", "req", "no_init"]   # req = priority insertion; init = initialization; no_init = no knowledge
    fc_values_found = set()

    # Seaborn theme to match your other plots
    sns.set_theme(style="darkgrid", context="notebook")
    rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "legend.frameon": False,
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
    })

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

    # ----------------- Styling helpers -----------------
    # Map method key -> human-friendly label (still used for debugging if needed)
    style_map = {
        ("Without CIs", "no_init"): "GES (no knowledge)",
        ("Without CIs", "init")   : "GES-Init",
        ("Without CIs", "req")    : "GES-Init (Repeats)",
        ("With CI Score", "init") : "Old LGES-Init (Safe)",
        ("With CI Score", "req")  : "Old LGES (Safe)",
        ("Prune With CI Score", "init"): "Old LGES-Init (Cons)",
        ("Prune With CI Score", "req") : "Old LGES (Cons)",
        ("LGES With CI Score", "no_init") : "LGES-0 (Safe, no knowledge)",
        ("LGES With CI Score", "init")    : "LGES-0-Init (Safe)",
        ("LGES With CI Score", "req")     : "LGES-0 (Safe)",
        ("XGES With CI Score", "no_init") : "LGES (Safe, no knowledge)",
        ("XGES With CI Score", "init")    : "LGES-Init (Safe)",
        ("XGES With CI Score", "req")     : "LGES (Safe)",
        ("LGES Prune With CI Score", "no_init"): "LGES-0 (Cons, no knowledge)",
        ("LGES Prune With CI Score", "init")   : "LGES-0-Init (Cons)",
        ("LGES Prune With CI Score", "req")    : "LGES-0 (Cons)",
        ("XGES Prune With CI Score", "no_init"): "LGES (Cons, no knowledge)",
        ("XGES Prune With CI Score", "init")   : "LGES-Init (Cons)",
        ("XGES Prune With CI Score", "req")    : "LGES (Cons)",
    }

    # Color by "family"
    def color_for(ci_type: str, label: str) -> str:
        # GES family
        if ci_type in {"Without CIs", "Iterate Without CIs"} or label.startswith("GES"):
            return "tab:blue"
        # LGES Safe
        if ("With CI Score" in ci_type and "Prune" not in ci_type) or ("Safe" in label):
            return "tab:orange"
        # LGES Cons
        if ("Prune" in ci_type) or ("Cons" in label):
            return "mediumpurple"
        # fallback
        return "tab:gray"

    # Line style by knowledge type
    linestyle_for = {"no_init": ":", "init": "--", "req": "-"}

    # Markers (kept small/consistent)
    marker_for = {"no_init": "o", "init": "o", "req": "o"}

    os.makedirs(output_dir, exist_ok=True)

    # ----------------- Plot -----------------
    for metric in ["SHD", "time"]:
        fig, ax = plt.subplots(figsize=(8.8, 6.2))

        for method_key, fc_dict in metrics_data[metric].items():
            ci_type, init_type = method_key
            label = style_map.get(method_key, None)
            if label is None:
                # skip unknown combos silently
                continue

            color = color_for(ci_type, label)
            ls    = linestyle_for.get(init_type, "-")
            mk    = marker_for.get(init_type, "o")

            # no-init horizontal line (fc=None)
            if None in fc_dict and len(fc_dict[None]) > 0:
                mean_val = float(np.mean(fc_dict[None]))
                ax.axhline(mean_val, color=color, linestyle=ls, linewidth=2)
                continue

            # line with (optional) error bars over fc
            xs, ys, es = [], [], []
            for fc in fc_values:
                vals = fc_dict.get(fc, [])
                if vals:
                    xs.append(fc)
                    ys.append(float(np.mean(vals)))
                    es.append(min(float(np.std(vals)), max_std))

            if xs:
                # seaborn-styled line
                sns.lineplot(x=xs, y=ys, ax=ax,
                             color=color, linestyle=ls,
                             marker=mk, dashes=False, linewidth=2,
                             markeredgewidth=1.1, markersize=5)
                # optional error bars
                if show_errorbars and any(e > 0 for e in es):
                    ax.errorbar(xs, ys, yerr=es, fmt="none",
                                ecolor=color, elinewidth=1.2, capsize=3)

        ax.set_xlabel("Fraction correct (fc)")
        ax.set_ylabel("Time (s)" if metric == "time" else "Structural Hamming Distance")
        if metric == "time":
            ax.set_yscale("log")

        # --- Set custom x-axis ticks ---
        ax.set_xlim(-0.03, 1.03)
        # ax.margins(x=0.2)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])

        sns.despine(ax=ax)
        ax.grid(True, linestyle="--", alpha=0.35)

        # ---- Legends: color (family) + linestyle (knowledge) ----
        color_handles = [
            Patch(facecolor="tab:blue", edgecolor="black", label="GES"),
            Patch(facecolor="tab:orange", edgecolor="black", label="LGES (Safe)"),
            Patch(facecolor="mediumpurple", edgecolor="black", label="LGES (Cons)"),
        ]
        style_handles = [
            Line2D([0], [0], color="black", lw=2, linestyle=":",  label="No knowledge"),
            Line2D([0], [0], color="black", lw=2, linestyle="--", label="Init"),
            Line2D([0], [0], color="black", lw=2, linestyle="-",  label="Priority"),
        ]

        leg1 = ax.legend(handles=color_handles, title="Family",
                         loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        ax.add_artist(leg1)
        ax.legend(handles=style_handles, title="Knowledge",
                  loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{metric}_vs_fc_nvars_{fixed_nvars}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_exp_no_prior(
    base_dir="./exp/results_no_prior",
    output_dir="./exp/plots_no_prior",
    xticks=(0, 25, 50, 75, 100, 125, 150),
    square=True,
):
    # ---------- INGEST ----------
    # metrics_data[metric][ci_type][num_nodes] = list of values
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    ci_types = {
        # main paper
        "gies",
        "xlgies",
        "xlgies_prune",
        "xlges",
        "xlges_prune"
        # appendix 
        # "lgies",
        # "lgies_prune",
        # "lges",
        # "lges_prune",
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
        base = ["gies", "lges", "xlges", "lgies", "xlgies", "lges_prune","xlges_prune", "lgies_prune", "xlgies_prune"]
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
            if metric.lower() == "time":
                factor = 10.0  # 1 order of magnitude
                lower_bound = np.maximum(y_mean / factor, 1e-12)  # endpoint min
                upper_bound = y_mean * factor                     # endpoint max

                # clamp the endpoints
                lower_endpoint = np.maximum(y_mean - y_std, lower_bound)
                upper_endpoint = np.minimum(y_mean + y_std, upper_bound)

                # convert endpoints -> asymmetric error deltas
                lower_err = y_mean - lower_endpoint
                upper_err = upper_endpoint - y_mean
                y_err = np.vstack([lower_err, upper_err])  # shape (2, N)
            else:
                y_err = y_std
            # y_err = y_std
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
                x_vals, y_mean, yerr=y_err,
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

        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific plotting functions.")
    parser.add_argument("--plot", type=str, choices=["obs_no_prior", "exp_no_prior", "obs_init"], 
                        required=True, help="Specify which plot function to run.")
    args = parser.parse_args()

    if args.plot == "obs_no_prior":
        plot_obs_no_prior()
        # plot_lges_time_shd()
    elif args.plot == "exp_no_prior":
        plot_exp_no_prior()
    elif args.plot == "obs_init":
        plot_fc_vs_metric_for_fixed_nvars()
