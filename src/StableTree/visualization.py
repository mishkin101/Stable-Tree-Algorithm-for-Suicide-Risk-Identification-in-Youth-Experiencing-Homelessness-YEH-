# src/StableTree/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union

def plot_pareto_frontier(distances, auc_scores, pareto_indices):
    """
    Plot the Pareto frontier of trees.
    
    Args:
        distances: List of distance values
        auc_scores: List of AUC scores
        pareto_indices: Indices of Pareto optimal trees
    """
    distances = np.array(distances)
    auc_scores = np.array(auc_scores)
    pareto_indices = set(pareto_indices)
    is_pareto = np.array([i in pareto_indices for i in range(len(distances))])
    #\\TODO #25 make this make a ong for all the auc, distiance, and trade-off pareto optimal trees @adbX
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(distances[~is_pareto], auc_scores[~is_pareto], c='blue', label='Dominated Trees', alpha=0.6)
    plt.scatter(distances[is_pareto], auc_scores[is_pareto], c='red', edgecolors='black', s=80, label='Pareto Optimal Trees')
    plt.xlabel("Stability (Lower is Better)")
    plt.ylabel("AUC (Higher is Better)")
    plt.title("Pareto Frontier of Decision Trees")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Don't call plt.show() so the figure can be saved by the logger


def plot_decision_tree(tree, feature_names, class_names=None, title="Decision Tree", max_depth=None):
    """
    Plot a decision tree.
    
    Args:
        tree: Trained decision tree classifier
        feature_names: List of feature names
        class_names: List of class names
        title: Title for the plot
    """
    plt.figure(figsize=(12, 8))
    plot_tree(tree, 
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            max_depth=max_depth)
    plt.title(title)




def plot_common_features(results, dataset, title="Common Features for all Trees"):
    """
    results :[
                [("age", 40.0), ("income", 20.0)],   # split 0
                [("education", 25.0), ("age", 15.0)], # split 1
                [("gender", 30.0), ("age", 10.0)]     # split 2
             ]
    dataset : str
        Name to show in the “Dataset” column for the last row.
    """
    split_names = ["Root", "Left", "Right"]
    rows = []

    for idx, feats in enumerate(results):
        # get the split name
        split = split_names[idx] if idx < len(split_names) else f"Split{idx}"
        if len(feats) >= 1: 
            f1, p1 = feats[0]
        else:
            f1, p1 = "-", 0.0
        if len(feats) >= 2:
            f2, p2 = feats[1]
        else:
            f2, p2 = "-", 0.0
        ds_label = dataset if idx == len(results) - 1 else ""
        rows.append([
            ds_label,
            split,
            f1,
            f"{p1:.2f}",
            f2,
            f"{p2:.2f}",
        ])

    # Build DataFrame
    df = pd.DataFrame(rows, columns=[
        "Dataset", "Split",
        "Feature 1", "Frequency (%)",
        "Feature 2", "Frequency (%)"
    ])

    # Plot
    fig, ax = plt.subplots(figsize=(12, len(rows)*0.4 + 1))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title(title, pad=20)
    plt.tight_layout()
    #return fig  # or plt.show(), or save with your logger


"==========Aggregation Plotting=========="

def plot_avg_feature_std_from_dict(mean_std_dict, group, output_name="avg_feature_std"):
   
    labels = list(mean_std_dict.keys())
    means = [mean_std_dict[key] for key in mean_std_dict.keys()]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, means)

    # Label each bar with its label
    for bar, lbl in zip(bars, labels):
        bar.set_label(lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Average per-feature std deviation")
    ax.set_title("Feature‐Importance Stability Across Strategies")
    ax.legend(loc="upper right")

    out_path = group.group_path/ f"{output_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_distinct_top_features(distinct_count_dict: dict, group, output_name: str = "distinct_top_features"):
    """
    Plot and save a bar chart showing the number of distinct top-k features
    for each selection strategy in an experiment group.
    """
    labels = ["Stability–Accuracy", "AUC‐maximizing", "Distance‐Minimizing"]

    counts = [len(distinct_count_dict[selection_strategy]) for selection_strategy in distinct_count_dict.keys()]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, counts)

    # Label each bar with its strategy name
    for bar, lbl in zip(bars, labels):
        bar.set_label(lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Count of distinct top-k features")
    ax.set_title("Distinct Top-k Features per Strategy")
    ax.legend(loc="upper right")

    # Save into the experiment group's folder
    out_path = group.group_path / f"{output_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path




def plot_aggregate_metrics(dataset_dict: Dict[str, dict], group, paired_labels, output_prefix: str = "aggregate_metrics"):
    plots_dir = Path(group.group_path)
    # path to the group‐level metrics.json (for tables)
    metrics_json = Path("experiments").resolve() / group.group_name / "metrics.json"

    # unpack the paired_labels into two parallel lists:
    #   - aliases: use the file‐stem of each dataset
    #   - preds:   the predictor variable name
    aliases, preds = zip(*paired_labels)
    # ensure all aliases are Path objects, then use .stem
    aliases = [Path(a).stem for a in aliases]
    aliases = [
        "YEH_Suicide" 
            if stem == "DataSet_Combined_SI_SNI_Baseline_FE" 
            else stem
        for stem in aliases
    ]
    # now build our x‐tick labels as e.g. "YEH_Suicide_suicidea"
    labels = [f"{alias}_{pred}" for alias, pred in zip(aliases, preds)]
    print(f"labels{labels}")

    # datasets are still the keys of dataset_dict, in the same order
    datasets = list(dataset_dict.keys())
    n_ds = len(datasets)
    x    = np.arange(n_ds)
    width = 0.1

    def _save(fig, name):
        out = plots_dir / f"{output_prefix}_{name}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {out}")

    def _save_csv(df, name):
        out = plots_dir / f"{output_prefix}_{name}.csv"
        df.to_csv(out, index=False)
        print(f"Saved CSV: {out}")
        return out

    # 1) Feature‐importance std dev
    strat_keys = list(next(iter(dataset_dict.values()))["feature_std"].keys())
    print(f"strat_keys {strat_keys}")
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, strat in enumerate(strat_keys):
        vals = [dataset_dict[ds]["feature_std"][strat] for ds in datasets]
        ax.bar(x + i*width, vals, width, label=strat)
    ax.set_xticks(x + width*(len(strat_keys)-1)/2)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Mean feature-importance STD")
    ax.set_title("Feature-Importance Stability")
    ax.legend()
    _save(fig, "feature_std")

    # 2) Distinct top-k features count
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, strat in enumerate(strat_keys):
        counts = [len(dataset_dict[ds]["distinct_top_features"][strat]) for ds in datasets]
        ax.bar(x + i*width, counts, width, label=strat)
    ax.set_xticks(x + width*(len(strat_keys)-1)/2)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Count of distinct top-k features")
    ax.set_title("Distinct Top-3 Features")
    ax.legend()
    _save(fig, "distinct_top_features")

    # 3–6) generic mean+std blocks
    def _plot_mean_std_block(key_name, title, ylabel):
        fig, ax = plt.subplots(figsize=(8, 4))
        mean_map = {ds: dataset_dict[ds][key_name]["mean"] for ds in datasets}
        std_map  = {ds: dataset_dict[ds][key_name]["std"]  for ds in datasets}
        strat2   = list(next(iter(mean_map.values())).keys())
        for i, strat in enumerate(strat2):
            mvals = [mean_map[ds][strat] for ds in datasets]
            errs  = [std_map[ds][strat]  for ds in datasets]
            ax.bar(x + i*width, mvals, width, yerr=errs, capsize=3, label=strat)
        ax.set_xticks(x + width*(len(strat2)-1)/2)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        _save(fig, key_name)

    _plot_mean_std_block("tree_nodes",       "Mean Tree Node Counts",    "Node Count")
    _plot_mean_std_block("tree_depth",       "Mean Tree Depths",         "Depth")
    _plot_mean_std_block("optimal_auc",      "Mean Optimal AUC",         "AUC")
    _plot_mean_std_block("optimal_distance", "Mean Optimal Distance",   "Distance")

    # 7) Stability–Trade-Off table (inner helper)
    def _gen_tradeoff_table(alias: str, ds_key: str, predictor: str):
       
        metrics = dataset_dict[ds_key]

        # build rows
        rows = []
        # AUC‐maximizing
        auc_m = metrics["optimal_auc"]["mean"]["selected_auc_tree_auc"]
        auc_s = metrics["optimal_auc"]["std"]["selected_auc_tree_auc"]
        d_m   = metrics["optimal_distance"]["mean"]["selected_auc_tree_distance"]
        d_s   = metrics["optimal_distance"]["std"]["selected_auc_tree_distance"]
        f_s   = metrics["feature_std"]["auc_max_importances"]
        t3    = len(metrics["distinct_top_features"]["auc_max_importances"])
        nd    = metrics["tree_nodes"]["mean"]["auc_tree_nodes"]
        dp    = metrics["tree_depth"]["mean"]["auc_tree_depth"]
        rows.append({
            "Method":             "CART Pareto AUC",
            "AUC":                f"{auc_m:.3f} ({auc_s:.3f})",
            "Distance":           f"{d_m:.3f} ({d_s:.3f})",
            "Feat. Import. Std.": f"{f_s:.3f}",
            "Feat. in Top-3":     str(t3),
            "Nodes":              f"{nd:.3f}",
            "Tree Depth":         f"{dp:.3f}",
        })

        # stability–accuracy extreme
        auc_m2 = metrics["optimal_auc"]["mean"]["selected_tree_auc"]
        auc_s2 = metrics["optimal_auc"]["std"]["selected_tree_auc"]
        d_m2   = metrics["optimal_distance"]["mean"]["selected_tree_distance"]
        d_s2   = metrics["optimal_distance"]["std"]["selected_tree_distance"]
        f_s2   = metrics["feature_std"]["stability_accuracy_importances"]
        t32    = len(metrics["distinct_top_features"]["stability_accuracy_importances"])
        nd2    = metrics["tree_nodes"]["mean"]["stability_tree_nodes"]
        dp2    = metrics["tree_depth"]["mean"]["stability_tree_depth"]
        rows.append({
            "Method":             "CART Pareto Stability",
            "AUC":                f"{auc_m2:.3f} ({auc_s2:.3f})",
            "Distance":           f"{d_m2:.3f} ({d_s2:.3f})",
            "Feat. Import. Std.": f"{f_s2:.3f}",
            "Feat. in Top-3":     str(t32),
            "Nodes":              f"{nd2:.3f}",
            "Tree Depth":         f"{dp2:.3f}",
        })

        # Distance‐minimizing
        auc_m3 = metrics["optimal_auc"]["mean"]["selected_dist_tree_auc"]
        auc_s3 = metrics["optimal_auc"]["std"]["selected_dist_tree_auc"]
        d_m3   = metrics["optimal_distance"]["mean"]["selected_dist_tree_distance"]
        d_s3   = metrics["optimal_distance"]["std"]["selected_dist_tree_distance"]
        f_s3   = metrics["feature_std"]["dist_min_importances"]
        t33    = len(metrics["distinct_top_features"]["dist_min_importances"])
        nd3    = metrics["tree_nodes"]["mean"]["dist_tree_nodes"]
        dp3    = metrics["tree_depth"]["mean"]["dist_tree_depth"]
        rows.append({
            "Method":             "CART Pareto Distance",
            "AUC":                f"{auc_m3:.3f} ({auc_s3:.3f})",
            "Distance":           f"{d_m3:.3f} ({d_s3:.3f})",
            "Feat. Import. Std.": f"{f_s3:.3f}",
            "Feat. in Top-3":     str(t33),
            "Nodes":              f"{nd3:.3f}",
            "Tree Depth":         f"{dp3:.3f}",
        })

        # render & save
        df = pd.DataFrame(rows, columns=[
            "Method","AUC","Distance",
            "Feat. Import. Std.","Feat. in Top-3",
            "Nodes","Tree Depth"
        ])
        # save the raw table as CSV
        _save_csv(df, f"tradeoff_{alias}_{predictor}")
        # save as PNG
        fig, ax = plt.subplots(figsize=(10, 2.5), dpi=300)
        ax.axis("off")
        tbl = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        fig.tight_layout()
        _save(fig, f"tradeoff_{alias}_{predictor}")

    _plot_mean_std_block('tree_nodes',       'Mean Tree Node Counts', 'Node Count')
    _plot_mean_std_block('tree_depth',       'Mean Tree Depths',      'Depth')
    _plot_mean_std_block('optimal_auc',      'Mean Optimal AUC',      'AUC')
    _plot_mean_std_block('optimal_distance', 'Mean Optimal Distance', 'Distance')
    # finally, call it once per dataset:
    for ds_key, alias, pred in zip(datasets, aliases, preds):
        _gen_tradeoff_table(alias, ds_key, pred)



# def generate_stability_trade_off_table(metrics_json_path: Union[str, Path], dataset_name: str, output_path: Union[str, Path]) -> Path:
#     """
#     Load the group‐level metrics.json, extract the aggregated comparison
#     between the AUC‐maximizing and distance‐minimizing Pareto trees for
#     one dataset, and save it as a PNG table matching the LaTeX layout.
#     """
#     metrics_json_path = Path(metrics_json_path)
#     output_path = Path(output_path)
    
#     # 1) Load metrics.json
#     with open(metrics_json_path, "r") as f:
#         all_metrics = json.load(f)
#     ds_metrics = all_metrics["dataset_aggregates"][dataset_name]

#     # 2) Build two rows: CART Pareto AUC and CART Pareto Distance
#     rows = []
#     # AUC‐maximizing row
#     auc_mean = ds_metrics["optimal_auc"]["mean"]["selected_auc_tree_auc"]
#     auc_std  = ds_metrics["optimal_auc"]["std"]["selected_auc_tree_auc"]
#     dist_mean = ds_metrics["optimal_distance"]["mean"]["selected_auc_tree_distance"]
#     dist_std  = ds_metrics["optimal_distance"]["std"]["selected_auc_tree_distance"]
#     feat_std_auc = ds_metrics["feature_std"]["auc_max_importances"]
#     top3_auc = len(ds_metrics["distinct_top_features"]["auc_max_importances"])
#     nodes_auc = ds_metrics["tree_nodes"]["mean"]["auc_tree_nodes"]
#     depth_auc = ds_metrics["tree_depth"]["mean"]["auc_tree_depth"]
#     rows.append({
#         "Method":               "CART Pareto AUC",
#         "AUC":                  f"{auc_mean:.3f} ({auc_std:.3f})",
#         "Distance":             f"{dist_mean:.3f} ({dist_std:.3f})",
#         "Feat. Import. Std.":   f"{feat_std_auc:.3f}",
#         "Feat. in Top-3":       f"{top3_auc}",
#         "Nodes":                f"{nodes_auc:.3f}",
#         "Tree Depth":           f"{depth_auc:.3f}",
#     })

#     # Distance‐minimizing row
#     auc_mean_d = ds_metrics["optimal_auc"]["mean"]["selected_dist_tree_auc"]
#     auc_std_d  = ds_metrics["optimal_auc"]["std"]["selected_dist_tree_auc"]
#     dist_mean_d = ds_metrics["optimal_distance"]["mean"]["selected_dist_tree_distance"]
#     dist_std_d  = ds_metrics["optimal_distance"]["std"]["selected_dist_tree_distance"]
#     feat_std_dist = ds_metrics["feature_std"]["dist_min_importances"]
#     top3_dist = len(ds_metrics["distinct_top_features"]["dist_min_importances"])
#     nodes_dist = ds_metrics["tree_nodes"]["mean"]["dist_tree_nodes"]
#     depth_dist = ds_metrics["tree_depth"]["mean"]["dist_tree_depth"]
#     rows.append({
#         "Method":               "CART Pareto Distance",
#         "AUC":                  f"{auc_mean_d:.3f} ({auc_std_d:.3f})",
#         "Distance":             f"{dist_mean_d:.3f} ({dist_std_d:.3f})",
#         "Feat. Import. Std.":   f"{feat_std_dist:.3f}",
#         "Feat. in Top-3":       f"{top3_dist}",
#         "Nodes":                f"{nodes_dist:.3f}",
#         "Tree Depth":           f"{depth_dist:.3f}",
#     })

#     # stability-Trade-Off Row
#     auc_mean_d = ds_metrics["optimal_auc"]["mean"]["selected_tree_auc"]
#     auc_std_d  = ds_metrics["optimal_auc"]["std"]["selected_tree_auc"]
#     dist_mean_d = ds_metrics["optimal_distance"]["mean"]["selected_tree_distance"]
#     dist_std_d  = ds_metrics["optimal_distance"]["std"]["selected_tree_distance"]
#     feat_std_dist = ds_metrics["feature_std"]["stability_accuracy_importances"]
#     top3_dist = len(ds_metrics["distinct_top_features"]["stability_accuracy_importances"])
#     nodes_dist = ds_metrics["tree_nodes"]["mean"]["stability_tree_nodes"]
#     depth_dist = ds_metrics["tree_depth"]["mean"]["stability_tree_depth"]
#     rows.append({
#         "Method":               "CART Pareto Trade-Off",
#         "AUC":                  f"{auc_mean_d:.3f} ({auc_std_d:.3f})",
#         "Distance":             f"{dist_mean_d:.3f} ({dist_std_d:.3f})",
#         "Feat. Import. Std.":   f"{feat_std_dist:.3f}",
#         "Feat. in Top-3":       f"{top3_dist}",
#         "Nodes":                f"{nodes_dist:.3f}",
#         "Tree Depth":           f"{depth_dist:.3f}",
#     })

#     # 3) Create DataFrame
#     df = pd.DataFrame(rows, columns=[
#         "Method", "AUC", "Distance",
#         "Feat. Import. Std.", "Feat. in Top-3",
#         "Nodes", "Tree Depth"
#     ])

#     # 4) Render as a table and save to PNG
#     fig, ax = plt.subplots(figsize=(10, 2.5), dpi=300)
#     ax.axis("off")
#     table = ax.table(
#         cellText=df.values,
#         colLabels=df.columns,
#         cellLoc="center",
#         loc="center"
#     )
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1, 1.5)
#     fig.tight_layout()
#     _save(fig, f"tradeoff_{label}")

#     return output_path



