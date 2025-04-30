# src/StableTree/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path

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




def plot_aggregate_metrics(dataset_dict: Dict[str, dict], group, output_prefix: str = "aggregate_metrics"):
    plots_dir = Path(group.group_path)
    datasets = list(dataset_dict.keys())
    # strip “.csv” if present
    labels = [ds[:-4] if ds.lower().endswith(".csv") else ds for ds in datasets]

    n_ds = len(datasets)
    x = np.arange(n_ds)
    width = 0.1   # skinnier bars

    def _save(fig, name):
        path = plots_dir / f"{output_prefix}_{name}.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        #print(f"Saved plot: {path}")

    # 1) Feature-importance std dev
    strat_keys = list(next(iter(dataset_dict.values()))['feature_std'].keys())
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, strat in enumerate(strat_keys):
        means = [dataset_dict[ds]['feature_std'][strat] for ds in datasets]
        ax.bar(x + i*width, means, width, label=strat)
    ax.set_xticks(x + width*(len(strat_keys)-1)/2)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_ylabel('Mean feature-importance STD')
    ax.set_title('Feature-Importance Stability')
    ax.legend()
    _save(fig, 'feature_std')

    # 2) Distinct top-k features count
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, strat in enumerate(strat_keys):
        counts = [len(dataset_dict[ds]['distinct_top_features'][strat]) for ds in datasets]
        ax.bar(x + i*width, counts, width, label=strat)
    ax.set_xticks(x + width*(len(strat_keys)-1)/2)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_ylabel('Count of distinct top-k features')
    ax.set_title('Distinct Top-3 Features')
    ax.legend()
    _save(fig, 'distinct_top_features')

    # Generic mean+std block
    def _plot_mean_std_block(key_name, title, ylabel):
        fig, ax = plt.subplots(figsize=(8, 4))
        mean_map = {ds: dataset_dict[ds][key_name]['mean'] for ds in datasets}
        std_map  = {ds: dataset_dict[ds][key_name]['std']  for ds in datasets}
        strat_keys2 = list(next(iter(mean_map.values())).keys())
        for i, strat in enumerate(strat_keys2):
            means = [mean_map[ds][strat] for ds in datasets]
            errs  = [std_map[ds][strat]  for ds in datasets]
            ax.bar(x + i*width, means, width, yerr=errs, capsize=3, label=strat)
        ax.set_xticks(x + width*(len(strat_keys2)-1)/2)
        ax.set_xticklabels(labels, rotation=0, ha='center')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        _save(fig, key_name)

    _plot_mean_std_block('tree_nodes',       'Mean Tree Node Counts', 'Node Count')
    _plot_mean_std_block('tree_depth',       'Mean Tree Depths',      'Depth')
    _plot_mean_std_block('optimal_auc',      'Mean Optimal AUC',      'AUC')
    _plot_mean_std_block('optimal_distance', 'Mean Optimal Distance', 'Distance')
