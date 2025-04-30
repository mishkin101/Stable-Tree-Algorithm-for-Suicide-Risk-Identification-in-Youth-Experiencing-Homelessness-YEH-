# src/StableTree/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

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

""""""
def plot_common_features(results, dataset, title  =" Common Features for all Trees"):
    # Prepare the rows for each dataset+split
    # Prepare rows
    rows = []
    split_names = ["Root", "Left", "Right"]
    for splits in results:
        for idx, split in enumerate(split_names):
            feats = splits[idx] if idx < len(splits) else []
            f1, p1 = feats[0] if len(feats) > 0 else ("-", 0.0)
            f2, p2 = feats[1] if len(feats) > 1 else ("-", 0.0)
            ds_label = dataset if idx == len(split_names) - 1 else ""
            rows.append([ds_label, split, f1, f"{p1:.2f}", f2, f"{p2:.2f}"])

    # Create DataFrame
    df = pd.DataFrame(rows, columns=[
        "Dataset", "Split",
        "Feature 1", "Frequency (%)",
        "Feature 2", "Frequency (%)"
    ])

    # Plot table
    fig, ax = plt.subplots(figsize=(12, df.shape[0] * 0.4 + 1))
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title(title, pad=20)
    plt.tight_layout()




# \\TODO #23 Make plot to show stf across gini std @mishkin101
def avg_gini_importance_std(std_gini_importance_list):
    return 