# src/StableTree/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

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

def plot_decision_tree(tree, feature_names, class_names=None, title="Decision Tree"):
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
            filled=True)
    plt.title(title)
    # Don't call plt.show() so the figure can be saved by the logger