import numpy as np
from typing import List, Tuple

def pareto_optimal_trees(distances, auc_scores):
    """
    Find Pareto optimal trees based on distances and AUC scores.
    
    Args:
        distances: List of distance values (lower is better)
        auc_scores: List of AUC scores (higher is better)
        
    Returns:
        List of indices of Pareto optimal trees
    """
    pareto_trees = []
    for i, (d_i, a_i) in enumerate(zip(distances, auc_scores)):
        dominated = False
        for j, (d_j, a_j) in enumerate(zip(distances, auc_scores)):
            if i != j and ((d_j <= d_i and a_j > a_i) or (d_j < d_i and a_j >= a_i)):
                dominated = True
                break
        if not dominated:
            pareto_trees.append(i)
    return pareto_trees

def select_final_tree(distances, auc_scores, pareto_indices, epsilon=0.01):
    """
    Select the final tree from Pareto optimal candidates.
    
    Args:
        distances: List of distance values
        auc_scores: List of AUC scores
        pareto_indices: Indices of Pareto optimal trees
        epsilon: Threshold for AUC score margin
        
    Returns:
        Index of the selected tree
    """
    best_auc = max(auc_scores)
    candidates = [i for i in pareto_indices if auc_scores[i] >= (1 - epsilon) * best_auc]
    if not candidates:
        candidates = pareto_indices
    best_idx = max(candidates, key=lambda i: auc_scores[i] - distances[i])
    return best_idx