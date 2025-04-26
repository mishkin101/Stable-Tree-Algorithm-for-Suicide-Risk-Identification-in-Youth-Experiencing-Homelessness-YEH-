'''
=====Imports=======

=====structure=====
-- all functions need access to pareto trees from main.py

'''
import numpy as np 
import pandas as pd
from collections import Counter

'''we record the two most commonly selected features in each of the
  first three splits, along with their selection frequencies.'''
def common_features(tree_set, n_splits=3, top_k=2):
    """
    Record the two most commonly selected features in each of the
    first three splits, along with their selection frequencies.
    """
    n_trees = len(tree_set)
    split_counters = [Counter() for _ in range(n_splits)]

    # Count which feature is used at split 0, split 1, split 2, …
    for tree in tree_set:
        feat_arr = tree.tree_.feature
        split_features = [f for f in feat_arr if f >= 0]
        for i in range(min(n_splits, len(split_features))):
            split_counters[i][ split_features[i] ] += 1

    results = []
    for counter in split_counters:
        # top_k most common (feat, count) pairs
        most_common = counter.most_common(top_k)
        # convert counts → percentage
        freq_list = [(feat, (count / n_trees) * 100) for feat, count in most_common]
        # sanitize into plain Python types
        sanitized = [(int(feat), float(freq)) for feat, freq in freq_list]
        results.append(sanitized)
    return results



def gini_importance(tree):
    """
    Compute the standard deviation of Gini importances for a single fitted tree.

    Parameters:
    ----------
    tree : object
        A fitted sklearn DecisionTree estimator exposing `feature_importances_`.

    Returns:
    -------
    float
        The standard deviation of the feature_importances_ vector for the tree.
    """
    # Extract the importance vector
    if hasattr(tree, 'feature_importances_'):
        importances = tree.feature_importances_
    elif hasattr(tree, 'feature_importance'):
        importances = tree.feature_importance()
    else:
        raise AttributeError(
            f"Tree object {type(tree).__name__} must have `feature_importances_` or `feature_importance()`"
        )

    # Compute and return the standard deviation across features
    std = np.std(importances)
    return float(std)


