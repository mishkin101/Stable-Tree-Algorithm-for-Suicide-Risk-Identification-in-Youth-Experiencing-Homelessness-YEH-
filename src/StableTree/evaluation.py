import numpy as np 
from collections import Counter

def common_features(tree_set, n_splits=3, top_k=2):
    """
    Record the two most commonly selected features in each of the
    first three splits, along with their selection frequencies.
    Output example: Frequenicies of top 2 common features: [[(5, 40.0), (6, 20.0)], [(6, 20.0), (0, 15.0)], [(52, 15.0), (5, 10.0)]]
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
  import numpy as np

def feature_importance_std_across_trees(trees):
    all_imps = []
    for tree in trees:
        imps = tree.feature_importances_
        all_imps.append(np.asarray(imps))
    # Stack into shape (n_trees (1 tree = 1 row), n_features)
    all_imps = np.vstack(all_imps)
    # Compute std dev across the first axis (over trees)
    stds = np.std(all_imps, axis=0)
    return stds



