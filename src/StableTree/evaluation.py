import numpy as np 
from collections import Counter

from collections import Counter

def common_features(tree_set, feature_names, n_splits=3, top_k=2):
    """
    Record the two most commonly selected features in each of the
    first three splits, along with their selection frequencies.
    treats any negative feature index as a leaf node and
    labels it "LEAF_NODE".
    """
    n_trees = len(tree_set)
    split_counters = [Counter() for _ in range(n_splits)]

    # Count which feature is used at split 0, split 1, split 2, …
    for tree in tree_set:
        feat_arr = tree.tree_.feature
        # include leaf nodes (which are < 0) if you want to count them;
        # otherwise this stays as is:
        split_features = feat_arr  # no filtering
        for i in range(min(n_splits, len(split_features))):
            split_counters[i][ split_features[i] ] += 1

    results = []
    for counter in split_counters:
        # top_k most common (feat_idx, count) pairs
        most_common = counter.most_common(top_k)
        # convert counts → percentage
        freq_list = [
            (feat_idx, (count / n_trees) * 100)
            for feat_idx, count in most_common
        ]
        # map idx→name, but use "LEAF_NODE" if idx < 0
        named = [
            (
                "LEAF_NODE" if feat_idx < 0 else feature_names[feat_idx],
                float(freq_pct)
            )
            for feat_idx, freq_pct in freq_list
        ]
        results.append(named)

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



