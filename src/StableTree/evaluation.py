import numpy as np 
from collections import Counter
import json
from pathlib import Path

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



"============= Aggregation Metrics ==================="

def compute_avg_feature_std(group):
    keys = ["stability_accuracy_importances","auc_max_importances","dist_min_importances"]
    all_imps = {}
    logs_dir = Path("logs").resolve()
    mean_std_dict = {}
    for key in keys:
        # collect every experiment's importance‐vector for this strategy
        all_vals = []
        for exp in group.experiments:
            mpath = logs_dir / exp / "metrics.json"
            with open(mpath, "r") as f:
                metrics = json.load(f)
            imp_list = metrics.get(key)
            if imp_list is None:
                raise KeyError(f"{key} not found in {mpath}")
            all_vals.append(imp_list)
        
        # stack into array of shape (n_experiments, n_features)
        arr = np.array(all_vals, dtype=float)
        # per‐feature standard deviation across experiments
        per_feat_std = np.std(arr, axis=0)      
        # mean of those stds
        mean_std = float(per_feat_std.mean())   
        mean_std_dict[key] = mean_std

    return mean_std_dict


def count_distinct_top_features(group,top_k: int = 3) -> dict[str, set[str]]:
    """
    For each selection strategy key, count and return the set of unique feature names
    that ever appear in the top_k importances across all experiments in the group.
    """
    keys = ["stability_accuracy_importances","auc_max_importances","dist_min_importances"]
    result: dict[str, set[str]] = {}
    logs_dir = Path("logs").resolve()
    gp = group.group_path
    for key in keys:
        distinct_idxs = set()
        for exp in group.experiments:
            metrics_path = logs_dir/exp/"metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            imp = metrics.get(key)
            if not imp:
                continue
            # find indices of the top_k largest importances
            top_idxs = sorted(
                range(len(imp)),
                key=lambda i: imp[i],
                reverse=True
            )[:top_k]
            distinct_idxs.update(top_idxs)
        # map indices to feature names
        features = {group.feature_names[i] for i in distinct_idxs}
        result[key] = features
        print(f"  {key:20s} top 3 unique features:{features}")

    return result

