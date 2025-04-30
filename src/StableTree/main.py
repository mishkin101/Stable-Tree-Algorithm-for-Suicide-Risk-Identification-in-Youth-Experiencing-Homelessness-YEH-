# src/StableTree/main.py
from constants import (
    FEATURE_SETS,
    DEPTHS,
    MIN_SAMPLES,
    NUM_BOOTSTRAPS,
    RANDOM_SEED,
    DATA_DIR

)
from data import prepare_data, random_train_split
from models import bootstrap_trees, evaluate_predictive_power
from distance import compute_average_distances
from pareto import pareto_optimal_trees, select_final_tree, select_auc_maximizing_tree, select_distance_minimizing_tree
from visualization import plot_pareto_frontier, plot_decision_tree, \
plot_common_features, plot_aggregate_metrics
from logging_utils import ExperimentLogger
from evaluation import common_features, compute_avg_feature_std, count_distinct_top_features, \
aggregate_tree_depth, aggregate_tree_nodes , aggregate_optimal_auc, aggregate_optimal_distance
from sklearn.preprocessing import label_binarize
import visualize_like_orig as vis_orig
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Turn the warning into an exception
warnings.filterwarnings('error', category=UndefinedMetricWarning)

import numpy as np
import pandas as pd
import sys
import argparse
from datetime import datetime
from pathlib import Path
import json
import os
from shutil   import rmtree

# Add required paths to system path if needed
src_path = Path("src/dt-distance").resolve()
data_path = Path("data").resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
sys.path.append(str(data_path))


class ExperimentGroup:
    """Manages experiments across multiple datasets and seeds."""

    def __init__(self, group_name: str, data_paths: list[str]):
        """Initialize an experiment group with a unique name."""
        if group_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            group_name = f"experiment_group_{timestamp}"
        
        self.group_name = group_name
        self.group_path = Path(f"experiments/{group_name}")
        self.group_path.mkdir(parents=True, exist_ok=True)

        # # Store dataset path and feature names
        # self.data_path = Path(data_path) if data_path is not None else None
        # self.feature_names: list[str] = []

        # store list of dataset path
        self.data_paths = [Path(p) for p in data_paths]
        # feature names per dataset
        self.feature_names: dict[str, list[str]] = {}
  

       # Create a metadata file for the group
        self.metadata_path = self.group_path / "group_metadata.json"
        self.experiments = []
        
        # Initialize with empty metadata
        self._save_metadata()
    
    def add_experiment(self, experiment_name):
        """Add an experiment to this group."""
        self.experiments.append(experiment_name)
        self._save_metadata()
        return experiment_name

    def set_feature_names(self, dataset: Path, feature_list: list[str]):
        """Store feature names used across this experiment group."""
        key = dataset.name
        self.feature_names[key] = feature_list
        self._save_metadata()

    def _save_metadata(self):
        """Save metadata about this experiment group."""
        metadata = {
            "group_name": self.group_name,
            "created_at": datetime.now().isoformat(),
            "data_paths": [str(p) for p in self.data_paths],
            "feature_names": self.feature_names,
            "experiments": self.experiments,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def get_summary(self):
        """Get a summary of all experiments in this group."""
        summary = {"experiments": []}
        
        for exp_name in self.experiments:
            exp_path = Path(f"experiments/{exp_name}")
            metrics_path = exp_path / "metrics.json"
            
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                summary["experiments"].append({"name": exp_name, "metrics": metrics})
        return summary



def run_experiment(seed: int, label: str, dataset: Path, experiment_group: ExperimentGroup):
    """Run a single experiment with the specified random seed."""
    # Create a unique experiment name based on timestamp and seed
    rng = np.random.default_rng(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}_seed_{seed}_{dataset.name}"
    dataset_name = dataset.stem
    
    # Add this experiment to the group if provided
    if experiment_group:
        experiment_group.add_experiment(experiment_name)
    
    # Initialize the logger
    logger = ExperimentLogger(experiment_name)
    
    # Log configuration
    logger.log_config({
        "DEPTHS": DEPTHS,
        "MIN_SAMPLES": MIN_SAMPLES,
        "NUM_BOOTSTRAPS": NUM_BOOTSTRAPS,
        "RANDOM_SEED": seed,
        "DATA_PATH": str(dataset),
    })
    
    # Set random seed for reproducibility
    # np.random.seed(seed)

    # Use dataset path from group
    DATA_PATH = dataset if dataset else None
    if DATA_PATH is None:
        raise ValueError("DATA_PATH must be provided via --data-path when creating ExperimentGroup")
    logger.log_config({"DATA_PATH": str(DATA_PATH)})
    # Load and prepare data
    df = pd.read_csv(DATA_PATH)
    
    # Log the label being used
    logger.log_config({"label": label})
    
    X_full, y_full, X_train, X_test, y_train, y_test = prepare_data(
        df, FEATURE_SETS[label], label, rng
    )

    # Store feature names in group metadata (once)
    ds_name = dataset.name   # or ds.name if that's the variable you used
    if experiment_group and ds_name not in experiment_group.feature_names:
        experiment_group.set_feature_names(dataset, X_full.columns.tolist())

    # Log dataset metrics
    dataset_metrics = {
        "num_samples_full": len(X_full),
        "num_samples_train": len(X_train),
        "num_samples_test": len(X_test),
        "train_shape": X_train.shape,
    }
    logger.log_metrics(dataset_metrics)
    
    print(f"Experiment: {experiment_name} - Seed: {seed} - Dataset: {dataset_name}")
    print(f"Number of samples in the full dataset: {len(X_full)}")
    print(f"Number of samples in the training set: {len(X_train)}")
    print(f"Number of samples in the test set: {len(X_test)}")
    print(f"Shape of training set: {X_train.shape}")

    # Create random split for baseline trees
    X0, y0 = random_train_split(X_train.values, y_train.values)
    logger.log_metrics({"random_split_shape": (X0.shape[0], y0.shape[0])})
    print(f"Shape of random split: {X0.shape}, {y0.shape}")

    # Generate bootstrap trees
    T0 = bootstrap_trees(X0, y0, DEPTHS, MIN_SAMPLES, NUM_BOOTSTRAPS)
    logger.log_metrics({"num_trees_T0": len(T0)})
    print(f"Number of trees in T0: {len(T0)}")

    T = bootstrap_trees(
        X_train.values, y_train.values, DEPTHS, MIN_SAMPLES, NUM_BOOTSTRAPS
    )
    
    logger.log_metrics({"num_trees_T": len(T)})
    print(f"Number of trees in T: {len(T)}")

    # Compute distances between tree sets
    distances = compute_average_distances(T0, T, X_train, y_train)
    logger.log_metrics({"num_distances": len(distances)})
    print(f"Number of distances computed: {len(distances)}")

    # Evaluate predictive performance
    auc_scores = evaluate_predictive_power(T, X_test.values, y_test.values)
    logger.log_metrics({"average_auc": float(np.mean(auc_scores))})
    print(f"Average AUC score: {np.mean(auc_scores)}")

    # Find Pareto optimal trees
    pareto_trees = pareto_optimal_trees(distances, auc_scores)
    logger.log_metrics({"num_pareto_trees": len(pareto_trees)})
    print(f"Number of Pareto optimal trees: {len(pareto_trees)}")
 
    # Find the most common features
    features = X_full.columns.tolist()
    common_feat_list = common_features(T, feature_names = features)
    logger.log_metrics({"common_feature_freq": common_feat_list})
    print(f"Frequenicies of top 2 common features: {common_feat_list}")

    # Select the final tree
    selected_tree_index = select_final_tree(distances, auc_scores, pareto_trees)
    logger.log_metrics({
        "selected_tree_index": int(selected_tree_index),
        "selected_tree_distance": float(distances[selected_tree_index]),
        "selected_tree_auc": float(auc_scores[selected_tree_index])
    })
    print(f"Selected stability-accuracy trade-off final tree index: {selected_tree_index}")
    selected_tree = T[selected_tree_index]

    # --- compute and log depth & node‐count ---
    stab_depth = selected_tree.tree_.max_depth
    stab_nodes = selected_tree.tree_.node_count
    logger.log_metrics({
        "stability_tree_depth": int(stab_depth),
        "stability_tree_nodes": int(stab_nodes)
    })
    print(f"Stability-accuracy tree depth: {stab_depth}, nodes: {stab_nodes}")

    # log & print its feature importances
    stab_feat_imp = selected_tree.feature_importances_
    logger.log_metrics({
        "stability_accuracy_importances": stab_feat_imp.tolist()
    })
    # print("Stability-accuracy trade-off tree feature importances:", stab_feat_imp)

    # select final auc tree
    selected_auc_tree_index = select_auc_maximizing_tree(auc_scores, pareto_trees)
    logger.log_metrics({
        "selected_auc_tree_index": int(selected_auc_tree_index),
        "selected_auc_tree_distance": float(distances[selected_auc_tree_index]),
        "selected_auc_tree_auc": float(auc_scores[selected_auc_tree_index])
    })
    print(f"Selected AUC maximizing tree index: {selected_auc_tree_index}")
    selected_auc_tree = T[selected_auc_tree_index]

    # compute and log depth & node‐count ---
    auc_depth = selected_auc_tree.tree_.max_depth
    auc_nodes = selected_auc_tree.tree_.node_count
    logger.log_metrics({
        "auc_tree_depth": int(auc_depth),
        "auc_tree_nodes": int(auc_nodes)
    })
    print(f"AUC-maximizing tree depth: {auc_depth}, nodes: {auc_nodes}")

    # log & print its feature importances
    auc_feat_imp = selected_auc_tree.feature_importances_
    logger.log_metrics({
        "auc_max_importances": auc_feat_imp.tolist()
    })
    # print("AUC-maximizing tree feature importances:", auc_feat_imp)

    #select final distance tree
    selected_dist_tree_index = select_distance_minimizing_tree(distances, pareto_trees)
    logger.log_metrics({
        "selected_dist_tree_index": int(selected_dist_tree_index),
        "selected_dist_tree_distance": float(distances[selected_dist_tree_index]),
        "selected_dist_tree_auc": float(auc_scores[selected_dist_tree_index])
    })
    print(f"Selected distance minimizing tree index: {selected_dist_tree_index}")
    selected_dist_tree = T[selected_dist_tree_index]

    # compute and log depth & node‐count ---
    dist_depth = selected_dist_tree.tree_.max_depth
    dist_nodes = selected_dist_tree.tree_.node_count
    logger.log_metrics({
        "dist_tree_depth": int(dist_depth),
        "dist_tree_nodes": int(dist_nodes)
    })
    print(f"Distance-minimizing tree depth: {dist_depth}, nodes: {dist_nodes}")

    # log & print its feature importances
    dist_feat_imp = selected_dist_tree.feature_importances_
    logger.log_metrics({
        "dist_min_importances": dist_feat_imp.tolist()
    })
    # print("Distance-minimizing tree feature importances:", dist_feat_imp)



    
    """======= Original Code Metrics======="""
    vis_orig.visualize_tree(selected_tree, X_full.columns, label=label, logger = logger, fig_name="original_decision_tree")
    vis_orig.save_feature_importance(selected_tree, X_full.columns.tolist(), label=label, logger = logger, fig_name="original_feature_importance")
    
    # Predict labels and probabilities
    y_pred_prob = selected_tree.predict_proba(X_test.values)
    y_pred = selected_tree.predict(X_test.values)
    y_test_bin = label_binarize(y_test.values, classes=np.unique(y_test))
    
    vis_orig.plot_roc_curve(
    y_test_bin,
    y_pred_prob,
    label=label,
    logger = logger,
    fig_name ="original_roc_curve"
    )

    vis_orig.write_metrics(
    FEATURE_SETS[label],
    label,
    y_test=y_test,
    y_pred=y_pred,
    y_pred_prob=y_pred_prob,
    y_test_bin=y_test_bin,
    logger = logger
    )
    
    # #find the gini importance
    # std_gini_importance = gini_importance(selected_tree)
    # logger.log_metrics({"std_gini_importance": float(std_gini_importance)})
    # print(f"Standard deviation of Gini Importance: {std_gini_importance}")

    # Visualize results
    plot_decision_tree(
        selected_tree,
        feature_names=X_full.columns,
        class_names=["No", "Yes"],
        title=f"Pareto Optimal Tree (Seed {seed})",
    )
    logger.save_figure("decision_tree")
    
    # Save trimmed tree (max depth = 2)
    plot_decision_tree(
        selected_tree,
        feature_names=X_full.columns,
        class_names=["No", "Yes"],
        title=f"Trimmed Tree (Seed {seed})",
        max_depth=1
    )
    logger.save_figure("trimmed_decision_tree")

    #save common_features for each experiment
    # dataset_name = os.path.splitext(os.path.basename(DATA_PATH))[0]
    plot_common_features(common_feat_list, dataset=dataset_name)
    logger.save_figure("top_common_features")


    plot_pareto_frontier(distances, auc_scores, pareto_trees)
    logger.save_figure("pareto_frontier")
    
    return experiment_name


def _make_jsonable(o):
    if isinstance(o, dict):
        return {k: _make_jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_make_jsonable(v) for v in o]
    if isinstance(o, set):
        return sorted(list(o))
    return o


def main():
  
    ''''add more datasets here as needed'''

    data_path_dict = {  1: DATA_DIR / "DataSet_Combined_SI_SNI_Baseline_FE.csv",
                        2: DATA_DIR / "breast_cancer.csv"
    }

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Run StableTree experiments with multiple seeds')
    parser.add_argument('--seeds', type=int, nargs='+', default=[RANDOM_SEED], 
                        help='List of random seeds to use for experiments')
    parser.add_argument('--labels', nargs='+', required=True,
                        help='One label per dataset (must match FEATURE_SETS keys)')
    parser.add_argument('--group-name', type=str, default= "suicide_test",
                        help='Name for the experiment group')
    parser.add_argument('--datasets', nargs='+', default = data_path_dict[1],
                        help='Path to the dataset CSVs')
    args = parser.parse_args()
    if len(args.labels) != len(args.datasets):
        parser.error(f"--labels ({len(args.labels)}) must match --datasets ({len(args.datasets)})")
    group = ExperimentGroup(args.group_name, args.datasets)
    group_logger = ExperimentLogger(group_name= args.group_name)
    print(f"Created experiment group: {group.group_name}")

    dataset_dict= {}
    for ds, label in zip(group.data_paths, args.labels):
        try:
            # Run experiments for each seed
            for seed in args.seeds:
                print(f"\n{'=' * 50}")
                print(f"Running for dataset {ds.stem} with seed {seed}")
                print(f"\n{'=' * 50}")
                experiment_name = run_experiment(seed, label, ds, group)
                print(f"Completed experiment: {experiment_name}")

            '''======Aggregate Statistics========='''
            # Compute the average standard deviation of Gini Importance across multiple experiments for every pareto selection strategy
            print("\nAggregate feature‐importance stability across experiments:")
            mean_std_feature_dict = compute_avg_feature_std(group, ds.name)
            for key, mean_std in mean_std_feature_dict.items():
                print(f"  {key:20s} mean(std) = {mean_std:.5f}")
            # Compute the top 3 distinct features in all experiments for every pareto selection strategy
            distinct_feats_dict = count_distinct_top_features(group, ds.name)

            # Compute the mean and std of aggregated tree nodes across multiple experiments for every pareto selection strategy
            mean_nodes_dict, std_nodes_dict = aggregate_tree_nodes(group, ds.name)

            # Compute the mean and std of aggregated tree nodes across multiple experiments for every pareto selection strategy
            mean_depth_dict, std_depth_dict = aggregate_tree_depth(group, ds.name)

            # Compute the mean and std of aggregated optimal tree auc across multiple experiments for every pareto selection strategy
            mean_auc_dict, std_auc_dict = aggregate_optimal_auc(group, ds.name)

            # Compute the mean and std of aggregated optimal tree distance across multiple experiments for every pareto selection strategy
            mean_dist_dict, std_dist_dict = aggregate_optimal_distance(group, ds.name)
            '''================================='''

            dataset_dict[ds.name] = {
                "feature_std":            mean_std_feature_dict,
                "distinct_top_features":  distinct_feats_dict,
                "tree_nodes": {
                    "mean": mean_nodes_dict,
                    "std":  std_nodes_dict,
                },
                "tree_depth": {
                    "mean": mean_depth_dict,
                    "std":  std_depth_dict,
                },
                "optimal_auc": {
                    "mean": mean_auc_dict,
                    "std":  std_auc_dict,
                },
                "optimal_distance": {
                    "mean": mean_dist_dict,
                    "std":  std_dist_dict,
                },
            }

            serializable = _make_jsonable(dataset_dict)
            group_logger.log_metrics({"dataset_aggregates": serializable})  
            print(f"saved group metrics to metrics.json")
            '''======Aggregate Plotting========='''
            plot_aggregate_metrics(dataset_dict, group)
            '''================================='''
        
        except Exception as e:
            print(f"\nError encountered: {e!r}\nCleaning up logs and experiment folder…")

            # 1) remove logs/<each_experiment>
            logs_root = Path("logs").resolve()
            for exp_name in getattr(group, "experiments", []):
                log_dir = logs_root / exp_name
                if log_dir.exists():
                    rmtree(log_dir)

            # 2) remove the experiments/<group_name> folder
            if group.group_path.exists():
                rmtree(group.group_path)

            # re-raise so you still see the traceback
            raise


        # Generate and save group summary
        summary = group.get_summary()
        summary_path = group.group_path / "group_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nExperiment group summary saved to {summary_path}")
        print(f"Total experiments run: {len(args.seeds)}")

        


if __name__ == "__main__":
    main()