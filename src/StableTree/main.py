# src/StableTree/main.py
from constants import (
    FEATURE_SETS,
    DEPTHS,
    MIN_SAMPLES,
    NUM_BOOTSTRAPS,
    RANDOM_SEED,
)
from data import prepare_data, random_train_split
from models import bootstrap_trees, evaluate_predictive_power
from distance import compute_average_distances
from pareto import pareto_optimal_trees, select_final_tree, select_auc_maximizing_tree, select_distance_minimizing_tree
from visualization import plot_pareto_frontier, plot_decision_tree, plot_common_features
from logging_utils import ExperimentLogger
from evaluation import common_features, gini_importance
from sklearn.preprocessing import label_binarize
import visualize_like_orig as vis_orig

import numpy as np
import pandas as pd
import sys
import argparse
from datetime import datetime
from pathlib import Path
import json
import os

# Add required paths to system path if needed
src_path = Path("src/dt-distance").resolve()
data_path = Path("data").resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
sys.path.append(str(data_path))


class ExperimentGroup:
    """Manages a group of related experiments with different random seeds."""
    
    def __init__(self, group_name=None):
        """Initialize an experiment group with a unique name."""
        if group_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            group_name = f"experiment_group_{timestamp}"
        
        self.group_name = group_name
        self.group_path = Path(f"experiments/{group_name}")
        self.group_path.mkdir(parents=True, exist_ok=True)
        
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
    
    def _save_metadata(self):
        """Save metadata about this experiment group."""
        metadata = {
            "group_name": self.group_name,
            "created_at": datetime.now().isoformat(),
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
                summary["experiments"].append({
                    "name": exp_name,
                    "metrics": metrics
                })
    
    def aggregate_statistcs(self, meta_key):
        
        return self
    


def run_experiment(seed, label="suicidea", experiment_group=None):
    """Run a single experiment with the specified random seed."""
    # Create a unique experiment name based on timestamp and seed
    rng = np.random.default_rng(seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}_seed_{seed}"
    
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
        "DATA_PATH": "data/DataSet_Combined_SI_SNI_Baseline_FE.csv"
    })
    
    # Set random seed for reproducibility
    # np.random.seed(seed)

    DATA_PATH = "data/DataSet_Combined_SI_SNI_Baseline_FE.csv"

    # Load and prepare data
    df = pd.read_csv(DATA_PATH)
    
    # Log the label being used
    logger.log_config({"label": label})
    
    X_full, y_full, X_train, X_test, y_train, y_test = prepare_data(
        df, FEATURE_SETS[label], label, rng
    )

    # Log data metrics
    dataset_metrics = {
        "num_samples_full": len(X_full),
        "num_samples_train": len(X_train),
        "num_samples_test": len(X_test),
        "train_shape": X_train.shape,
    }
    logger.log_metrics(dataset_metrics)
    
    print(f"Experiment: {experiment_name} - Seed: {seed}")
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

    # select final auc tree
    selected_auc_tree_index = select_auc_maximizing_tree(distances, auc_scores, pareto_trees)
    logger.log_metrics({
        "selected_auc_tree_index": int(selected_auc_tree_index),
        "selected_auc_tree_distance": float(distances[selected_auc_tree_index]),
        "selected_auc_tree_auc": float(auc_scores[selected_auc_tree_index])
    })
    print(f"Selected AUC maximizing tree index: {selected_auc_tree_index}")
    selected_auc_tree = T[selected_auc_tree_index]

    #select final distance tree
    selected_dist_tree_index = select_final_tree(distances, auc_scores, pareto_trees)
    logger.log_metrics({
        "selected_dist_tree_index": int(selected_dist_tree_index),
        "selected_dist_tree_distance": float(distances[selected_dist_tree_index]),
        "selected_dist_tree_auc": float(auc_scores[selected_dist_tree_index])
    })
    print(f"Selected distance minimizing tree index: {selected_dist_tree_index}")
    selected_auc_tree = T[selected_dist_tree_index]
    
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
    dataset_name = os.path.splitext(os.path.basename(DATA_PATH))[0]
    plot_common_features(common_feat_list, dataset=dataset_name)
    logger.save_figure("top_common_features")


    plot_pareto_frontier(distances, auc_scores, pareto_trees)
    logger.save_figure("pareto_frontier")
    
    return experiment_name



def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Run StableTree experiments with multiple seeds')
    parser.add_argument('--seeds', type=int, nargs='+', default=[RANDOM_SEED], 
                        help='List of random seeds to use for experiments')
    parser.add_argument('--label', type=str, default="suicidea", 
                        help='Target label to predict (suicidea or suicattempt)')
    parser.add_argument('--group-name', type=str, default=None,
                        help='Name for the experiment group (default: auto-generated)')
    args = parser.parse_args()
    
    # Create an experiment group
    group = ExperimentGroup(args.group_name)
    print(f"Created experiment group: {group.group_name}")
    
    # Run experiments for each seed
    for seed in args.seeds:
        print(f"\n{'='*50}")
        print(f"Running experiment with seed {seed}")
        print(f"{'='*50}")
        experiment_name = run_experiment(seed, args.label, group)
        print(f"Completed experiment: {experiment_name}")
    
    '''======TODO=======
    Compute aggregate statistics on experiment group 
    Sve plot of aggregate metrics to experiment group
    '''

    # Generate and save group summary
    summary = group.get_summary()
    summary_path = group.group_path / "group_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExperiment group summary saved to {summary_path}")
    print(f"Total experiments run: {len(args.seeds)}")


if __name__ == "__main__":
    main()