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
from pareto import pareto_optimal_trees, select_final_tree
from visualization import plot_pareto_frontier, plot_decision_tree
from logging_utils import ExperimentLogger

import numpy as np
import pandas as pd
import sys
from datetime import datetime
from pathlib import Path

# Add required paths to system path if needed
src_path = Path("src/dt-distance").resolve()
data_path = Path("data").resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
sys.path.append(str(data_path))


def main():
    # Create a unique experiment name based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}"
    
    # Initialize the logger
    logger = ExperimentLogger(experiment_name)
    
    # Log configuration
    logger.log_config({
        "DEPTHS": DEPTHS,
        "MIN_SAMPLES": MIN_SAMPLES,
        "NUM_BOOTSTRAPS": NUM_BOOTSTRAPS,
        "RANDOM_SEED": RANDOM_SEED,
        "DATA_PATH": "data/DataSet_Combined_SI_SNI_Baseline_FE.csv"
    })
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    DATA_PATH = "data/DataSet_Combined_SI_SNI_Baseline_FE.csv"

    # Load and prepare data
    df = pd.read_csv(DATA_PATH)
    label = "suicidea"  # Can be changed to "suicattempt"
    
    # Log the label being used
    logger.log_config({"label": label})
    
    X_full, y_full, X_train, X_test, y_train, y_test = prepare_data(
        df, FEATURE_SETS[label], label
    )

    # Log data metrics
    dataset_metrics = {
        "num_samples_full": len(X_full),
        "num_samples_train": len(X_train),
        "num_samples_test": len(X_test),
        "train_shape": X_train.shape,
    }
    logger.log_metrics(dataset_metrics)
    
    print(f"Number of samples in the full dataset: {len(X_full)}")
    print(f"Number of samples in the training set: {len(X_train)}")
    print(f"Number of samples in the test set: {len(X_test)}")
    print(f"Shape of training set: {X_train.shape}")

    # Create random split for baseline trees
    X0, y0 = random_train_split(X_train.values, y_train.values)
    logger.log_metrics({"random_split_shape": (X0.shape, y0.shape)})
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
    logger.log_metrics({"average_auc": np.mean(auc_scores)})
    print(f"Average AUC score: {np.mean(auc_scores)}")

    # Find Pareto optimal trees
    pareto_trees = pareto_optimal_trees(distances, auc_scores)
    logger.log_metrics({"num_pareto_trees": len(pareto_trees)})
    print(f"Number of Pareto optimal trees: {len(pareto_trees)}")

    # Select the final tree
    selected_tree_index = select_final_tree(distances, auc_scores, pareto_trees)
    logger.log_metrics({"selected_tree_index": selected_tree_index})
    print(f"Selected tree index: {selected_tree_index}")

    # Visualize results
    selected_tree = T[selected_tree_index]
    plot_decision_tree(
        selected_tree,
        feature_names=X_full.columns,
        class_names=["No", "Yes"],
        title="Pareto Optimal Tree",
    )
    logger.save_figure("decision_tree")

    plot_pareto_frontier(distances, auc_scores, pareto_trees)
    logger.save_figure("pareto_frontier")


if __name__ == "__main__":
    main()