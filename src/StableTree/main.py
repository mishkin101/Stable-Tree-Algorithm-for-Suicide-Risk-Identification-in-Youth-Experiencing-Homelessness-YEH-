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

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add required paths to system path if needed
src_path = Path("src/dt-distance").resolve()
data_path = Path("data").resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
sys.path.append(str(data_path))


def main():
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    DATA_PATH = "data/DataSet_Combined_SI_SNI_Baseline_FE.csv"

    # Load and prepare data
    df = pd.read_csv(DATA_PATH)
    label = "suicidea"  # Can be changed to "suicattempt"
    X_full, y_full, X_train, X_test, y_train, y_test = prepare_data(
        df, FEATURE_SETS[label], label
    )

    print(f"Number of samples in the full dataset: {len(X_full)}")
    print(f"Number of samples in the training set: {len(X_train)}")
    print(f"Number of samples in the test set: {len(X_test)}")
    print(f"Shape of training set: {X_train.shape}")

    # Create random split for baseline trees
    X0, y0 = random_train_split(X_train.values, y_train.values)
    print(f"Shape of random split: {X0.shape}, {y0.shape}")

    # Generate bootstrap trees
    T0 = bootstrap_trees(X0, y0, DEPTHS, MIN_SAMPLES, NUM_BOOTSTRAPS)
    print(f"Number of trees in T0: {len(T0)}")

    T = bootstrap_trees(
        X_train.values, y_train.values, DEPTHS, MIN_SAMPLES, NUM_BOOTSTRAPS
    )
    print(f"Number of trees in T: {len(T)}")

    # Compute distances between tree sets
    distances = compute_average_distances(T0, T, X_train, y_train)
    print(f"Number of distances computed: {len(distances)}")

    # Evaluate predictive performance
    auc_scores = evaluate_predictive_power(T, X_test.values, y_test.values)
    print(f"Average AUC score: {np.mean(auc_scores)}")

    # Find Pareto optimal trees
    pareto_trees = pareto_optimal_trees(distances, auc_scores)
    print(f"Number of Pareto optimal trees: {len(pareto_trees)}")

    # Select the final tree
    selected_tree_index = select_final_tree(distances, auc_scores, pareto_trees)
    print(f"Selected tree index: {selected_tree_index}")

    # Visualize results
    selected_tree = T[selected_tree_index]
    plot_decision_tree(
        selected_tree,
        feature_names=X_full.columns,
        class_names=["No", "Yes"],
        title="Pareto Optimal Tree",
    )

    plot_pareto_frontier(distances, auc_scores, pareto_trees)


if __name__ == "__main__":
    main()
