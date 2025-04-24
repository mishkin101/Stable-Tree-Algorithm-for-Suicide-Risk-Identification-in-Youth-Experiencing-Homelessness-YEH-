import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def load_data(filepath):
    """Loads dataset from a CSV file."""
    return pd.read_csv(filepath)

def run_single_experiment(X, y, max_depth=None, test_size=0.2, random_state=None):
    """
    Runs a single train-test split and fits a decision tree.

    Parameters:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        max_depth (int): Max depth of the tree.
        test_size (float): Proportion of test set.
        random_state (int): Seed for reproducibility.

    Returns:
        float: Accuracy on test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

def run_multiple_experiments(X, y, max_depth=None, test_size=0.2, n_splits=5):
    """
    Runs multiple train-test splits and returns the average accuracy.

    Parameters:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        max_depth (int): Max depth of the tree.
        test_size (float): Proportion of test set.
        n_splits (int): Number of train-test splits.

    Returns:
        list of float: List of accuracies for each split.
    """
    accuracies = []
    for i in range(n_splits):
        acc = run_single_experiment(X, y, max_depth=max_depth, test_size=test_size, random_state=i)
        accuracies.append(acc)
    return accuracies

def main(filepath, max_depth=None, test_size=0.2, n_splits=5):
    """
    Main function to load data and run experiments.

    Parameters:
        filepath (str): Path to CSV file.
        max_depth (int): Max depth of the decision tree.
        test_size (float): Test set proportion.
        n_splits (int): Number of train-test splits.
    """
    data = load_data(filepath)
    X = data.drop(columns=['target'])  # Assumes 'target' is the label column
    y = data['target']

    results = run_multiple_experiments(X, y, max_depth=max_depth, test_size=test_size, n_splits=n_splits)
    print(f"Accuracies: {results}")
    print(f"Average Accuracy: {np.mean(results):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run decision tree experiments.")
    parser.add_argument('filepath', type=str, help='Path to CSV dataset')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of decision tree')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of test data')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of train/test splits')
    args = parser.parse_args()

    main(args.filepath, args.max_depth, args.test_size, args.n_splits)
