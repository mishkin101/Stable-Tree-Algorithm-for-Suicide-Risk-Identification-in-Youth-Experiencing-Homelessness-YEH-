# File: bertsimas_stable/stable_trainer.py

import numpy as np
import itertools
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from .Paths import tree_distance


class StableTrainer:
    """
    A class implementing the basic steps from
    Bertsimas & Digalakis for training stable trees.
    """

    def __init__(self,
                 depths=[3, 5, 7],
                 min_samples_leaf=[5, 10],
                 random_state=42,
                 max_depth_for_lambda=7):
        """
        Parameters
        ----------
        depths : list
            Candidate max_depth hyperparams to explore.
        min_samples_leaf : list
            Candidate min_samples_leaf hyperparams to explore.
        random_state : int
            Random seed for reproducibility.
        max_depth_for_lambda : int
            Used to set lambda = 2 * max_depth_for_lambda in the distance metric.
        """
        self.depths = depths
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        # This sets how heavily to penalize differences in class labels
        # in the path distance metric (lambda_val).
        self.lambda_val = 2 * max_depth_for_lambda

        # Collections of trees
        self.T0_ = None
        self.T_ = None

        # Metadata
        self.distances_ = []
        self.performances_ = []
        self.pareto_indices_ = []
        self.pareto_trees_ = []

        # Will hold final chosen stable tree
        self.stable_tree_ = None
        # Hold image of the final stable tree
        self.stable_tree_image_ = None
        self.best_idx_ = None

        # Global feature ranges for distance computations
        self.global_lower_ = None
        self.global_upper_ = None

    ############################################################################
    # 1) Private utility: train a "collection" of candidate trees
    ############################################################################
    def _train_collection(self, X, y):
        """
        Train a collection of DecisionTreeClassifiers by varying depths
        and min_samples_leaf. Returns a list of fitted sklearn estimators.
        """
        trees = []
        for depth, min_leaf in itertools.product(self.depths, self.min_samples_leaf):
            clf = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=min_leaf,
                random_state=self.random_state
            )
            clf.fit(X, y)
            trees.append(clf)
        return trees

    def fit(self, X0, y0, X_full, y_full, X_test=None, y_test=None):
        """
        The main interface method that:
         1) Trains collection T0 from (X0, y0)
         2) Trains collection T  from (X_full, y_full)
         3) Computes distances from each tree in T to T0
         4) Computes performance (accuracy) on (X_test, y_test) if provided,
            otherwise on (X_full, y_full).
         5) Identifies Pareto frontier in (distance, performance) space
         6) Selects a final stable tree (self.stable_tree_)

        Returns self for chaining.
        """
        # 1) Train T0
        # print("[INFO] Training first collection (T0)...")
        self.T0_ = self._train_collection(X0, y0)
        # print(f"  -> T0: {len(self.T0_)} trees")

        # 2) Train T
        # print("[INFO] Training second collection (T)...")
        self.T_ = self._train_collection(X_full, y_full)
        # print(f"  -> T: {len(self.T_)} trees")

        # 3) Distances (average from each tree in T to T0)
        self.global_lower_ = X_full.min().values
        self.global_upper_ = X_full.max().values
        self.distances_ = []
        for tree_b in self.T_:
            d_b = 0.0
            for tree_beta in self.T0_:
                d_b += tree_distance(
                    tree_beta, tree_b,
                    global_lower=self.global_lower_,
                    global_upper=self.global_upper_,
                    lambda_val=self.lambda_val
                )
            d_b /= len(self.T0_)
            self.distances_.append(d_b)

        # 4) Performance
        self.performances_ = []
        if X_test is not None and y_test is not None:
            # Evaluate all trees on holdout test
            for tree_b in self.T_:
                y_pred = tree_b.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                self.performances_.append(acc)
        else:
            # In-sample fallback
            for tree_b in self.T_:
                acc = accuracy_score(y_full, tree_b.predict(X_full))
                self.performances_.append(acc)

        # 5) Pareto frontier
        pairs = list(zip(self.distances_, self.performances_))
        self.pareto_indices_ = self._get_pareto_indices(pairs)
        self.pareto_trees_ = [self.T_[i] for i in self.pareto_indices_]

        # 6) Choose final stable tree
        self._select_stable_tree()

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X using the chosen stable tree.
        """
        if self.stable_tree_ is None:
            raise ValueError("No stable_tree_ found. Call .fit(...) first.")
        return self.stable_tree_.predict(X)
    
    def _get_pareto_indices(self, pairs):
        """
        Identify indices that are non-dominated in the (distance,performance) space.
        """
        indices = []
        for i, (di, pi) in enumerate(pairs):
            dominated = False
            for j, (dj, pj) in enumerate(pairs):
                if j != i:
                    # j dominates i if dj <= di, pj >= pi, and at least one strict
                    if (dj <= di and pj >= pi) and (dj < di or pj > pi):
                        dominated = True
                        break
            if not dominated:
                indices.append(i)
        return indices

    def _select_stable_tree(self):
        """
        Pick the final stable tree from the Pareto set, based on the 
        distance threshold approach described in train_stable docstring.
        """
        if not self.distances_:
            print("[WARN] No distances available. Did you call .fit(...) yet?")
            return

        dist_min = min(self.distances_)
        dist_max = max(self.distances_)
        threshold = dist_min + 0.2 * (dist_max - dist_min)

        candidate_indices = [
            i for i in self.pareto_indices_
            if self.distances_[i] <= threshold
        ]
        if candidate_indices:
            # Among those candidates, pick the best accuracy
            best_idx = max(candidate_indices,
                           key=lambda i: self.performances_[i])
        else:
            # fallback: best accuracy overall
            best_idx = np.argmax(self.performances_)

        self.best_idx_ = best_idx
        self.stable_tree_ = self.T_[best_idx]

        print(f"[INFO] Final chosen stable tree index = {best_idx}, "
              f"distance = {self.distances_[best_idx]:.4f}, "
              f"perf = {self.performances_[best_idx]:.4f}")

    def plot_stable_tree(self, feature_names=None, class_names=None, max_depth_to_plot=None):
        """
        Visualize the final chosen stable tree using sklearn.tree.plot_tree.
        """
        if self.stable_tree_ is None:
            print("No stable_tree_ found; please call .fit(...) first.")
            return

        plt.figure(figsize=(12, 8))
        plot_tree(
            self.stable_tree_,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            max_depth=max_depth_to_plot
        )
        
        # plt.show()
        # return plt
