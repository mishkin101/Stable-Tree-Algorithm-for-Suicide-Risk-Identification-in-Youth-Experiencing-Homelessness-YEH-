import numpy as np
from typing import List, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

def train_decision_tree(X, y, depth, min_samples_leaf):
    """
    Train a decision tree with specified parameters.
    
    Args:
        X: Feature matrix
        y: Target vector
        depth: Maximum depth of the tree
        min_samples_leaf: Minimum samples per leaf
        
    Returns:
        Trained decision tree classifier
    """
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_samples_leaf)
    clf.fit(X, y)
    return clf

def bootstrap_trees(X, y, depths, min_samples, B):
    """
    Create B bootstrap trees by sampling with replacement from data.
    
    Args:
        X: Feature matrix
        y: Target vector
        depths: List of possible tree depths
        min_samples: List of possible minimum samples per leaf
        B: Number of bootstrap samples to create
        
    Returns:
        List of trained decision trees
    """
    trees = []
    for _ in range(B):
        X_sample, y_sample = resample(X, y, replace=True)
        depth = np.random.choice(depths)
        min_leaf = np.random.choice(min_samples)
        # print(f"Training tree with depth {depth} and min_samples_leaf {min_leaf}")
        tree = train_decision_tree(X_sample, y_sample, depth, min_leaf)
        trees.append(tree)
    return trees

def evaluate_predictive_power(trees, X_holdout, y_holdout):
    """
    Evaluate a list of trees using AUC score.
    
    Args:
        trees: List of trained decision trees
        X_holdout: Feature matrix for evaluation
        y_holdout: Target vector for evaluation
        
    Returns:
        List of AUC scores for each tree
    """
    auc_scores = []
    for tree in trees:
        y_proba = tree.predict_proba(X_holdout)[:, 1]
        auc = roc_auc_score(y_holdout, y_proba)
        auc_scores.append(auc)
    return auc_scores