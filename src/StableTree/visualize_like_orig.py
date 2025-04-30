# visualization_simpler.py (updated to use ExperimentLogger for saving)
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    auc,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Import ExperimentLogger
from logging_utils import ExperimentLogger

# --------------------------------------------------------------------------- #
# 1. Display an image at native resolution (unchanged)
# --------------------------------------------------------------------------- #
def plot_im(im: np.ndarray, dpi: int = 300) -> None:
    """Display a saved PNG in its native resolution."""
    px, py = im.shape[:2]
    size = (py / dpi, px / dpi)
    fig = plt.figure(figsize=size, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(im, aspect="auto")
    plt.show(block=False)


# --------------------------------------------------------------------------- #
# 2. Decision-tree visualisation
# --------------------------------------------------------------------------- #

def visualize_tree(
    tree: DecisionTreeClassifier,
    feature_names: Iterable[str],
    *,
    label: str,
    class_names: Iterable[str] = ("Safe", "Risk!"),
    title: str | None = None,
    dpi: int = 300,
    logger: ExperimentLogger,
    fig_name: str,
) -> Path:
    """
    Save a coloured PNG of a fitted tree **without** Graphviz, using ExperimentLogger.

    Parameters
    ----------
    tree : DecisionTreeClassifier
    feature_names : Iterable[str]
    label : str                  # e.g. "Seed 0"
    class_names : Iterable[str]
    title : str | None
    dpi : int
    logger : ExperimentLogger    # logger for saving
    fig_name : str               # name (no extension) for saving

    Returns
    -------
    Path to saved PNG
    """
    fig, ax = plt.subplots(figsize=(20, 12), dpi=dpi)
    plot_tree(
        tree,
        ax=ax,
        feature_names=list(feature_names),
        class_names=list(class_names),
        filled=True,
        rounded=True,
        proportion=True,
    )
    ax.set_title(title or f"Decision Tree – {label}", fontsize=14)
    fig.tight_layout()

    # Save with logger
    saved_path = logger.save_figure(fig_name)
    plt.close(fig)
    return saved_path


# --------------------------------------------------------------------------- #
# 3. Feature-importance bar-plot
# --------------------------------------------------------------------------- #

def save_feature_importance(
    tree: DecisionTreeClassifier,
    feat_names: List[str],
    *,
    label: str,
    top_k: int = 20,
    dpi: int = 300,
    logger: ExperimentLogger,
    fig_name: str,
) -> Path:
    """
    Save a horizontal bar plot of the **top-k** relative importances using ExperimentLogger.
    """
    fi = 100 * tree.feature_importances_ / tree.feature_importances_.max()
    idx = np.argsort(fi)[-top_k:]
    pos = np.arange(len(idx)) + 0.5

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(idx) + 1), dpi=dpi)
    ax.barh(pos, fi[idx], align="center")
    ax.set_yticks(pos, np.array(feat_names)[idx])
    ax.set_xlabel("Relative importance (%)")
    ax.set_title("Feature importance")
    fig.tight_layout()

    # Save with logger
    saved_path = logger.save_figure(fig_name)
    plt.close(fig)
    return saved_path


# --------------------------------------------------------------------------- #
# 4. Micro-average ROC curve
# --------------------------------------------------------------------------- #

def plot_roc_curve(
    y_test_bin: np.ndarray,
    y_score: np.ndarray,
    *,
    label: str,
    dpi: int = 300,
    logger: ExperimentLogger,
    fig_name: str,
) -> Path:
    """
    Save a micro-average ROC curve using ExperimentLogger.
    """
    # Determine binary vs multiclass
    if y_score.shape[1] == 2 and y_test_bin.shape[1] == 1:
        y_true = y_test_bin.ravel()
        y_score_pos = y_score[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_score_pos)
        roc_auc = auc(fpr, tpr)
    else:
        y_true_flat = y_test_bin[:, :y_score.shape[1]].ravel()
        y_score_flat = y_score.ravel()
        fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)
        roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", lw=2)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.05),
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="ROC – micro average",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()

    # Save with logger
    saved_path = logger.save_figure(fig_name)
    plt.close(fig)
    return saved_path


# --------------------------------------------------------------------------- #
# 5. Persist plain-text metrics (unchanged)
# --------------------------------------------------------------------------- #
def write_metrics(
    features: List[str],
    label: str,
    *,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_prob: np.ndarray,
    y_test_bin: np.ndarray,
    logger = ExperimentLogger,
) -> Path:
    #\\TODO #24 check to see why classification.py has certain classes(0/1) where the model never predicts that class @adbX
    # This function is unchanged and writes metrics text files.
    metrics_path = logger.experiment_dir / f"original_metrics_{label}.txt"
    with metrics_path.open("w") as f:
        f.write("-" * 60 + "\n")
        f.write(f"Features: {features}\n")
        f.write(f"Label   : {label}\n\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
        if y_pred_prob.shape[1] == 2 and y_test_bin.shape[1] == 1:
            auc_val = roc_auc_score(y_test_bin.ravel(), y_pred_prob[:, 1])
        else:
            auc_val = roc_auc_score(
                y_test_bin[:, :y_pred_prob.shape[1]],
                y_pred_prob,
                average="micro",
            )
        f.write(f"AUC     : {auc_val:.4f}\n")
        f.write("Classification report:\n")
        f.write(classification_report(y_test, y_pred, digits=3, zero_division=0))
        f.write(
            "\nNote:\n"
            "  Sensitivity == Recall of the positive class\n"
            "  Specificity == Recall of the negative class\n"
        )
        f.write("-" * 60 + "\n")

    logging.info("Metrics written to %s", metrics_path)
    return metrics_path
