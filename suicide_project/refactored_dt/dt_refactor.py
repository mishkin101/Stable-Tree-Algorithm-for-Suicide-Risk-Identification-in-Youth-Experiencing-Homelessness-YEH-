"""decision_tree_refactor.py
Refactored, modular version of *run8.py* and *run9.py*
```
python decision_tree_refactor.py  # runs both targets sequentially

python decision_tree_refactor.py --label suicidea       # single run
python decision_tree_refactor.py --label suicattempt       # single run
```
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils.class_weight import compute_class_weight

###############################################################################
# -----------------------------  CONFIGURATION  ----------------------------- #
###############################################################################
DATA = Path("data/DataSet_Combined_SI_SNI_Baseline_FE.csv")
DATA_PATH = DATA.resolve()
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Feature lists are copied verbatim from run8.py and run9.py
FEATURE_SETS: Dict[str, List[str]] = {
    "suicidea": [
        "age", "gender", "sexori", "raceall", "trauma_sum", "cesd_score", "harddrug_life", "school", "degree", "job", "sex", "concurrent", "exchange", "children", "weapon", "fight", "fighthurt", "ipv", "ptsd_score", "alcfirst", "potfirst", "staycurrent", "homelage", "time_homeless_month", "jail", "jailkid", "gettherapy", "sum_alter", "sum_family", "sum_home_friends", "sum_street_friends", "sum_unknown_alter", "sum_talk_once_week", "sum_alter3close", "prop_family_harddrug", "prop_friends_harddrug", "prop_friends_home_harddrug", "prop_friends_street_harddrug", "prop_alter_all_harddrug", "prop_enc_badbehave", "prop_alter_homeless", "prop_family_emosup", "prop_friends_emosup", "prop_friends_home_emosup", "prop_friends_street_emosup", "prop_alter_all_emosup", "prop_family_othersupport", "prop_friends_othersupport", "prop_friends_home_othersupport", "prop_friends_street_othersupport", "prop_alter_all_othersupport", "sum_alter_staff", "prop_object_badbehave", "prop_enc_goodbehave", "prop_alter_school_job", "sum_alter_borrow"],
    "suicattempt": [
        "age", "gender", "sexori", "raceall", "trauma_sum", "cesd_score", "harddrug_life", "school", "degree", "job", "sex", "concurrent", "exchange", "children", "weapon", "fight", "fighthurt", "ipv", "ptsd_score", "alcfirst", "potfirst", "staycurrent", "homelage", "time_homeless_month", "jail", "jailkid", "gettherapy", "sum_alter", "prop_family", "prop_home_friends", "prop_street_friends", "prop_unknown_alter", "sum_talk_once_week", "sum_alter3close", "prop_family_harddrug", "prop_friends_harddrug", "prop_friends_home_harddrug", "prop_friends_street_harddrug", "prop_alter_all_harddrug", "prop_enc_badbehave", "prop_alter_homeless", "prop_family_emosup", "prop_friends_emosup", "prop_friends_home_emosup", "prop_friends_street_emosup", "prop_alter_all_emosup", "prop_family_othersupport", "prop_friends_othersupport", "prop_friends_home_othersupport", "prop_friends_street_othersupport", "prop_alter_all_othersupport", "sum_alter_staff", "prop_object_badbehave", "prop_enc_goodbehave", "prop_alter_school_job", "sum_alter_borrow"],
}

# Model hyper‑parameters (copied verbatim from the original scripts)
MODEL_PARAMS = {
    "suicidea": dict(min_samples_leaf=10, min_samples_split=20, max_depth=4),
    "suicattempt": dict(min_samples_leaf=10, min_samples_split=30, max_depth=4),
}

###############################################################################
# ----------------------------  UTILITY FUNCTIONS  -------------------------- #
###############################################################################


def setup_logging() -> None:
    """Configure root logger the same way both original scripts did."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root.addHandler(handler)
    # Avoid matplotlib debug noise
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def plot_im(im: np.ndarray, dpi: int = 300) -> None:
    """Display a saved PNG in native resolution (replicates original helper)."""
    px, py = im[:, :, 0].shape
    size = (py / float(dpi), px / float(dpi))
    fig = plt.figure(figsize=size, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    for side in ("right", "left", "top", "bottom"):
        ax.spines[side].set_color("none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(im, aspect="auto")
    # plt.show()


def visualize_tree(tree: DecisionTreeClassifier, feature_names: List[str], *, label: str) -> Path:
    """Save a colour‑coded PNG of the fitted tree and return the path."""
    dot_data = export_graphviz(
        tree,
        out_file=None,
        filled=True,
        rounded=True,
        proportion=True,
        precision=2,
        feature_names=feature_names,
        class_names=("Safe", "Risk!"),
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    for node in graph.get_node_list():
        label_txt = node.get_label()
        if label_txt:
            values = [float(v) for v in label_txt.split("value = [")[1].split("]")[0].split(",")]
            values = [int(255 * v) for v in values]
            node.set_fillcolor(f"#{values[1]:02x}{values[0]:02x}00")
    path = OUTPUT_DIR / f"Decision_Tree_{label}.png"
    graph.write_png(str(path))
    return path


def save_feature_importance(tree: DecisionTreeClassifier, feat_names: List[str], *, label: str) -> None:
    """Plot feature importance (identical logic, but isolated)."""
    fi = 100.0 * tree.feature_importances_ / tree.feature_importances_.max()
    idx = np.argsort(fi)[-20:]
    pos = np.arange(idx.shape[0]) + 0.5
    plt.figure()
    plt.barh(pos, fi[idx], align="center")
    plt.yticks(pos, np.array(feat_names)[idx])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"Feature_Importance_{label}.png")
    # plt.show()
    logging.info("Feature importance saved.")


def plot_roc_curve(y_test_bin: np.ndarray, y_score: np.ndarray, *, label: str) -> None:
    """Recreate micro‑average ROC curve plot."""
    y_true_flat = y_test_bin[:, : y_score.shape[1]].ravel()
    y_score_flat = y_score.ravel()

    fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="navy", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (Micro)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"ROC_Curve_{label}.png")
    # plt.show()


def write_metrics(
    features: List[str],
    label: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_prob: np.ndarray,
    y_test_bin: np.ndarray,
) -> None:
    """Persist metrics exactly like the original scripts (one file per label)."""
    metrics_path = OUTPUT_DIR / f"Metrics_{label}.txt"
    with metrics_path.open("w") as f:
        f.write("-" * 53 + "\n")
        f.write("Features: " + str(features) + "\n")
        f.write("Label: " + label + "\n\n")
        f.write("Accuracy: " + str(accuracy_score(y_test, y_pred)) + "\n")
        f.write(
            "AUC: "
            + str(roc_auc_score(y_test_bin[:, :-1], y_pred_prob, average="micro"))
            + "\n"
        )
        f.write("Classification Report:\n" + classification_report(y_test, y_pred))
        f.write(
            "Note: \nSensitivity == Recall of the Positive Class.\n"
            "Specificity == Recall of the Negative Class.\n"
        )
        f.write("-" * 53 + "\n")
    logging.info("Metrics written to %s", metrics_path)

###############################################################################
# ---------------------------  PIPELINE FUNCTIONS  -------------------------- #
###############################################################################


def prepare_data(df: pd.DataFrame, features: List[str], label: str):
    """Drop NA, split 75 / 25 exactly like the originals (no shuffle)."""
    subset = df[features + [label]].dropna().copy()
    cutoff = int(round(len(subset) * 0.75))
    X_train = subset.iloc[:cutoff, :-1].values
    y_train = subset.iloc[:cutoff, -1].values
    X_test = subset.iloc[cutoff:, :-1].values
    y_test = subset.iloc[cutoff:, -1].values
    return subset, X_train, X_test, y_train, y_test


def fit_model(X_train: np.ndarray, y_train: np.ndarray, *, params: dict, y_full: np.ndarray):
    cw = compute_class_weight(class_weight="balanced", classes=np.unique(y_full), y=y_full)
    clf = DecisionTreeClassifier(
        criterion="gini",
        class_weight={0: cw[0], 1: cw[1]},
        min_impurity_decrease=0.01,
        **params,
    )
    clf.fit(X_train, y_train)
    return clf


def run_experiment(label: str) -> None:
    logging.info("==== Running experiment for %s ====", label)

    # ---------------------------------------------------------------------
    # Load & prepare data
    # ---------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    features = FEATURE_SETS[label]
    subset, X_train, X_test, y_train, y_test = prepare_data(df, features, label)

    # ---------------------------------------------------------------------
    # Fit model
    # ---------------------------------------------------------------------
    clf = fit_model(X_train, y_train, params=MODEL_PARAMS[label], y_full=subset[label].values)

    # ---------------------------------------------------------------------
    # Visualisations & metrics
    # ---------------------------------------------------------------------
    tree_img = visualize_tree(clf, features, label=label)
    logging.info("Tree visual saved at %s", tree_img)

    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    save_feature_importance(clf, features, label=label)

    n_classes = 2
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes + 1)))

    print("-" * 53)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test_bin[:, :-1], y_pred_prob, average="micro"))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("-" * 53)

    write_metrics(features, label, y_test, y_pred, y_pred_prob, y_test_bin)
    plot_roc_curve(y_test_bin, y_pred_prob, label=label)

    # Display tree in‑line (replicates original convenience)
    mpl.rcParams["figure.dpi"] = 300
    plot_im(mpimg.imread(tree_img))

###############################################################################
# ---------------------------------- MAIN ----------------------------------- #
###############################################################################


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run suicide risk Decision Trees")
    parser.add_argument("--label", choices=list(FEATURE_SETS.keys()), help="Run a single label only")
    args = parser.parse_args()

    setup_logging()

    labels_to_run = [args.label] if args.label else list(FEATURE_SETS.keys())
    for lbl in labels_to_run:
        run_experiment(lbl)
        logging.info("==== Ran experiment %s successfully ====", lbl)


if __name__ == "__main__":
    main()
