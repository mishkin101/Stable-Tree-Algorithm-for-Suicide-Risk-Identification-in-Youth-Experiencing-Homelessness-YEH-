from pathlib import Path
import sys

src_path = Path("src/dt-distance").resolve()
sys.path.append(str(src_path))

from dt_distance.distance_calculator import DistanceCalculator
from alive_progress import alive_bar

def compute_average_distances(T0, T, X_train, y_train):
    """
    Compute average distances between two sets of trees.
    
    Args:
        T0: First set of decision trees
        T: Second set of decision trees 
        X_train: Feature matrix
        y_train: Target vector
        
    Returns:
        List of average distances for each tree in T
    """
    X_train_values = X_train.values if hasattr(X_train, "values") else X_train
    distances: list[float] = []

    with alive_bar(len(T), title="Computing average tree distances", dual_line=True, spinner="waves", bar="smooth") as bar:
        for tree_b in T:
            d_b = 0.0
            for tree_beta in T0:
                calc = DistanceCalculator(tree_beta, tree_b, X=X_train_values, y=y_train)
                d_b += calc.compute_tree_distance()

            mean_dist = d_b / len(T0)
            distances.append(mean_dist)
            bar()

    return distances