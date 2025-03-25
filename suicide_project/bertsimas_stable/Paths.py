import numpy as np
from scipy.optimize import linear_sum_assignment

def extract_paths_from_tree(clf, global_lower, global_upper):
    """
    Returns a list of path dicts. Each path dict has:
      - 'lower_bounds': array of floats (initially global_lower)
      - 'upper_bounds': array of floats (initially global_upper)
      - 'predicted_class': int  (argmax of leaf distribution)
    For simplicity, we handle only numeric features here.
    """

    # We'll do a DFS from the root to each leaf, updating lower/upper bounds.
    paths = []
    tree_ = clf.tree_
    # n_features = clf.n_features_in_

    def recurse(node_id, cur_lower, cur_upper):
        left_child = tree_.children_left[node_id]
        right_child = tree_.children_right[node_id]

        # If leaf: children_left == children_right
        if left_child == right_child:
            # Leaf node -> get predicted class from value (argmax)
            values = tree_.value[node_id][0]  # shape (K,) for K classes
            pred_class = int(np.argmax(values))
            # Store a path dictionary
            path_info = {
                "lower_bounds": cur_lower.copy(),
                "upper_bounds": cur_upper.copy(),
                "predicted_class": pred_class
            }
            paths.append(path_info)
        else:
            # Not a leaf; split on feature[feat] <= threshold
            feat = tree_.feature[node_id]
            thr  = tree_.threshold[node_id]

            # Left branch: X[feat] <= thr
            new_upper = cur_upper.copy()
            new_upper[feat] = min(new_upper[feat], thr)
            recurse(left_child, cur_lower.copy(), new_upper)

            # Right branch: X[feat] > thr
            new_lower = cur_lower.copy()
            # In scikit-learn, the right branch is strictly greater than thr
            new_lower[feat] = max(new_lower[feat], thr)
            recurse(right_child, new_lower, cur_upper.copy())

    # Start recursion from the root
    # Initialize each feature's bounds to [global_lower, global_upper]
    recurse(
        node_id = 0,
        cur_lower = global_lower.copy(),
        cur_upper = global_upper.copy()
    )

    return paths

def path_distance(p, q, global_lower, global_upper, lambda_val=1.0):
    """
    Distance = sum over features of normalized difference in [upper, lower] 
               + (lambda_val if predicted_class differs).
    """
    num_feats = len(p['upper_bounds'])
    dist_val = 0.0
    for j in range(num_feats):
        feat_range = max(global_upper[j] - global_lower[j], 1e-12)
        dist_val += (abs(p['upper_bounds'][j] - q['upper_bounds'][j])
                     + abs(p['lower_bounds'][j] - q['lower_bounds'][j])
                    ) / (2.0 * feat_range)
    # Add label mismatch
    if p['predicted_class'] != q['predicted_class']:
        dist_val += lambda_val
    return dist_val

def path_weight(p, global_lower, global_upper):
    """
    Path weight measures how much of the feature space this path covers, 
    for the features actually 'used' (if it tightens global range).
    """
    num_feats = len(p['upper_bounds'])
    w = 0.0
    for j in range(num_feats):
        feat_range = max(global_upper[j] - global_lower[j], 1e-12)
        # 1 if path actually splits on this feature => bounds < global
        used = (p['lower_bounds'][j] > global_lower[j] or 
                p['upper_bounds'][j] < global_upper[j])
        if used:
            w += ((p['upper_bounds'][j] - p['lower_bounds'][j]) / feat_range)
    return w

def tree_distance(tree1, tree2, global_lower, global_upper, lambda_val=1.0):
    """
    1) Extract all paths P1, P2
    2) Build a cost matrix of shape (len(P1), len(P2)) for matching path p1->p2
    3) We'll add 'skip' columns for each p1 so we can leave that path unmatched 
       at cost = w(p1).
    4) Solve the assignment problem, forcing each p2 to be matched exactly once,
       and each p1 to match exactly one among (all p2 + skip).
    """

    P1 = extract_paths_from_tree(tree1, global_lower, global_upper)
    P2 = extract_paths_from_tree(tree2, global_lower, global_upper)
    M = len(P1)
    N = len(P2)

    # If both are empty trees, distance=0
    if M == 0 and N == 0:
        return 0.0
    if M == 0:
        # Then T1 has no paths, T2 has some => cost is sum of unmatched T2?
        # By symmetry, let's also skip T2. Or consider "lack of path" as a cost.
        # For consistency, do the same logic. We'll do a minimal approach:
        return sum(path_weight(p2, global_lower, global_upper) for p2 in P2)
    if N == 0:
        # Then T2 has no paths => cost is sum of unmatched T1
        return sum(path_weight(p1, global_lower, global_upper) for p1 in P1)

    # Create cost matrix for matching p1->p2
    cost_main = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            cost_main[i, j] = path_distance(
                P1[i], P2[j], global_lower, global_upper, lambda_val=lambda_val
            )

    # For skipping p1, cost = w(p1). We'll implement skip as extra columns.
    skip_cols = np.zeros((M, M))
    for i in range(M):
        skip_cols[i, i] = path_weight(P1[i], global_lower, global_upper)
        # 0 for off-diagonal => can't pick skip_j for p1 != j
        # We'll enforce that in the assignment constraints.

    # Combine into one big cost matrix: shape (M, N + M)
    # First N columns = match P2. Next M columns = skip option.
    big_cost = np.hstack([cost_main, skip_cols])

    # We want a bipartite assignment with constraints:
    # - each row i (for P1) picks exactly 1 column
    # - each real p2 column j in [0..N-1] can be chosen at most once
    # - each skip column j in [N..N+M-1] can be chosen at most once, but also
    #   skip_j can only be used by row i=j if i == j. We'll handle that by
    #   putting high cost (inf) for mismatched i!=j.
    # So let's fix skip_cols off diagonal to a large number:
    INF = 1e9
    for i in range(M):
        for j in range(M):
            if i != j:
                big_cost[i, N + j] = INF

    # Now we have an M x (N+M) cost matrix.
    # We'll run the Hungarian method, but we need each of the N path2 columns
    # to be chosen exactly once, and each skip column at most once, etc.
    # The Hungarian algorithm only ensures each row is matched once & each col 
    # is matched at most once. We want each "p2 column" exactly once if possible.
    # If M < N, we can't match all P2. So some of them will effectively remain unmatched.
    # We can do symmetrical approach if we want to penalize unmatched P2 as well. 
    # For demonstration, let's do a partial approach that tries to minimize cost 
    # but doesn't strictly require each P2 to be matched. That means some columns
    # might remain unmatched.  If you want each P2 matched, you'd do a symmetrical 
    # extension with dummy rows, etc.

    row_ind, col_ind = linear_sum_assignment(big_cost)
    total_cost = big_cost[row_ind, col_ind].sum()
    return total_cost