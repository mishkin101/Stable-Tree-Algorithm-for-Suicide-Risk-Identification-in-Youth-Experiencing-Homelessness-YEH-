import numpy as np
from scipy.optimize import linprog
from .tree_parser import TreeParser
from .data_processor import DataProcessor  # Used if X and y are provided
from .path_extractor import PathExtractor, compute_paths_distance
from .problem_params import ProblemParams  # Ensure you have this class defined appropriately

class DistanceCalculator:
    """
    Enhanced to automatically handle data, tree processing, and compute distances between
    decision trees with options for manual problem parameter specifications.

    Note: convention on the indices of the matching variables x_pq 
    - p indexes the paths in the larger tree (so each p maps to exactly one q)
    - q indexes the paths in the smaller tree (so each q maps to at least one p)
    """
    def __init__(self, tree1, tree2, problem_params=None, X=None, y=None, dataset_name=None,
                 max_depth=None, normalize_distance=True, outcome_weight_in_path=1, round_digits=4):
        """
        Initializes with sklearn trees and either a ProblemParams object or data/dataset name,
        including options for normalization and outcome weighting.
        
        :param tree1: First sklearn decision tree model.
        :param tree2: Second sklearn decision tree model.
        :param problem_params: A ProblemParams object specifying feature information (optional).
        :param X: Feature data array (optional, used if problem_params is not provided).
        :param y: Target data array (optional, used if problem_params is not provided).
        :param dataset_name: Name of a predefined dataset to automatically load (optional).
        :param max_depth: Maximum depth of the trees for normalization.
        :param normalize_distance: Whether to normalize the distance calculation.
        :param outcome_weight_in_path: Weight of the outcome in the path distance calculation.
        """

        # Use provided problem_params or process data to obtain them
        if problem_params:
            self.problem_params = problem_params
        else:
            data_processor = DataProcessor(data=X, target=y, dataset_name=dataset_name)
            self.problem_params = data_processor.get_problem_params()

        # Check if input is list of paths or sklearn tree models
        paths_tree_1 = self._check_and_convert_paths(tree1)
        paths_tree_2 = self._check_and_convert_paths(tree2)
        # Identify larger and smaller tree
        self._find_larger_tree(paths_tree_1, paths_tree_2)
        
        # Distance calculation parameters
        if max_depth == None and hasattr(tree1, 'get_depth') and hasattr(tree2, 'get_depth'):
            self.max_depth = max(tree1.get_depth(), tree2.get_depth())  # Determine max_depth based on the trees
        elif max_depth == None:
            self.max_depth = 10
        else:
            self.max_depth = max_depth
        self.normalize_distance = normalize_distance
        self.outcome_weight_in_path = outcome_weight_in_path
        self.round_digits = round_digits

        # Matching outcome parameters
        self.matching_done = False
        self.matching = None
        self.distance = np.nan

    def _check_and_convert_paths(self, tree_or_paths):
        """
        Checks if input is a list of PathExtractor instances or a tree model.
        Converts dictionary paths to PathExtractor instances if necessary.
        """
        if isinstance(tree_or_paths, list) and all(isinstance(item, PathExtractor) for item in tree_or_paths):
            return tree_or_paths
        elif isinstance(tree_or_paths, list):
            # Convert dictionary paths to PathExtractor instances
            return [PathExtractor(**path_dict) for path_dict in tree_or_paths]
        else:
            return TreeParser(tree_or_paths, self.problem_params).get_paths()
        
    def _find_larger_tree(self, paths_tree_1, paths_tree_2):
        """
        Distinguish which tree has more paths and properly name trees for consistency
        """
        n_paths_1, n_paths_2 = len(paths_tree_1), len(paths_tree_2)
        if n_paths_2 > n_paths_1:
            self._n_paths_large, self._n_paths_small = (n_paths_2, n_paths_1)
            self._paths_large, self._paths_small = (paths_tree_2, paths_tree_1)
            self._tree_1_is_larger = False
        else:
            self._n_paths_large, self._n_paths_small = (n_paths_1, n_paths_2)
            self._paths_large, self._paths_small = (paths_tree_1, paths_tree_2)
            self._tree_1_is_larger = True

    def _compute_paths_distances(self):
        """
        Compute all pairwise path distances for input trees, including outcome weighting.
        """
        self._D = {}
        for i, path_i in enumerate(self._paths_large):
            for j, path_j in enumerate(self._paths_small):
                self._D[(i, j)] = compute_paths_distance(path_i, path_j, self.problem_params, self.outcome_weight_in_path)
    
    def _generate_lp_constraints(self):
        """
        Generates linear programming constraints for path matching allowing multiple matches
        for paths in the larger tree.
        """
        # Initialize the list for equality and inequality constraints
        A_eq, A_inq = [], []
        b_eq, b_inq = [], []
        bounds = []

        # Equality constraints ensuring each path in the smaller tree is matched exactly once
        for i in range(self._n_paths_large):
            row = [1 if j // self._n_paths_small == i else 0 for j in range(self._n_paths_small * self._n_paths_large)]
            A_eq.append(row)
            b_eq.append(1)

        # Inequality constraints allowing paths in the larger tree to match with multiple paths in the smaller tree
        for j in range(self._n_paths_small):
            col = [-1 if i % self._n_paths_small == j else 0 for i in range(self._n_paths_small * self._n_paths_large)]
            A_inq.append(col)
            b_inq.append(-1)

        bounds = [(0, None) for _ in range(self._n_paths_small * self._n_paths_large)]

        return A_eq, b_eq, A_inq, b_inq, bounds

    def _path_matching(self, print_solver_output=False):
        """
        Solves the path matching problem using linear programming, allowing for multiple
        matches from the larger tree to the smaller tree's paths.
        """
        # Create the cost vector for the LP problem
        c = [self._D.get((i, j), float('inf')) for i in range(self._n_paths_large) for j in range(self._n_paths_small)]
        
        # Generate constraints
        A_eq, b_eq, A_inq, b_inq, bounds = self._generate_lp_constraints()
        
        # Solve the LP with both equality and inequality constraints
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_inq, b_ub=b_inq, bounds=bounds, method='highs', 
                         options={'disp': print_solver_output})
        
        if not result.success:
            raise ValueError("LP didn't solve successfully.")
        
        self._soln = result.x
        self._objective = result.fun 

    def _decode_soln(self):
        """
        Decodes the solution from the linear programming output, identifying matched paths
        and calculating the total distance based on the matching.
        """
        self.matching = []
        distance = 0  # Reset distance to 0 before calculating
        for i, x in enumerate(self._soln):
            if x > 0.5:  # Assuming a threshold for binary decision
                path_1_index = i // self._n_paths_small
                path_2_index = i % self._n_paths_small
                matched_distance = self._D[(path_1_index, path_2_index)]
                distance += matched_distance
                if self._tree_1_is_larger:
                    self.matching.append((path_1_index, path_2_index))  
                else:
                    self.matching.append((path_2_index, path_1_index))  
        self._distance_unnormalized = distance
        self._distance_normalized = distance / (np.power(2, self.max_depth) * (2 * self.max_depth + self.outcome_weight_in_path))

    def compute_tree_distance(self, print_solver_output=False):
        """
        Computes the normalized distance between two decision trees.
        """
        self.matching_done = True
        # Compute path weights and distances between all pairs of paths
        self._compute_paths_distances()
        # Run path matching
        self._path_matching(print_solver_output=print_solver_output)
        # Decode solution
        self._decode_soln()
        if self.normalize_distance:
            self.distance = round(self._distance_normalized, ndigits=self.round_digits)
        else:
            self.distance = round(self._distance_unnormalized, ndigits=self.round_digits)
        return self.distance
