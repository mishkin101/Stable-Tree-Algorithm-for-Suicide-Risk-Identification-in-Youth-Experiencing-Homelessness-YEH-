from copy import deepcopy
import numpy as np
from .path_extractor import PathExtractor

class TreeParser:
    """
    Class to convert a decision tree into a format suitable for path extraction and analysis,
    storing each path from root to leaf.
    """
    
    def __init__(self, tree, problem):
        """
        Initializes the TreeConverter with a decision tree and problem parameters.
        
        :param tree: The decision tree model (assumed to be from sklearn).
        :param problem: An instance containing problem parameters.
        """
        self.tree = tree.tree_  # Access the underlying tree structure from sklearn's DecisionTreeClassifier/Regressor
        self.problem = problem
        self.paths = self._parse_tree()

    def _parse_tree(self):
        """
        Parses the decision tree, converting it into a series of path objects.
        
        :return: A list of PathExtractor objects, each representing a unique path from root to leaf.
        """
        feature_index = self.problem.feature_index
        feature_types = self.problem.feature_types
        feature_num_categories = self.problem.feature_num_categories
        feature_lower_bounds = self.problem.feature_lower_bounds
        feature_upper_bounds = self.problem.feature_upper_bounds

        stack = [0]  # Start with the root node
        open_paths = []  # Paths that are being explored
        finalized_paths = []  # Completed paths from root to leaf

        # Initialize the first path with the root node's details
        open_paths.append(PathExtractor(
            feature_index=feature_index,
            feature_types=feature_types,
            feature_num_categories=feature_num_categories,
            path_lower_bounds={f: feature_lower_bounds[f] for f in feature_index if feature_types[f] == 'numerical'},
            path_upper_bounds={f: feature_upper_bounds[f] for f in feature_index if feature_types[f] == 'numerical'},
            path_categories={f: [True for _ in range(feature_num_categories[f])] for f in feature_index if feature_types[f] == 'categoric'},
            path_outcome=np.argmax(self.tree.value[0])
        ))

        while stack:
            node_id = stack.pop(0)
            path = open_paths.pop(0)

            if self.tree.children_left[node_id] == -1:  # Check if node is leaf
                path.add_leaf_node(node_id, np.argmax(self.tree.value[node_id]))
                finalized_paths.append(path)
            else:
                for child, direction in [(self.tree.children_left[node_id], 'left'), (self.tree.children_right[node_id], 'right')]:
                    child_path = deepcopy(path)
                    child_path.add_split_node(
                        node_id=node_id,
                        split_feature_id=self.tree.feature[node_id],
                        parallel_split=True,  # sklearn decision trees only have binary splits
                        split=self.tree.threshold[node_id],
                        direction=direction
                    )
                    open_paths.append(child_path)
                    stack.append(child)

        return finalized_paths

    def get_paths(self):
        """
        Returns the list of paths extracted from the decision tree.
        
        :return: A list of PathExtractor objects.
        """
        return self.paths
