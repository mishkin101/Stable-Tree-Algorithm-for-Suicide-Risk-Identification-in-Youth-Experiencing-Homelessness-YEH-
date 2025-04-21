import numpy as np

class PathExtractor:
    """
    Extracts and manipulates paths within a decision tree, allowing for the calculation
    of path weights and distances between paths, which are essential for decision tree
    distance calculations.
    """
    
    def __init__(self, feature_index, feature_types, feature_num_categories, 
                 path_lower_bounds, path_upper_bounds, path_categories, path_outcome):
        """
        Initializes a PathExtractor instance with detailed path and feature information.
        
        :param feature_index: List of feature indices.
        :param feature_types: Dictionary mapping feature index to feature type (numerical or categoric).
        :param feature_num_categories: Dictionary mapping categoric feature index to number of categories.
        :param path_lower_bounds: Dictionary of numeric feature index to feature lower bound across path.
        :param path_upper_bounds: Dictionary of numeric feature index to feature upper bound across path.
        :param path_categories: Dictionary of categoric feature index to feature categories across path.
        :param path_outcome: Path outcome (real value or class number for regression or classification respectively).
        """
        self.feature_index = feature_index
        self.feature_types = feature_types
        self.feature_num_categories = feature_num_categories
        self.path_lower_bounds = path_lower_bounds
        self.path_upper_bounds = path_upper_bounds
        self.path_categories = path_categories
        self.path_outcome = path_outcome
        self.path_node_ids = []
        
    def add_split_node(self, node_id, split_feature_id, parallel_split, split, direction):
        """
        Adds a split node to the path, updating bounds or categories based on the split type and direction.
        
        :param node_id: ID of the node being added.
        :param split_feature_id: Feature index of the split.
        :param parallel_split: Boolean indicating if the split is parallel (True) or categoric (False).
        :param split: Split value or category list, depending on split type.
        :param direction: Direction of the split ('left' or 'right').
        """
        self.path_node_ids.append(node_id)
        
        if parallel_split:  # For numerical features
            if direction == 'left':
                self.path_upper_bounds[split_feature_id] = split
            else:
                self.path_lower_bounds[split_feature_id] = split
        else:  # For categoric features
            if direction == 'left':
                self.path_categories[split_feature_id] = split
            else:
                self.path_categories[split_feature_id] = [not i for i in split]
                    
    def add_leaf_node(self, node_id, node_outcome):
        """
        Adds a leaf node to the path, setting the final outcome of the path.
        
        :param node_id: ID of the leaf node.
        :param node_outcome: Outcome associated with the leaf node.
        """
        self.path_node_ids.append(node_id)
        self.path_outcome = node_outcome

def compute_paths_distance(path_1, path_2, problem_params, outcome_weight=1):
    """
    Computes the distance between two paths, considering both feature bounds and outcomes.

    :param path_1: The first path object.
    :param path_2: The second path object.
    :param problem_params: An instance of ProblemParams containing problem specifications.
    :param outcome_weight: Weight assigned to the outcome difference in the distance computation.
    :return: The computed distance between the two paths.
    """
    d = 0
    for j in problem_params.feature_index:
        if problem_params.feature_types[j] == 'numerical':
            if j in path_1.path_upper_bounds and j in path_2.path_upper_bounds and j in path_1.path_lower_bounds and j in path_2.path_lower_bounds:
                u = abs(path_1.path_upper_bounds[j] - path_2.path_upper_bounds[j])
                l = abs(path_1.path_lower_bounds[j] - path_2.path_lower_bounds[j])
                d += (u + l) / (problem_params.feature_upper_bounds[j] - problem_params.feature_lower_bounds[j]) / 2
        else:  # problem_params.feature_types[j] == 'categoric':
            unique_categories_1 = set(path_1.path_categories[j])
            unique_categories_2 = set(path_2.path_categories[j])
            # Calculate the proportion of categories that are different
            d += (len(unique_categories_1.union(unique_categories_2)) - len(unique_categories_1.intersection(unique_categories_2))) / problem_params.feature_num_categories[j]
    if problem_params.problem_type == 'classification':
        d += outcome_weight * 1 * (path_1.path_outcome != path_2.path_outcome)
    else:  # problem_params.problem_type == 'regression':
        d += outcome_weight * abs(path_1.path_outcome - path_2.path_outcome) / (problem_params.response_upper_bound - problem_params.response_lower_bound)
    return d
