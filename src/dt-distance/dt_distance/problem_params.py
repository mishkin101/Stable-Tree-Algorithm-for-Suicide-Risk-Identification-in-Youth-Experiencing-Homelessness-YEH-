import numpy as np

class ProblemParams:
    """
    Stores parameters relevant to the decision tree problem, including feature information
    and problem type (classification or regression).
    """
    def __init__(self, feature_index, feature_types, feature_lower_bounds, feature_upper_bounds,
                 feature_num_categories, problem_type, feature_names=None, response_upper_bound=None,
                 response_lower_bound=None):
        self.feature_index = feature_index
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.feature_lower_bounds = feature_lower_bounds
        self.feature_upper_bounds = feature_upper_bounds
        self.feature_num_categories = feature_num_categories
        self.problem_type = problem_type
        self.response_upper_bound = response_upper_bound
        self.response_lower_bound = response_lower_bound