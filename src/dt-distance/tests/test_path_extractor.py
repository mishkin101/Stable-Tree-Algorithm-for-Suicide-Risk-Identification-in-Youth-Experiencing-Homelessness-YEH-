import unittest
import numpy as np

from dt_distance.path_extractor import PathExtractor, compute_paths_distance
from dt_distance.problem_params import ProblemParams

class TestPathExtractor(unittest.TestCase):
    def setUp(self):
        # Setup for testing
        self.feature_index = [0, 1]
        self.feature_types = {0: 'numerical', 1: 'categorical'}
        self.feature_num_categories = {1: 2}
        self.path_lower_bounds = {0: 0}
        self.path_upper_bounds = {0: 5}
        self.path_categories = {1: [True, False]}
        self.path_outcome = 1
        self.problem_params = ProblemParams(
            feature_index=self.feature_index,
            feature_types=self.feature_types,
            feature_lower_bounds={0: 0},
            feature_upper_bounds={0: 10},
            feature_num_categories=self.feature_num_categories,
            problem_type='classification',
            feature_names=['feature_0', 'feature_1']
        )

    def test_add_split_node(self):
        # Test adding a split node
        path = PathExtractor(
            self.feature_index,
            self.feature_types,
            self.feature_num_categories,
            self.path_lower_bounds,
            self.path_upper_bounds,
            self.path_categories,
            self.path_outcome
        )
        path.add_split_node(node_id=1, split_feature_id=0, parallel_split=True, split=2.5, direction='left')
        self.assertEqual(path.path_upper_bounds[0], 2.5)

    def test_add_leaf_node(self):
        # Test adding a leaf node
        path = PathExtractor(
            self.feature_index,
            self.feature_types,
            self.feature_num_categories,
            self.path_lower_bounds,
            self.path_upper_bounds,
            self.path_categories,
            self.path_outcome
        )
        path.add_leaf_node(node_id=2, node_outcome=2)
        self.assertEqual(path.path_outcome, 2)

    def test_distance_numerical_features(self):
        # Expected distance calculation for numerical feature (0.5 for feature 0) + (1 for different outcomes)
        expected_distance = 0.5 + 1
        path1 = PathExtractor(self.feature_index, self.feature_types, self.feature_num_categories, 
                              {0: 0}, {0: 5}, {1: [True, False]}, 0)
        path2 = PathExtractor(self.feature_index, self.feature_types, self.feature_num_categories, 
                              {0: 5}, {0: 10}, {1: [False, True]}, 1)
        computed_distance = compute_paths_distance(path1, path2, self.problem_params, outcome_weight=1)
        self.assertAlmostEqual(computed_distance, expected_distance, places=2)

    def test_distance_categorical_features(self):
        # Expected distance calculation for categorical feature (0.5 for feature 1) + (1 for different outcomes)*(2 outcome w)
        expected_distance = 0.5 + 2*1
        path1 = PathExtractor(self.feature_index, self.feature_types, self.feature_num_categories, 
                              {}, {}, {1: [True]}, 0)
        path2 = PathExtractor(self.feature_index, self.feature_types, self.feature_num_categories, 
                              {}, {}, {1: [False, True]}, 1)
        computed_distance = compute_paths_distance(path1, path2, self.problem_params, outcome_weight=2)
        self.assertAlmostEqual(computed_distance, expected_distance, places=2)

if __name__ == '__main__':
    unittest.main()
