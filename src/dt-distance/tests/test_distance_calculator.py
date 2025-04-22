import unittest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from dt_distance.distance_calculator import DistanceCalculator
from dt_distance.tree_parser import TreeParser
from dt_distance.path_extractor import PathExtractor
from dt_distance.data_processor import DataProcessor
from dt_distance.problem_params import ProblemParams
from sklearn.datasets import load_iris

class TestDistanceCalculator(unittest.TestCase):
    def setUp(self):
        # Create a simple problem for testing
        self.problem_params_simple = ProblemParams(
            feature_index=[0],
            feature_types={0: 'numerical'},
            feature_lower_bounds={0: 0},
            feature_upper_bounds={0: 10},
            feature_num_categories={},
            problem_type='classification',
            feature_names=['feature'],
            response_upper_bound=None,
            response_lower_bound=None,
        )
        self.problem_params_mixed = ProblemParams(
            feature_index=[0,1],
            feature_types={0: 'numerical', 1: 'categorical'},
            feature_lower_bounds={0: 0},
            feature_upper_bounds={0: 10},
            feature_num_categories={1: 2},
            problem_type='classification',
            feature_names=['feature'],
            response_upper_bound=None,
            response_lower_bound=None,
        )
        
    def test_distance_with_different_outcomes(self):
        # Setup for trees with paths leading to different outcomes
        paths_tree_1 = [
            PathExtractor([0], {'0': 'numerical'}, {}, {0: 0}, {0: 5}, {}, 0)
        ]
        paths_tree_2 = [
            PathExtractor([0], {'0': 'numerical'}, {}, {0: 0}, {0: 5}, {}, 1)
        ]
        distance_calculator = DistanceCalculator(paths_tree_1, paths_tree_2, 
                                                 problem_params=self.problem_params_simple,
                                                 normalize_distance=False
                                                 )
        calculated_distance = distance_calculator.compute_tree_distance()
        # Expected distance due to different outcomes
        expected_distance = 1  # Assuming outcome_weight=1 and normalized distance calculation is not considered for simplicity
        self.assertAlmostEqual(expected_distance, calculated_distance, msg="Distance with different outcomes should match expected.")

    def test_distance_with_higher_outcome_weight(self):
        # Setup for trees with paths leading to different outcomes
        paths_tree_1 = [
            PathExtractor([0], {0: 'numerical'}, {}, {0: 0}, {0: 5}, {}, 0)
        ]
        paths_tree_2 = [
            PathExtractor([0], {0: 'numerical'}, {}, {0: 0}, {0: 5}, {}, 1)
        ]
        distance_calculator = DistanceCalculator(paths_tree_1, paths_tree_2, 
                                                problem_params=self.problem_params_simple,
                                                outcome_weight_in_path=2,  # Setting higher outcome weight
                                                normalize_distance=False
                                                )
        calculated_distance = distance_calculator.compute_tree_distance()
        expected_distance = 2  # Since the outcome weight is doubled, the impact on distance should reflect this
        self.assertAlmostEqual(expected_distance, calculated_distance, msg="Distance with higher outcome weight should match expected.")

    def test_distance_with_mixed_feature_types(self):
        # Setup for trees with mixed feature types
        paths_tree_1 = [
            PathExtractor([0, 1], {0: 'numerical', 1: 'categorical'}, {1: 2}, {0: 0}, {0: 5}, {1: [False]}, 0)
        ]
        paths_tree_2 = [
            PathExtractor([0, 1], {0: 'numerical', 1: 'categorical'}, {1: 2}, {0: 3}, {0: 8}, {1: [False, True]}, 0)
        ]
        distance_calculator = DistanceCalculator(paths_tree_1, paths_tree_2, 
                                                problem_params=self.problem_params_mixed,
                                                normalize_distance=False
                                                )
        calculated_distance = distance_calculator.compute_tree_distance()
        expected_distance = 0.8 
        self.assertAlmostEqual(expected_distance, calculated_distance, msg="Distance with mixed feature types should match expected.")

    def test_distance_with_different_path_bounds(self):
        # Setup for trees with different path bounds but identical outcomes
        paths_tree_1 = [
            PathExtractor([0], {0: 'numerical'}, {}, {0: 0}, {0: 2}, {}, 0)
        ]
        paths_tree_2 = [
            PathExtractor([0], {0: 'numerical'}, {}, {0: 3}, {0: 5}, {}, 0)
        ]
        distance_calculator = DistanceCalculator(paths_tree_1, paths_tree_2, 
                                                problem_params=self.problem_params_simple,
                                                normalize_distance=False
                                                )
        calculated_distance = distance_calculator.compute_tree_distance()
        expected_distance = 0.3  
        self.assertAlmostEqual(expected_distance, calculated_distance, msg="Distance with different path bounds should match expected.")

    def test_identical_trees_with_different_normalization_depths(self):
        # Setup for trees with paths leading to identical outcomes
        paths_tree_1 = [
            PathExtractor([0], {0: 'numerical'}, {}, {0: 0}, {0: 5}, {}, 0),
            PathExtractor([0], {0: 'numerical'}, {}, {0: 5}, {0: 10}, {}, 1)
        ]
        paths_tree_2 = [
            PathExtractor([0], {0: 'numerical'}, {}, {0: 0}, {0: 5}, {}, 0),
            PathExtractor([0], {0: 'numerical'}, {}, {0: 5}, {0: 10}, {}, 1)
        ]
        distance_calculator = DistanceCalculator(paths_tree_1, paths_tree_2, 
                                                problem_params=self.problem_params_simple,
                                                normalize_distance=True,
                                                max_depth=10  # An arbitrary deeper depth for normalization
                                                )
        calculated_distance = distance_calculator.compute_tree_distance()
        # Expected distance should still be 0, as the trees are identical, but this checks the normalization process
        expected_distance = 0
        self.assertEqual(expected_distance, calculated_distance, msg="Distance for identical trees with different normalization depths should be 0.")


    def test_distance_between_identical_iris_trees(self):
        """Test that the distance between two identical Iris dataset trees is 0."""
        self.iris_data, self.iris_target = load_iris(return_X_y=True)
        self.iris_clf1 = DecisionTreeClassifier(max_depth=2,random_state=0)
        self.iris_clf1.fit(self.iris_data, self.iris_target)
        self.iris_clf2 = DecisionTreeClassifier(max_depth=2,random_state=0)
        self.iris_clf2.fit(self.iris_data, self.iris_target)
        problem_params_iris = ProblemParams(
            feature_index=list(range(4)),
            feature_types={i: 'numerical' for i in range(4)},
            feature_lower_bounds={i: np.min(self.iris_data[:, i]) for i in range(4)},
            feature_upper_bounds={i: np.max(self.iris_data[:, i]) for i in range(4)},
            feature_num_categories={},
            problem_type='classification'
        )
        distance_calculator = DistanceCalculator(self.iris_clf1, self.iris_clf2, problem_params=problem_params_iris)
        distance = distance_calculator.compute_tree_distance()
        self.assertEqual(distance, 0, "Distance between identical Iris trees should be 0.")

if __name__ == '__main__':
    unittest.main()
