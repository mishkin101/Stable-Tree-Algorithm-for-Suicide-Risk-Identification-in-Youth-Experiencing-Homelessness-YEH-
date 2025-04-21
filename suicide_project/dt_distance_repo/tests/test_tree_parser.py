import unittest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from dt_distance.tree_parser import TreeParser
from dt_distance.problem_params import ProblemParams

class TestTreeParser(unittest.TestCase):
    def setUp(self):
        # Create a simple decision tree for testing
        X = [[0, 0], [1, 1], [2, 2]]
        y = [0, 1, 1]
        self.clf = DecisionTreeClassifier(max_depth=2)
        self.clf.fit(X, y)
        
        # Problem parameters mimicking the features in X
        self.problem_params = ProblemParams(
            feature_index=[0, 1],
            feature_types={0: 'numerical', 1: 'numerical'},
            feature_lower_bounds={0: 0, 1: 0},
            feature_upper_bounds={0: 2, 1: 2},
            feature_num_categories={},
            problem_type='classification'
        )

    def test_tree_parsing(self):
        # Parse the tree and retrieve paths
        tree_parser = TreeParser(self.clf, self.problem_params)
        paths = tree_parser.get_paths()
        
        # Verify that paths were created
        self.assertTrue(len(paths) > 0)
        
        # Check properties of a path
        path = paths[0]
        self.assertIn(0, path.feature_index)  # Check if the path includes feature 0
        self.assertIn(1, path.feature_index)  # Check if the path includes feature 1
        
    def test_iris_dataset_parsing(self):
        # Load the Iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Train a decision tree classifier on the Iris dataset
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        
        # Define problem parameters for the Iris dataset
        problem_params = ProblemParams(
            feature_index=list(range(X.shape[1])),
            feature_types={i: 'numerical' for i in range(X.shape[1])},  # All features in Iris are numerical
            feature_lower_bounds={i: np.min(X[:, i]) for i in range(X.shape[1])},
            feature_upper_bounds={i: np.max(X[:, i]) for i in range(X.shape[1])},
            feature_num_categories={},  # No categorical features in Iris
            problem_type='classification'  # Iris is a classification problem
        )
        
        # Parse the tree
        tree_parser = TreeParser(clf, problem_params)
        paths = tree_parser.get_paths()
        
        # Assertions to verify the parsing
        self.assertTrue(len(paths) > 0, "Paths should be extracted from the decision tree.")
        
        # Verifying the structure of one of the paths
        path = paths[0]
        self.assertTrue(len(path.path_node_ids) > 0, "Each path should have node IDs.")
        self.assertTrue(0 in path.feature_index, "Feature 0 should be present in the path.")
        self.assertTrue(path.path_outcome in y, "The path outcome should be a valid class label.")
        
        
if __name__ == '__main__':
    unittest.main()
