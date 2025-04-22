import unittest
import unittest
import numpy as np
from dt_distance.utils import determine_feature_type, determine_problem_type

class TestUtils(unittest.TestCase):
    def test_determine_feature_type_np(self):
        self.assertEqual(determine_feature_type(np.array([1, 2, 3, 4, 5])), 'numerical')
        self.assertEqual(determine_feature_type(np.array(['a', 'b', 'c', 'd', 'e'])), 'categorical')
        self.assertEqual(determine_feature_type(np.array([1, 2, 1, 2])), 'numerical')
        self.assertEqual(determine_feature_type(np.array([1.0, 2.0, 3.0])), 'numerical')

    def test_determine_problem_type_by_dtype(self):
        self.assertEqual(determine_problem_type(np.array([1, 0, 1, 0, 1], dtype=int)), 'classification')
        self.assertEqual(determine_problem_type(np.array([1.5, 2.3, 3.1], dtype=float)), 'regression')

if __name__ == '__main__':
    unittest.main()
