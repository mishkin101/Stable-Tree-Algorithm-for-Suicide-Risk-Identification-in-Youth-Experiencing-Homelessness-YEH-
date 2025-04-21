import unittest
import numpy as np
import pandas as pd
from sklearn import datasets

from dt_distance.problem_params import ProblemParams  
from dt_distance.utils import determine_feature_type, determine_problem_type
from dt_distance.data_processor import DataProcessor 

class TestDataProcessor(unittest.TestCase):
    def test_predefined_dataset_loading(self):
        """Test loading of predefined datasets and parameter setup."""
        # Use breast_cancer dataset as an example
        processor = DataProcessor(dataset_name='breast_cancer')
        problem_params = processor.get_problem_params()

        self.assertIsNotNone(problem_params)
        self.assertEqual(problem_params.problem_type, 'classification')
        self.assertTrue(len(problem_params.feature_names) > 0)
        self.assertTrue(problem_params.feature_names[0] == 'mean radius')
        self.assertTrue(len(problem_params.feature_types) > 0)

    def test_custom_data_processing(self):
        """Test processing of custom data arrays."""
        # Generate custom data
        data = np.array([[1, 2.5], [2, 3.6], [1, 2.7], [2, 3.8]])
        target = np.array([0, 1, 0, 1])  # Binary target for classification
        feature_names = ['feature_1', 'feature_2']

        processor = DataProcessor(data=data, target=target, feature_names=feature_names)
        problem_params = processor.get_problem_params()

        self.assertEqual(problem_params.problem_type, 'classification')
        self.assertEqual(len(problem_params.feature_types), 2)
        self.assertIn('feature_1', problem_params.feature_names)
        self.assertIn('feature_2', problem_params.feature_names)

        # Check if feature types are correctly determined
        for f_type in problem_params.feature_types.values():
            self.assertIn(f_type, ['numerical', 'categorical'])

if __name__ == '__main__':
    unittest.main()
