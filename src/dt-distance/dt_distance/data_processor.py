import numpy as np
import pandas as pd
from sklearn import datasets
from .problem_params import ProblemParams
from .utils import determine_feature_type, determine_problem_type

class DataProcessor:
    """
    Processes input data for decision tree analysis, automatically handling feature ranges,
    names, and problem parameters. Supports loading predefined datasets or custom data arrays.
    """

    def __init__(self, data=None, target=None, feature_names=None, feature_types=None,
                 dataset_name=None):
        """
        Initializes the DataProcessor with either a predefined dataset name or custom data.
        
        :param data: Custom feature data array.
        :param target: Custom target data array.
        :param feature_names: Optional list of feature names for custom data.
        :param feature_types: Optional dict specifying the type ('numerical' or 'categorical')
                              for each feature in custom data.
        :param dataset_name: Predefined dataset name for automatic loading ('breast_cancer', etc.).
        """
        self.data = data
        self.target = target
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.dataset_name = dataset_name
        self.problem_type = None
        self.problem_params = None
        
        if dataset_name:
            self._load_predefined_dataset()
        else:
            self._process_custom_data()

    def _load_predefined_dataset(self):
        """
        Loads predefined datasets from sklearn and prepares problem parameters.
        """
        if self.dataset_name == 'breast_cancer':
            dataset = datasets.load_breast_cancer()
            self.data, self.target = dataset.data, dataset.target
            self.feature_names = dataset.feature_names
            self.feature_types = {i: 'numerical' for i in range(self.data.shape[1])} 
            self.problem_type = 'classification'
        # Add other predefined datasets here as elif blocks
            
        else:
            raise ValueError("Unsupported predefined dataset.")
        
        self._setup_problem_params()

    def _process_custom_data(self):
        """
        Processes custom data provided by the user, setting up problem parameters.
        """
        if self.data is None or self.target is None:
            raise ValueError("Custom data and target must be provided if no dataset name is given.")
        
        # Determine problem type based on the target array uniqueness
        self.problem_type = determine_problem_type(self.target) 
        
        # Automatically determine feature types if not provided
        if not self.feature_types:
            self.feature_types = {i: determine_feature_type(self.data[:, i]) for i in range(self.data.shape[1])}
        
        self._setup_problem_params()

    def _setup_problem_params(self):
        """
        Sets up problem parameters, including feature ranges and names.
        """
        feature_lower_bounds, feature_upper_bounds, feature_num_categories = {}, {}, {}
        for i, f_type in self.feature_types.items():
            if f_type == 'numerical':
                feature_lower_bounds[i] = np.min(self.data[:, i])
                feature_upper_bounds[i] = np.max(self.data[:, i])
            else:  # 'categorical'
                feature_num_categories[i] = len(np.unique(self.data[:, i]))
        
        self.problem_params = ProblemParams(
            feature_index=list(range(self.data.shape[1])),
            feature_types=self.feature_types,
            feature_lower_bounds=feature_lower_bounds,
            feature_upper_bounds=feature_upper_bounds,
            feature_num_categories=feature_num_categories,
            problem_type=self.problem_type,
            feature_names=self.feature_names,
            response_upper_bound=np.max(self.target) if self.problem_type == 'regression' else None,
            response_lower_bound=np.min(self.target) if self.problem_type == 'regression' else None
        )

    def get_problem_params(self):
        """
        Returns the problem parameters after data processing.
        
        :return: An instance of ProblemParams containing all necessary information for decision tree analysis.
        """
        return self.problem_params
