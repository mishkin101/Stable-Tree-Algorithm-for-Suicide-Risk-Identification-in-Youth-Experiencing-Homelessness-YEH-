import numpy as np

def determine_feature_type(column):
    """
    Determines if a NumPy array column should be considered 'categorical' or 'numerical'.
    
    Args:
    - column: A NumPy array column (1D array).
    
    Returns:
    - A string indicating the feature type ('categorical' or 'numerical').
    """
    try:
        # Attempt to convert the column to floats
        _ = column.astype(float)
        return 'numerical'
    except ValueError:
        # If conversion fails, consider it categorical
        return 'categorical'
    
# def determine_feature_type_np(column, max_categories=0):
#     """
#     Determines if a NumPy array column should be considered 'categorical' or 'numerical'.
    
#     Args:
#     - column: A NumPy array column (1D array).
#     - max_categories: Maximum number of unique values for a feature to be considered 'categorical'.
    
#     Returns:
#     - A string indicating the feature type ('categorical' or 'numerical').
#     """
#     # Check if column is of string type or object type
#     if column.dtype.kind == 'O' or column.dtype.kind == 'U':
#         return 'categorical'
#     # Check for number of unique values and integer type
#     elif len(np.unique(column)) <= max_categories and column.dtype.kind in 'biu':
#         return 'categorical'
#     else:
#         return 'numerical'
    
def determine_problem_type(target, max_classes=2):
    """
    Determines if a problem should be considered 'classification' or 'regression'.
    
    Args:
    - target: A NumPy array representing the target variable.
    
    Returns:
    - A string indicating the problem type ('classification' or 'regression').
    """
    if np.issubdtype(target.dtype, np.floating) and len(np.unique(target)) > max_classes:
        return 'regression'
    else:
        return 'classification'