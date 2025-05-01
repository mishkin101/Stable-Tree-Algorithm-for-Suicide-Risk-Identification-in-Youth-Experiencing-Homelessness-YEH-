import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

def prepare_data(df: pd.DataFrame, features: List[str], label: str, rng) -> Tuple:
    """
    Prepare data for training and testing by cleaning and splitting.
    
    Args:
        df: Input DataFrame
        features: List of feature column names
        label: Name of the target variable column
        
    Returns:
        Tuple containing X_full, y_full, X_train, X_test, y_train, y_test
    """
    df = df.replace('NaN', pd.NA)  # replace the string 'NaN' with actual NaN values
    df_full_cleaned = df[features + [label]].dropna().copy()
    X = df_full_cleaned[features]
    y = df_full_cleaned[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rng.integers(0, 2**32 - 1), stratify=y)
    X_full = df_full_cleaned[features]
    y_full = df_full_cleaned[label]
    return X_full, y_full, X_train, X_test, y_train, y_test


def down_sample():
    return



def random_train_split(X, y):
    """
    Create a random split of the data.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        X0, y0: Split features and targets
    """
    X_values = X.values if hasattr(X, 'values') else X
    y_values = y.values if hasattr(y, 'values') else y
    
    N = X_values.shape[0]
    indices = np.random.permutation(N)
    X0, y0 = X_values[indices[:N // 2]], y_values[indices[:N // 2]]
    return X0, y0