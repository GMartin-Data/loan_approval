"""
Cyclical encoding for date features preprocessing
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Meant to encode time data with cycles (days of week, month...)"""
    def __init__(self, column_name, cycle_length):
        self.column_name = column_name
        self.cycle_length = cycle_length
    
    def fit(self, X, y=None):
        # No fitting needed, implemented for compatibility with sklearn's API
        return self
        
    def transform(self, X, y=None):
        # Apply cyclical encoding directly without needing to fit
        X = X.copy()
        values = X[self.column_name]
        # Create the cyclical features
        X[f'{self.column_name}_sin'] = np.sin(2 * np.pi * values / self.cycle_length)
        X[f'{self.column_name}_cos'] = np.cos(2 * np.pi * values / self.cycle_length)
        # Drop the original column
        X.drop(columns=[self.column_name], inplace=True)
        return X
    
    def get_feature_names_out(self, input_features=None):
        # Generate names for the output features
        return np.array(
          [f'{self.column_name}_sin', f'{self.column_name}_cos'], dtype=object
        )
