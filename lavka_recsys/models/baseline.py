import numpy as np
import pickle
import polars as pl
import pandas as pd

from .base import Model

class RandomModel(Model):
    """Model outputing random values (baseline)"""

    def __init__(self, **params):
        super().__init__('random_baseline')
    
    def train(self, *args, **kwargs):
        pass

    def predict(self, features, *args, **kwargs):
        return np.random.rand(len(features),)
    
    def get_feature_importance(self):
        return {}
    
    def save(self, *args, **kwargs):
        pass

    @classmethod
    def load(cls, *args, **kwargs):
        return cls()
    

class SingleFeatureModel(Model):
    """Model sorting items according to a single feature (e.g. popularity)"""

    def __init__(self, feature_name, desc=True, **kwargs):
        super().__init__('single_feature')
        self.feature_name = feature_name
        self.desc = desc

    def train(self, *args, **kwargs):
        pass
        
    def predict(self, features: pd.DataFrame | pl.DataFrame, *args, **kwargs):
        if isinstance(features, pd.DataFrame):
            scores = features[self.feature_name].fillna(-np.inf).tolist()
        elif isinstance(features, pl.DataFrame):
            series = features.get_column(self.feature_name)
            # Replace nulls and NaNs
            min_val = series.filter(series.is_not_null()).min()
            series = series.fill_null(min_val).fill_nan(min_val)
            scores = series.to_list()
        else:
            raise TypeError("Unsupported DataFrame type")
        
        if not self.desc:
            scores = [-sc for sc in scores]
        return scores
    
    def get_feature_importance(self):
        return {self.feature_name: 1}
    
    def save(self, filename):
        data = {
            'feature_name': self.feature_name,
            'desc': self.desc
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return cls(**data)
