import numpy as np

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
