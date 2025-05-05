from typing import Iterable
from ..utils.custom_logging import get_logger


class Model:
    """Base model class"""
    
    def __init__(self, name, params=None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.logger = get_logger(self.__class__.__name__)
    
    def train(self, train_features, train_labels, *, cat_columns=None, train_group_ids=None, **kwargs):
        """
        Train model (to be implemented by subclasses).
        Args:
            train_features (pd.DataFrame or pl.DataFrame): Training features.
            train_labels (pd.Series or pl.Series): Training labels.
            train_group_ids (pd.Series or pl.Series): Request IDs for ranking models.
            cat_columns (list): Categorical column names.
            kwargs: Additional parameters for training.
        """
        raise NotImplementedError
    
    def predict(self, features, *, cat_columns=None, group_ids=None, **kwargs) -> Iterable[float]:
        """
        Make relevance predictions (to be implemented by subclasses).
        Args:
            features (pd.DataFrame or pl.DataFrame): Features for prediction.
            kwargs: Additional parameters for prediction.
        Returns:
            an iterable of scores
        """
        raise NotImplementedError
    
    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance if available.
        Returns:
            dict: Feature importance scores.
        """
        return {}
    
    def save(self, filename):
        """Save model to file"""
        raise NotImplementedError
    
    @classmethod
    def load(cls, filename):
        """Load model from file"""
        raise NotImplementedError
