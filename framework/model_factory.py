from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.metrics import log_loss, ndcg_score, roc_auc_score
from tqdm.auto import tqdm

from .custom_logging import get_logger
from .config import Config


class Model:
    """Base model class"""
    
    def __init__(self, name, params=None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.logger = get_logger(self.__class__.__name__)
    
    def train(self, train_features, train_labels, cat_col_names=None):
        """Train model (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def predict(self, features):
        """Make predictions (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        return {}
    
    def save(self, filename):
        """Save model to file"""
        raise NotImplementedError
    
    @classmethod
    def load(cls, filename):
        """Load model from file"""
        raise NotImplementedError


class CatBoostModel(Model):
    """CatBoost model implementation"""
    
    def __init__(self, **params):
        super().__init__('catboost', params)
        
        from catboost import CatBoostClassifier
        
        self.params = params
        self.model = CatBoostClassifier(**self.params)
    
    def train(self, train_features, train_labels, eval_set=None, cat_col_names=None):
        """Train CatBoost model"""
        from catboost import Pool
        
        # Convert to pandas for CatBoost
        if isinstance(train_features, pl.DataFrame):
            train_features = train_features.to_pandas()
        
        if isinstance(train_labels, pl.Series):
            train_labels = train_labels.to_list()
        
        # Create pool
        train_pool = Pool(train_features, train_labels)
        
        # Prepare eval set if provided
        if eval_set is not None:
            eval_features, eval_labels = eval_set
            
            if isinstance(eval_features, pl.DataFrame):
                eval_features = eval_features.to_pandas()
            
            if isinstance(eval_labels, pl.Series):
                eval_labels = eval_labels.to_list()
                
            eval_pool = Pool(eval_features, eval_labels)
            
            # Train with evaluation
            self.model.fit(train_pool, eval_set=eval_pool, cat_features=cat_col_names)
        else:
            # Train without evaluation
            self.model.fit(train_pool, cat_features=cat_col_names)
        
        return self
    
    def predict(self, features):
        """Make predictions with CatBoost"""
        # Convert to pandas for CatBoost
        if isinstance(features, pl.DataFrame):
            features = features.to_pandas()
            
        # Predict probabilities for class 1
        return self.model.predict_proba(features)[:, 1]
    
    def get_feature_importance(self):
        """Get feature importance from CatBoost"""
        return dict(zip(self.model.feature_names_, self.model.feature_importances_))
    
    def save(self, filename):
        """Save model to file"""
        self.model.save_model(filename)
    
    @classmethod
    def load(cls, filename):
        """Load model from file"""
        from catboost import CatBoostClassifier
        
        model = cls()
        model.model = CatBoostClassifier()
        model.model.load_model(filename)
        return model


class LightGBMModel(Model):
    """LightGBM model implementation"""
    
    def __init__(self, **params):
        super().__init__('lightgbm', params)
        
        import lightgbm as lgb

        self.params = params
        self.model = None  # Will be created during training
    
    def train(self, train_features, train_labels, eval_set=None):
        """Train LightGBM model"""
        import lightgbm as lgb
        
        # Convert to pandas for LightGBM
        if isinstance(train_features, pl.DataFrame):
            train_features = train_features.to_pandas()
        
        if isinstance(train_labels, pl.Series):
            train_labels = train_labels.to_list()
        
        # Create dataset
        train_data = lgb.Dataset(train_features, label=train_labels)
        
        # Prepare eval set if provided
        eval_datasets = None
        if eval_set is not None:
            eval_features, eval_labels = eval_set
            
            if isinstance(eval_features, pl.DataFrame):
                eval_features = eval_features.to_pandas()
            
            if isinstance(eval_labels, pl.Series):
                eval_labels = eval_labels.to_list()
                
            eval_data = lgb.Dataset(eval_features, label=eval_labels)
            eval_datasets = [eval_data]
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=eval_datasets
        )
        
        return self
    
    def predict(self, features):
        """Make predictions with LightGBM"""
        # Convert to pandas for LightGBM
        if isinstance(features, pl.DataFrame):
            features = features.to_pandas()
            
        return self.model.predict(features)
    
    def get_feature_importance(self):
        """Get feature importance from LightGBM"""
        return dict(zip(self.model.feature_name(), self.model.feature_importance()))
    
    def save(self, filename):
        """Save model to file"""
        self.model.save_model(filename)
    
    @classmethod
    def load(cls, filename):
        """Load model from file"""
        import lightgbm as lgb
        
        model = cls()
        model.model = lgb.Booster(model_file=filename)
        return model


class ModelFactory:
    """Factory for creating and managing models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {
            'catboost': CatBoostModel,
            'lightgbm': LightGBMModel,
            # Add more models as needed
        }
    
    def create_model(self):
        """Create a model instance based on type"""
        model_type = self.config.get(('model', 'type'))
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Get model parameters from config
        model_params = self.config.get(('model', 'config', model_type))
        
        # Create and return model instance
        return self.models[model_type](**model_params)