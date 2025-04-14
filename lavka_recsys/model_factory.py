from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dpath.util
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
    
    def train(self, train_features, train_labels, cat_columns=None, **kwargs):
        """
        Train model (to be implemented by subclasses).
        Args:
            train_features (pd.DataFrame or pl.DataFrame): Training features.
            train_labels (pd.Series or pl.Series): Training labels.
            cat_columns (list): Categorical column names.
            kwargs: Additional parameters for training.
        """
        raise NotImplementedError
    
    def predict(self, features, **kwargs):
        """
        Make relevance predictions (to be implemented by subclasses).
        Args:
            features (pd.DataFrame or pl.DataFrame): Features for prediction.
            kwargs: Additional parameters for prediction.
        """
        raise NotImplementedError
    
    def get_feature_importance(self):
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


class CatBoostModel(Model):
    """CatBoost model implementation"""
    
    def __init__(self, **params):
        super().__init__('catboost', params)
        from catboost import CatBoostClassifier
        
        self.params = params
        self.model = CatBoostClassifier(**self.params)
    
    def train(self, train_features, train_labels, eval_set=None, cat_columns=None):
        """Train CatBoost model.
        Args:
            train_features (pd.DataFrame or pl.DataFrame): Training features.
            train_labels (pd.Series or pl.Series): Training labels.
            eval_set (tuple): Tuple of evaluation features and labels.
            cat_columns (list): Categorical column names.
        """
        from catboost import Pool
        import pandas as pd
    
        # Convert to pandas for CatBoost
        if isinstance(train_features, pl.DataFrame):
            train_features = train_features.to_pandas()
        
        if isinstance(train_labels, pl.Series):
            train_labels = train_labels.to_list()
        
        # Convert categorical columns to string type to avoid CatBoost errors with floats
        if cat_columns:
            safe_cat_columns = []
            for col in cat_columns:
                if col in train_features.columns:
                    # Convert category to string to avoid float/int issues
                    self.logger.info(f"Converting categorical column '{col}' to string type")
                    train_features[col] = train_features[col].astype(str)
                    safe_cat_columns.append(col)
                else:
                    self.logger.warning(f"Categorical column '{col}' not found in features")
            
            # Use only the columns that actually exist
            cat_columns = safe_cat_columns
        
        self.logger.info(
            "Training CatBoost model with columns: "
            f"{train_features.columns.tolist()} "
            f"(cat_columns: {cat_columns})"
        )
        
        # Create pool
        train_pool = Pool(
            train_features, train_labels,
            cat_features=cat_columns
        )
        
        # Prepare eval set if provided
        eval_pool = None
        if eval_set is not None:
            eval_features, eval_labels = eval_set
            
            if isinstance(eval_features, pl.DataFrame):
                eval_features = eval_features.to_pandas()
            
            if isinstance(eval_labels, pl.Series):
                eval_labels = eval_labels.to_list()
            
            # Also convert categorical columns in evaluation set
            if cat_columns:
                for col in cat_columns:
                    if col in eval_features.columns:
                        eval_features[col] = eval_features[col].astype(str)
            
            eval_pool = Pool(eval_features, eval_labels, cat_features=cat_columns)
        
        # Check if model already has trees (for incremental learning)
        tree_count = getattr(self.model, 'tree_count_', None)
        if tree_count is None:
            # If tree_count_ attribute doesn't exist, try a different approach
            # Check if the model has already been fitted
            init_model = None
            try:
                # This will only succeed if the model has been fitted
                if hasattr(self.model, 'get_tree_count') and self.model.get_tree_count() > 0:
                    tree_count = self.model.get_tree_count()
                    init_model = self.model
                else:
                    tree_count = 0
                    init_model = None
            except Exception:
                # If any exception occurs, assume model is not trained
                tree_count = 0
                init_model = None
        else:
            # If tree_count_ attribute exists, use it to determine if we should use init_model
            init_model = self.model if tree_count > 0 else None
        
        # For CatBoost, we need to create a fresh model for each training session
        # since there's a bug with init_model that can cause feature type inconsistencies
        from catboost import CatBoostClassifier
        
        # Create a new instance with the same params
        self.model = CatBoostClassifier(**self.params)
        
        self.logger.info("Training CatBoost model from scratch")
        # Train with evaluation
        if eval_pool is not None:
            self.model.fit(train_pool, eval_set=eval_pool)
        else:
            self.model.fit(train_pool)
            
        # Log the number of trees
        try:
            tree_count = self.model.get_tree_count()
            self.logger.info(f"Model now has {tree_count} trees")
        except:
            self.logger.info("Could not determine tree count")
        
        return self
    
    def predict(self, features):
        """Make probability predictions with CatBoost"""
        # Convert to pandas for CatBoost
        if isinstance(features, pl.DataFrame):
            features = features.to_pandas()
            
        # Save a copy of the column names before any transformations
        original_columns = list(features.columns)
        
        # Make a copy of the dataframe to avoid modifying the original
        features_for_pred = features.copy()
        
        # Handle categorical features
        try:
            # First try to get categorical feature indices from the model
            feature_names = self.model.feature_names_
            cat_features = getattr(self.model, 'get_cat_feature_indices', lambda: [])()
            
            # If we have categorical features, convert them to strings
            if cat_features:
                # Get the actual categorical column names
                cat_columns = [feature_names[i] for i in cat_features if i < len(feature_names)]
                
                # Convert them to string
                for col in cat_columns:
                    if col in features_for_pred.columns:
                        self.logger.info(f"Converting categorical column for prediction: {col}")
                        features_for_pred[col] = features_for_pred[col].astype(str)
            
        except Exception as e:
            # If there was any error getting categorical features from the model,
            # try a more aggressive approach - convert all object/string columns to strings
            self.logger.warning(f"Error detecting categorical columns from model: {str(e)}")
            self.logger.info("Converting all potential categorical columns to strings")
            
            for col in features_for_pred.columns:
                # Convert any object or string columns to strings
                if features_for_pred[col].dtype == 'object' or 'str' in str(features_for_pred[col].dtype).lower():
                    self.logger.info(f"Converting potential categorical column: {col}")
                    features_for_pred[col] = features_for_pred[col].fillna('').astype(str)
        
        # Ensure we have the exact same columns the model was trained on
        try:
            # If the model has feature_names_ attribute, use it to ensure column order
            if hasattr(self.model, 'feature_names_'):
                # Check for missing columns
                missing_cols = set(self.model.feature_names_) - set(features_for_pred.columns)
                if missing_cols:
                    self.logger.warning(f"Missing columns in prediction data: {missing_cols}")
                    # Add missing columns as nulls
                    for col in missing_cols:
                        features_for_pred[col] = None
                        
                # Check for extra columns
                extra_cols = set(features_for_pred.columns) - set(self.model.feature_names_)
                if extra_cols:
                    self.logger.warning(f"Extra columns in prediction data will be ignored: {extra_cols}")
                    # Only keep needed columns
                    features_for_pred = features_for_pred[self.model.feature_names_]
                    
                # Ensure correct column order
                features_for_pred = features_for_pred[self.model.feature_names_]
                
        except Exception as e:
            self.logger.warning(f"Error aligning prediction columns: {str(e)}")
            # If we can't match columns exactly, just proceed with what we have
        
        try:
            # Create a CatBoost Pool for prediction
            from catboost import Pool
            
            # Get categorical feature indices again
            cat_features = getattr(self.model, 'get_cat_feature_indices', lambda: [])()
            if cat_features:
                cat_columns = [feature_names[i] for i in cat_features if i < len(feature_names)]
                pred_pool = Pool(features_for_pred, cat_features=cat_columns)
                self.logger.info(f"Created prediction pool with categorical features: {cat_columns}")
            else:
                pred_pool = Pool(features_for_pred)
                
            # Make predictions using the pool
            return self.model.predict_proba(pred_pool)[:, 1]
            
        except Exception as e:
            self.logger.error(f"Error creating Pool for prediction: {str(e)}")
            self.logger.info("Falling back to direct prediction")
            
            # As a last resort, try direct prediction
            return self.model.predict_proba(features_for_pred)[:, 1]
    
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
        self.logger = get_logger(self.__class__.__name__)
    
    def create_model(self, override_params=None) -> Model:
        """Create a model instance based on type"""
        model_type = self.config.get(('model', 'type'))
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Get model parameters from config
        model_params = self.config.get(('model', 'config', model_type))
        if override_params:
            model_params.update(override_params)
    
        # Create and return model instance
        self.logger.info(f"Creating {model_type} model with params: {model_params}")
        return self.models[model_type](**model_params)