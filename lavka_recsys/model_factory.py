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
    
    def train(self, train_features, train_labels, cat_columns=None, train_request_ids=None, **kwargs):
        """
        Train model (to be implemented by subclasses).
        Args:
            train_features (pd.DataFrame or pl.DataFrame): Training features.
            train_labels (pd.Series or pl.Series): Training labels.
            cat_columns (list): Categorical column names.
            train_request_ids (pd.Series or pl.Series): Request IDs for ranking models.
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
    
    def train(self, train_features, train_labels, eval_set=None, cat_columns=None, train_request_ids=None):
        """Train CatBoost model.
        Args:
            train_features (pd.DataFrame or pl.DataFrame): Training features.
            train_labels (pd.Series or pl.Series): Training labels.
            eval_set (tuple): Tuple of (eval_features, eval_labels, eval_request_ids).
            cat_columns (list): Categorical column names.
            train_request_ids (pd.Series or pl.Series): Request IDs for training data.
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
            # Handle both formats: (features, labels) or (features, labels, request_ids)
            if len(eval_set) >= 2:
                eval_features, eval_labels = eval_set[0], eval_set[1]
                eval_request_ids = eval_set[2] if len(eval_set) > 2 else None
                
                if isinstance(eval_features, pl.DataFrame):
                    eval_features = eval_features.to_pandas()
                
                if isinstance(eval_labels, pl.Series):
                    eval_labels = eval_labels.to_list()
                
                # Also convert categorical columns in evaluation set
                if cat_columns:
                    for col in cat_columns:
                        if col in eval_features.columns:
                            eval_features[col] = eval_features[col].astype(str)
                
                # For non-ranking models, we ignore the request_ids
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
        try:
            # Check if model attributes exist and are iterable
            if hasattr(self.model, 'feature_names_') and hasattr(self.model, 'feature_importances_'):
                # Handle case where one of them might be a scalar (0-d array)
                feature_names = self.model.feature_names_
                feature_importances = self.model.feature_importances_
                
                # Convert to lists if they're numpy arrays
                import numpy as np
                if isinstance(feature_names, np.ndarray) and feature_names.ndim == 0:
                    self.logger.warning("Feature names is a 0-d array, converting to list")
                    feature_names = [str(feature_names.item())]
                
                if isinstance(feature_importances, np.ndarray) and feature_importances.ndim == 0:
                    self.logger.warning("Feature importances is a 0-d array, converting to list")
                    feature_importances = [feature_importances.item()]
                
                # Handle NaN and None values
                if isinstance(feature_importances, (list, np.ndarray)):
                    # Replace NaN or None with 0.0
                    clean_importances = []
                    for imp in feature_importances:
                        if imp is None or (isinstance(imp, float) and np.isnan(imp)):
                            clean_importances.append(0.0)
                        else:
                            clean_importances.append(imp)
                    feature_importances = clean_importances
                    
                # Create dictionary with feature importances
                result = dict(zip(feature_names, feature_importances))
                
                # Verify no None values remain in result
                for key in list(result.keys()):
                    if result[key] is None or (isinstance(result[key], float) and np.isnan(result[key])):
                        self.logger.warning(f"Replacing None/NaN importance for feature '{key}' with 0.0")
                        result[key] = 0.0
                
                return result
            else:
                self.logger.warning("Model doesn't have feature_names_ or feature_importances_ attributes")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            # Return empty dictionary as fallback
            return {}
    
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
    
    def train(self, train_features, train_labels, eval_set=None, cat_columns=None, train_request_ids=None):
        """Train LightGBM model
        Args:
            train_features (pd.DataFrame or pl.DataFrame): Training features.
            train_labels (pd.Series or pl.Series): Training labels.
            eval_set (tuple): Tuple of (eval_features, eval_labels, eval_request_ids).
            cat_columns (list): Categorical column names.
            train_request_ids (pd.Series or pl.Series): Request IDs for training data.
        """
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
            # Handle both formats: (features, labels) or (features, labels, request_ids)
            if len(eval_set) >= 2:
                eval_features, eval_labels = eval_set[0], eval_set[1]
                # We can ignore eval_request_ids for LightGBM standard models
                
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
        try:
            # Check if model attributes exist and are iterable
            if hasattr(self.model, 'feature_name') and hasattr(self.model, 'feature_importance'):
                # Get feature names and importances
                feature_names = self.model.feature_name()
                feature_importances = self.model.feature_importance()
                
                # Convert to lists if they're numpy arrays
                import numpy as np
                
                # Handle edge cases where they might be scalars (0-d arrays)
                if isinstance(feature_names, np.ndarray) and feature_names.ndim == 0:
                    self.logger.warning("Feature names is a 0-d array, converting to list")
                    feature_names = [str(feature_names.item())]
                
                if isinstance(feature_importances, np.ndarray) and feature_importances.ndim == 0:
                    self.logger.warning("Feature importances is a 0-d array, converting to list")
                    feature_importances = [feature_importances.item()]
                
                # Handle empty arrays
                if isinstance(feature_names, (np.ndarray, list)) and len(feature_names) == 0:
                    self.logger.warning("Empty feature names list")
                    return {}
                
                if isinstance(feature_importances, (np.ndarray, list)) and len(feature_importances) == 0:
                    self.logger.warning("Empty feature importances list")
                    return {}
                
                # Handle NaN and None values
                if isinstance(feature_importances, (list, np.ndarray)):
                    # Replace NaN or None with 0.0
                    clean_importances = []
                    for imp in feature_importances:
                        if imp is None or (isinstance(imp, float) and np.isnan(imp)):
                            clean_importances.append(0.0)
                        else:
                            clean_importances.append(imp)
                    feature_importances = clean_importances
                
                # Create dictionary with feature importances
                result = dict(zip(feature_names, feature_importances))
                
                # Verify no None values remain in result
                for key in list(result.keys()):
                    if result[key] is None or (isinstance(result[key], float) and np.isnan(result[key])):
                        self.logger.warning(f"Replacing None/NaN importance for feature '{key}' with 0.0")
                        result[key] = 0.0
                
                return result
            else:
                self.logger.warning("Model doesn't have feature_name or feature_importance methods")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            # Return empty dictionary as fallback
            return {}
    
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


class CatBoostRankerModel(Model):
    """CatBoost model implementation for ranking tasks"""
    
    def __init__(self, **params):
        super().__init__('catboost_ranker', params)
        from catboost import CatBoostRanker
        
        # Set default loss function if not provided
        if 'loss_function' not in params:
            params['loss_function'] = 'YetiRank'
        
        # Remove unsupported parameters for CatBoostRanker
        if 'scale_pos_weight' in params:
            self.logger.warning("scale_pos_weight is not supported by CatBoostRanker, removing")
            params.pop('scale_pos_weight')
            
        self.params = params
        self.model = CatBoostRanker(**self.params)
        
        # Initialize empty feature importances that will be filled during training
        self._feature_importances = {}
    
    def train(self, train_features, train_labels, eval_set=None, cat_columns=None, train_request_ids=None):
        """Train CatBoost ranker model.
        Args:
            train_features (pd.DataFrame or pl.DataFrame): Training features.
            train_labels (pd.Series or pl.Series): Training labels.
            eval_set (tuple): Tuple of (eval_features, eval_labels, eval_request_ids).
            cat_columns (list): Categorical column names.
            train_request_ids (pd.Series or pl.Series): Request IDs for training data.
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
            "Training CatBoost ranker model with columns: "
            f"{train_features.columns.tolist()} "
            f"(cat_columns: {cat_columns})"
        )
        
        # Extract group ids for ranking (needed for ranking models)
        group_ids = None
        request_id_col = None
        
        # First try to use the passed train_request_ids
        if train_request_ids is not None:
            self.logger.info("Using provided train_request_ids for ranking model")
            if isinstance(train_request_ids, pl.Series):
                train_request_ids = train_request_ids.to_pandas()
            
            # Add the request_ids as a column to sort by
            request_id_col = 'request_id'
            train_features[request_id_col] = train_request_ids
            
            # Convert to string to avoid numeric issues with large IDs
            train_features[request_id_col] = train_features[request_id_col].astype(str)
            self.logger.info(f"Added request IDs as a column and converted to strings")
        # Fallback to looking for request_id in features
        elif 'request_id' in train_features.columns:
            self.logger.info("Using request_id column from features for grouping in ranking model")
            request_id_col = 'request_id'
            # Convert to string to avoid numeric issues with large IDs
            train_features[request_id_col] = train_features[request_id_col].astype(str)
        elif 'group_id' in train_features.columns:
            self.logger.info("Using group_id column from features for grouping in ranking model")
            request_id_col = 'group_id'
            # Convert to string to avoid numeric issues with large IDs
            train_features[request_id_col] = train_features[request_id_col].astype(str)
        else:
            self.logger.warning("No grouping information found for ranking model. Using default grouping.")
        
        # Sort by group id if available to ensure CatBoost's requirement that queryIds should be grouped
        if request_id_col is not None:
            self.logger.info(f"Sorting data by {request_id_col} for grouped ranking")
            train_features = train_features.sort_values(by=request_id_col)
            
            # Reorder train_labels to match the sorted features
            if isinstance(train_labels, pl.Series):
                train_labels = pd.Series(train_labels.to_list(), index=train_features.index).loc[train_features.index].reset_index(drop=True)
            else:
                train_labels = pd.Series(train_labels, index=train_features.index).loc[train_features.index].reset_index(drop=True)
            
            # Extract the sorted group_ids
            group_ids = train_features[request_id_col].values
            
            # Remove the request_id column to avoid using it as a feature
            train_features = train_features.drop(columns=[request_id_col])
        
        
        # Create pool
        train_pool = Pool(
            train_features, train_labels,
            cat_features=cat_columns,
            group_id=group_ids
        )
        
        # Prepare eval set if provided
        eval_pool = None
        if eval_set is not None:
            # Handle both formats: (features, labels) or (features, labels, request_ids)
            if len(eval_set) >= 2:
                eval_features, eval_labels = eval_set[0], eval_set[1]
                eval_request_ids = eval_set[2] if len(eval_set) > 2 else None
                
                if isinstance(eval_features, pl.DataFrame):
                    eval_features = eval_features.to_pandas()
                
                if isinstance(eval_labels, pl.Series):
                    eval_labels = eval_labels.to_list()
                
                # Also convert categorical columns in evaluation set
                if cat_columns:
                    for col in cat_columns:
                        if col in eval_features.columns:
                            eval_features[col] = eval_features[col].astype(str)
                
                # Extract group ids for evaluation set
                eval_group_ids = None
                eval_request_id_col = None
                
                # First try to use the passed eval_request_ids
                if eval_request_ids is not None:
                    self.logger.info("Using provided eval_request_ids for ranking model evaluation")
                    if isinstance(eval_request_ids, pl.Series):
                        eval_request_ids = eval_request_ids.to_pandas()
                    
                    # Add the request_ids as a column to sort by
                    eval_request_id_col = 'request_id'
                    eval_features[eval_request_id_col] = eval_request_ids
                    
                    # Convert to string to avoid numeric issues with large IDs
                    eval_features[eval_request_id_col] = eval_features[eval_request_id_col].astype(str)
                    self.logger.info(f"Added eval request IDs as a column and converted to strings")
                # Fallback to looking for request_id in features
                elif 'request_id' in eval_features.columns:
                    self.logger.info("Using request_id column from eval features for grouping")
                    eval_request_id_col = 'request_id'
                    # Convert to string to avoid numeric issues
                    eval_features[eval_request_id_col] = eval_features[eval_request_id_col].astype(str)
                elif 'group_id' in eval_features.columns:
                    self.logger.info("Using group_id column from eval features for grouping")
                    eval_request_id_col = 'group_id'
                    # Convert to string to avoid numeric issues
                    eval_features[eval_request_id_col] = eval_features[eval_request_id_col].astype(str)
                
                # Sort by group id if available to ensure CatBoost's requirement that queryIds should be grouped
                if eval_request_id_col is not None:
                    self.logger.info(f"Sorting eval data by {eval_request_id_col} for grouped ranking")
                    eval_features = eval_features.sort_values(by=eval_request_id_col)
                    
                    # Reorder eval_labels to match the sorted features
                    if isinstance(eval_labels, list):
                        eval_labels = pd.Series(eval_labels, index=eval_features.index).loc[eval_features.index].reset_index(drop=True).tolist()
                    else:
                        eval_labels = pd.Series(eval_labels, index=eval_features.index).loc[eval_features.index].reset_index(drop=True)
                    
                    # Extract the sorted group_ids
                    eval_group_ids = eval_features[eval_request_id_col].values
                    
                    # Remove the request_id column to avoid using it as a feature
                    eval_features = eval_features.drop(columns=[eval_request_id_col])
                
                eval_pool = Pool(
                    eval_features, 
                    eval_labels, 
                    cat_features=cat_columns,
                    group_id=eval_group_ids
                )
        
        # Create a new instance with the same params for training
        from catboost import CatBoostRanker
        self.model = CatBoostRanker(**self.params)
        
        self.logger.info("Training CatBoost ranker model")
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
        
        # Calculate and save feature importance during training, since we have the train_pool available
        try:
            self.logger.info("Calculating feature importance during training...")
            # Save feature importances for later retrieval
            self._feature_importances = {}
            
            # Try PredictionValuesChange first (best for rankings)
            try:
                importances = self.model.get_feature_importance(train_pool, type='PredictionValuesChange')
                self.logger.info("Successfully calculated PredictionValuesChange importance")
                # Create dictionary mapping feature names to importance values
                feature_names = self.model.feature_names_
                self._feature_importances = dict(zip(feature_names, importances))
            except Exception as e:
                self.logger.warning(f"PredictionValuesChange importance failed: {str(e)}")
                
                # Try LossFunctionChange next
                try:
                    importances = self.model.get_feature_importance(train_pool, type='LossFunctionChange')
                    self.logger.info("Successfully calculated LossFunctionChange importance")
                    # Create dictionary mapping feature names to importance values
                    feature_names = self.model.feature_names_
                    self._feature_importances = dict(zip(feature_names, importances))
                except Exception as e2:
                    self.logger.warning(f"LossFunctionChange importance failed: {str(e2)}")
                    
                    # Try default type as last resort
                    try:
                        importances = self.model.get_feature_importance(train_pool)
                        self.logger.info("Successfully calculated default importance")
                        # Create dictionary mapping feature names to importance values
                        feature_names = self.model.feature_names_
                        self._feature_importances = dict(zip(feature_names, importances))
                    except Exception as e3:
                        self.logger.warning(f"Default importance failed: {str(e3)}")
            
            # Log if we have feature importance
            if self._feature_importances:
                top_features = sorted(
                    self._feature_importances.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                self.logger.info(f"Top 5 feature importances: {top_features}")
            else:
                self.logger.warning("Could not calculate any feature importance during training")
                
        except Exception as e:
            self.logger.warning(f"Error calculating feature importance during training: {str(e)}")
        
        return self
    
    def predict(self, features, request_ids=None, **kwargs):
        """Make relevance predictions with CatBoost ranker
        
        Args:
            features (pd.DataFrame or pl.DataFrame): Features for prediction
            request_ids (pd.Series or pl.Series): Optional request IDs for ranking
            **kwargs: Additional arguments
            
        Returns:
            Prediction scores
        """
        # Convert to pandas for CatBoost
        if isinstance(features, pl.DataFrame):
            features = features.to_pandas()
            
        # Save a copy of the column names before any transformations
        original_columns = list(features.columns)
        
        # Make a copy of the dataframe to avoid modifying the original
        features_for_pred = features.copy()
        
        # Extract request_ids from features if not provided externally
        group_ids = None
        request_id_col = None
        original_index = features_for_pred.index.copy()  # Keep original index for returning predictions in the same order
        
        if request_ids is not None:
            # Convert to pandas if needed
            if isinstance(request_ids, pl.Series):
                request_ids = request_ids.to_pandas()
                
            # Add request_ids as a column for sorting
            request_id_col = 'request_id'
            features_for_pred[request_id_col] = request_ids.astype(str)
            self.logger.info(f"Added request_ids as a column for prediction and converted to strings")
        
        # Check for request_id/group_id columns in features
        elif 'request_id' in features_for_pred.columns:
            request_id_col = 'request_id'
            features_for_pred[request_id_col] = features_for_pred[request_id_col].astype(str)
            self.logger.info(f"Using request_id column from features for prediction")
        elif 'group_id' in features_for_pred.columns:
            request_id_col = 'group_id'
            features_for_pred[request_id_col] = features_for_pred[request_id_col].astype(str)
            self.logger.info(f"Using group_id column from features for prediction")
            
        # Sort by group id if available to ensure CatBoost's requirement that queryIds should be grouped
        if request_id_col is not None:
            self.logger.info(f"Sorting prediction data by {request_id_col} for grouped ranking")
            features_for_pred = features_for_pred.sort_values(by=request_id_col)
            
            # Extract the sorted group_ids
            group_ids = features_for_pred[request_id_col].values
            
            # Remove the request_id column to avoid using it as a feature
            features_for_pred = features_for_pred.drop(columns=[request_id_col])
        
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
        
        # Create a Pool if we have group_ids
        if group_ids is not None:
            from catboost import Pool
            try:
                # Get categorical features
                cat_features = []
                if hasattr(self.model, 'get_cat_feature_indices'):
                    cat_indices = self.model.get_cat_feature_indices()
                    if hasattr(self.model, 'feature_names_'):
                        feature_names = self.model.feature_names_
                        cat_features = [feature_names[i] for i in cat_indices if i < len(feature_names)]
                
                # Create prediction pool with group_ids
                pool = Pool(
                    data=features_for_pred,
                    cat_features=cat_features,
                    group_id=group_ids
                )
                self.logger.info("Using Pool with group_ids for prediction")
                predictions = self.model.predict(pool)
                
                # If we rearranged the data, we need to reorder the predictions to match the original order
                if group_ids is not None and hasattr(features_for_pred, 'index') and hasattr(original_index, 'equals') and not original_index.equals(features_for_pred.index):
                    self.logger.info("Reordering predictions to match original data order")
                    # Create a series with the current index
                    pred_series = pd.Series(predictions, index=features_for_pred.index)
                    # Reindex to get original order
                    predictions = pred_series.reindex(original_index).values
                
                return predictions
            except Exception as e:
                self.logger.warning(f"Failed to use Pool with group_ids: {str(e)}")
                
        # Fallback to standard prediction without grouping
        predictions = self.model.predict(features_for_pred)
        
        # If we rearranged the data, we need to reorder the predictions to match the original order
        if hasattr(features_for_pred, 'index') and hasattr(original_index, 'equals') and not original_index.equals(features_for_pred.index):
            self.logger.info("Reordering predictions to match original data order")
            try:
                # Create a series with the current index
                pred_series = pd.Series(predictions, index=features_for_pred.index)
                # Reindex to get original order
                predictions = pred_series.reindex(original_index).values
            except Exception as e:
                self.logger.warning(f"Failed to reorder predictions: {str(e)}")
        
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance from CatBoost ranker"""
        try:
            import numpy as np
            
            # First, check if we have pre-calculated importances from training
            if hasattr(self, '_feature_importances') and self._feature_importances:
                self.logger.info("Using pre-calculated feature importance from training")
                return self._feature_importances
            
            # Otherwise, try to calculate it now (though this will likely fail without train_pool)
            if hasattr(self.model, 'get_feature_importance') and hasattr(self.model, 'feature_names_'):
                # Get feature names
                feature_names = self.model.feature_names_
                
                # Try to use the method to get feature importance
                try:
                    # For CatBoostRanker, specify the type explicitly
                    feature_importances = self.model.get_feature_importance(type='FeatureImportance')
                    self.logger.info(f"Successfully calculated feature importance with type='FeatureImportance'")
                except Exception as e1:
                    self.logger.warning(f"Error getting feature importance with FeatureImportance: {str(e1)}")
                    try:
                        # Try the default type
                        feature_importances = self.model.get_feature_importance()
                        self.logger.info(f"Successfully calculated feature importance with default type")
                    except Exception as e2:
                        self.logger.warning(f"Error getting feature importance with default type: {str(e2)}")
                        # As a fallback, try to get it from the attribute directly
                        if hasattr(self.model, 'feature_importances_'):
                            feature_importances = self.model.feature_importances_
                            self.logger.info(f"Using feature_importances_ attribute directly")
                        else:
                            self.logger.warning("Could not calculate feature importance for CatBoostRanker")
                            return {}
                
                # Check if we have valid feature names and importances
                if len(feature_names) == 0:
                    self.logger.warning("Empty feature names list")
                    return {}
                
                # Handle case where one of them might be a scalar (0-d array)
                if isinstance(feature_names, np.ndarray) and feature_names.ndim == 0:
                    self.logger.warning("Feature names is a 0-d array, converting to list")
                    feature_names = [str(feature_names.item())]
                
                if isinstance(feature_importances, np.ndarray) and feature_importances.ndim == 0:
                    self.logger.warning("Feature importances is a 0-d array, converting to list")
                    feature_importances = [feature_importances.item()]
                
                # Handle NaN and None values
                if isinstance(feature_importances, (list, np.ndarray)):
                    # Replace NaN or None with 0.0
                    clean_importances = []
                    for imp in feature_importances:
                        if imp is None or (isinstance(imp, float) and np.isnan(imp)):
                            clean_importances.append(0.0)
                        else:
                            clean_importances.append(imp)
                    feature_importances = clean_importances
                
                # Make sure lengths match
                if len(feature_names) != len(feature_importances):
                    self.logger.warning(f"Feature names length ({len(feature_names)}) doesn't match importances length ({len(feature_importances)})")
                    # Truncate to the shorter length
                    min_len = min(len(feature_names), len(feature_importances))
                    feature_names = feature_names[:min_len]
                    feature_importances = feature_importances[:min_len]
                    
                # Create dictionary with feature importances
                result = dict(zip(feature_names, feature_importances))
                
                # Verify no None values remain in result
                for key in list(result.keys()):
                    if result[key] is None or (isinstance(result[key], float) and np.isnan(result[key])):
                        self.logger.warning(f"Replacing None/NaN importance for feature '{key}' with 0.0")
                        result[key] = 0.0
                
                return result
            else:
                self.logger.warning("Model doesn't have required methods/attributes for feature importance")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            # Return empty dictionary as fallback
            return {}
    
    def save(self, filename):
        """Save model to file"""
        # For saving via pickle (used by model cache), the _feature_importances attribute 
        # will be automatically included in the pickled data
        
        # For direct save to file, save both model and metadata
        try:
            # First save the model using CatBoost's save_model
            self.model.save_model(filename)
            self.logger.info(f"Saved CatBoost model to {filename}")
            
            # If we have feature importances, also save them to a metadata file
            if hasattr(self, '_feature_importances') and self._feature_importances:
                import pickle
                meta_filename = f"{filename}.meta"
                
                # Save the feature importances separately
                with open(meta_filename, 'wb') as f:
                    pickle.dump({'feature_importances': self._feature_importances}, f)
                
                self.logger.info(f"Saved model metadata to {meta_filename}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
    
    @classmethod
    def load(cls, filename):
        """Load model from file"""
        from catboost import CatBoostRanker
        import os
        
        # Create a new instance
        model = cls()
        
        # Load the CatBoost model
        model.model = CatBoostRanker()
        model.model.load_model(filename)
        
        # Try to load feature importances from metadata file
        try:
            import pickle
            meta_filename = f"{filename}.meta"
            
            if os.path.exists(meta_filename):
                with open(meta_filename, 'rb') as f:
                    metadata = pickle.load(f)
                    
                if 'feature_importances' in metadata:
                    model._feature_importances = metadata['feature_importances']
                    model.logger.info(f"Loaded feature importances with {len(model._feature_importances)} features")
            else:
                model.logger.info(f"No metadata file found at {meta_filename}")
        except Exception as e:
            model.logger.warning(f"Could not load metadata: {str(e)}")
                
        return model
        
    def __getstate__(self):
        """
        Special method for pickle protocol.
        Return the state of the object to be pickled.
        """
        # Get all attributes
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        """
        Special method for pickle protocol.
        Restore the state of the object from pickled state.
        """
        # Restore all attributes
        self.__dict__.update(state)


class ModelFactory:
    """Factory for creating and managing models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {
            'catboost': CatBoostModel,
            'catboost_ranker': CatBoostRankerModel,
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
        model_params = self.config.get(('model', 'config', model_type), {}).copy()
        
        # Apply GPU configuration for CatBoost models
        if model_type in ['catboost', 'catboost_ranker']:
            use_gpu = self.config.get(('model', 'use_gpu'), False)
            gpu_devices = self.config.get(('model', 'gpu_devices'), '0')
            thread_count = self.config.get(('model', 'thread_count'), -1)
            
            if use_gpu:
                self.logger.info(f"Enabling GPU training with devices: {gpu_devices}")
                model_params['task_type'] = 'GPU'
                model_params['devices'] = gpu_devices
            else:
                self.logger.info("Using CPU for training")
                model_params['task_type'] = 'CPU'
                
            # Set thread count for CPU processing
            model_params['thread_count'] = thread_count
            self.logger.info(f"Setting thread count to {thread_count}")
            
        # Apply GPU configuration for LightGBM models
        elif model_type == 'lightgbm':
            use_gpu = self.config.get(('model', 'use_gpu'), False)
            gpu_devices = self.config.get(('model', 'gpu_devices'), '0')
            thread_count = self.config.get(('model', 'thread_count'), -1)
            
            if use_gpu:
                self.logger.info(f"Enabling GPU training for LightGBM with device: {gpu_devices}")
                model_params['device'] = 'gpu'
                model_params['gpu_device_id'] = int(gpu_devices.split(',')[0])  # Use first device for LightGBM
            else:
                model_params['device'] = 'cpu'
                
            # Set number of threads
            if thread_count > 0:
                model_params['num_threads'] = thread_count
        
        # Apply overrides from parameters
        if override_params:
            model_params.update(override_params)
    
        # Create and return model instance
        self.logger.info(f"Creating {model_type} model with params: {model_params}")
        return self.models[model_type](**model_params)