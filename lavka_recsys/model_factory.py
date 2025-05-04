from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import pickle
import polars as pl

from .custom_logging import get_logger
from .config import Config


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
    """General CatBoost model implementation"""

    def _prepare_pool(self,
        features: pd.DataFrame | pl.DataFrame,
        labels: iter[str] = None,
        *,
        group_ids = None,
        cat_columns = None
        ):
        """Train CatBoost model.
        Args:
            features (pd.DataFrame or pl.DataFrame): Features
            labels (pd.Series or pl.Series): Labels (optional).
            group_ids (pd.Series or pl.Series): Group IDs (optional).
            cat_columns (list): Categorical column names.
        """
        from catboost import Pool
    
        if isinstance(features, pl.DataFrame):
            features = features.to_pandas()
        if isinstance(labels, pl.Series):
            labels = labels.to_list()
            
        # Convert categorical columns to string type
        cat_columns = cat_columns or []
        for col in cat_columns:
            self.logger.info(f"Converting categorical column '{col}' to string type")
            features[col] = features[col].astype(str)
        
        pool = Pool(
            features,
            label=labels,
            cat_features=cat_columns,
            group_id=group_ids
        )
        return pool


class CatBoostClassifierModel(CatBoostModel):
    """CatBoostClassifier"""
    
    def __init__(self, **params):
        super().__init__('catboost_classifier', params)
        from catboost import CatBoostClassifier
        
        self.params = params
        self.model = CatBoostClassifier(**self.params)
    
    def train(self,
        train_features: pd.DataFrame | pl.DataFrame,
        train_labels: iter[str],
        *,
        eval_features=None,
        eval_labels=None,
        cat_columns=None,
        **kwargs
    ):
        """Train CatBoost model.
        Args:
            train_features (pd.DataFrame or pl.DataFrame): Training features.
            train_labels (pd.Series or pl.Series): Training labels.
            train_group_ids (pd.Series or pl.Series): Request IDs for training data.
            cat_columns (list): Categorical column names.
        """
        train_pool = self._prepare_pool(
            train_features,
            labels=train_labels,
            cat_columns=cat_columns,
        )
        eval_pool = None if eval_features is None else self._prepare_pool(
            eval_features,
            labels=eval_labels,
            cat_columns=cat_columns,
        )
        
        self.logger.info(
            "Training CatBoost model with columns: "
            f"{train_features.columns.tolist()} "
            f"(cat_columns: {cat_columns})"
        )
        self.model.fit(train_pool, eval_set=eval_pool, verbose=False, plot=True)
        return self
    
    def predict(self, features: pd.DataFrame | pl.DataFrame):
        """Make probability predictions with CatBoost"""
        # Make a copy of the dataframe to avoid modifying the original
        feature_names = self.model.feature_names_

        # Ensure we have the exact same columns the model was trained on
        missing_cols = set(self.model.feature_names_) - set(features.columns)
        if missing_cols:
            self.logger.error(f"Missing columns in prediction data: {missing_cols}")
            raise ValueError(f"Missing columns in prediction data: {missing_cols}")
        
        cat_feature_indices = self.model.get_cat_feature_indices()            
        if cat_feature_indices:
            cat_columns = [feature_names[i] for i in cat_feature_indices]
        else:
            cat_columns = None

        pool = self._prepare_pool(features, cat_columns=cat_columns)
        return self.model.predict_proba(pool)[:, 1]
    
    def get_feature_importance(self):
        """Get feature importance from CatBoost"""
        feature_names = self.model.feature_names_
        feature_importances = self.model.feature_importances_
        
        # Convert to lists if they're numpy arrays
        if isinstance(feature_names, np.ndarray) and feature_names.ndim == 0:
            self.logger.warning("Feature names is a 0-d array, converting to list")
            feature_names = [str(feature_names.item())]
        if isinstance(feature_importances, np.ndarray) and feature_importances.ndim == 0:
            self.logger.warning("Feature importances is a 0-d array, converting to list")
            feature_importances = [feature_importances.item()]
        
        # Replace NaN or None with 0.0
        if isinstance(feature_importances, (list, np.ndarray)):
            clean_importances = []
            for imp in feature_importances:
                if imp is None or (isinstance(imp, float) and np.isnan(imp)):
                    clean_importances.append(0.0)
                else:
                    clean_importances.append(imp)
            feature_importances = clean_importances
            
        # Create dictionary with feature importances
        result = dict(zip(feature_names, feature_importances))
        return result
    
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


class CatBoostRankerModel(CatBoostModel):
    """CatBoost model implementation for ranking tasks"""
    
    def __init__(self, **params):
        super().__init__('catboost_ranker', params)
        from catboost import CatBoostRanker
            
        self.params = params
        self.model = CatBoostRanker(**self.params)
        
        # Initialize empty feature importances that will be filled during training
        self._feature_importances = {}
    
    def train(self,
        train_features: pd.DataFrame | pl.DataFrame,
        train_labels: iter[str],
        *,
        train_group_ids = None,
        eval_features = None,
        eval_labels = None,
        eval_group_ids = None,
        cat_columns = None,
        **kwargs
    ):
        """Train CatBoost ranker model.
        Args:
            train_features (pd.DataFrame or pl.DataFrame): Training features.
            train_labels (pd.Series or pl.Series): Training labels.
            train_group_ids (pd.Series or pl.Series): Request IDs for training data.
            eval_features (pd.DataFrame or pl.DataFrame): Evaluation features.
            eval_labels (pd.Series or pl.Series): Evaluation labels.
            eval_group_ids (pd.Series or pl.Series): Request IDs for evaluation data.
            cat_columns (list): Categorical column names.
        """
        if train_group_ids is None:
            train_features, train_labels, train_group_ids, _ = self._extract_group_ids(
                train_features, train_labels
            )
        if eval_group_ids is None:
            eval_features, eval_labels, eval_group_ids, _ = self._extract_group_ids(
                eval_features, eval_labels
            )
        
        train_pool = self._prepare_pool(
            train_features,
            labels=train_labels,
            group_ids=train_group_ids,
            cat_columns=cat_columns,
        )
        eval_pool = None if eval_features is None else self._prepare_pool(
            eval_features,
            labels=eval_labels,
            group_ids=eval_group_ids,
            cat_columns=cat_columns,
        )
        self.model.fit(train_pool, eval_set=eval_pool, plot=True, verbose=False)

        # Calculate and save feature importance during training, since we have the train_pool available
        self._compute_feature_importances(train_pool)
        return self
    
    def predict(self, features, group_ids=None, **kwargs):
        """Make relevance predictions with CatBoost ranker
        
        Args:
            features (pd.DataFrame or pl.DataFrame): Features for prediction
            group_ids (pd.Series or pl.Series): Group IDs for ranking
            **kwargs: Additional arguments
            
        Returns:
            Prediction scores
        """   
        if isinstance(features, pl.DataFrame):
           features = features.to_pandas()

        # Save the "true" order of the rows
        original_index = features.index.copy()
    
        if group_ids is None:
           features, _, group_ids, _ = self._extract_group_ids(features)
 
        # Ensure we have the exact same columns the model was trained on
        missing_cols = set(self.model.feature_names_) - set(features.columns)
        if missing_cols:
            self.logger.error(f"Missing columns in prediction data: {missing_cols}")
            raise ValueError(f"Missing columns in prediction data: {missing_cols}")
        
        cat_feature_indices = self.model.get_cat_feature_indices()
        feature_names = self.model.feature_names_          
        if cat_feature_indices:
            cat_columns = [feature_names[i] for i in cat_feature_indices]
        else:
            cat_columns = None
        
        pool = self._prepare_pool(
            features,
            group_ids=group_ids,
            cat_columns=cat_columns
        )
        predictions = self.model.predict(pool)
        
        # Reindex to get original order
        self.logger.info("Reordering predictions to match original data order")
        predictions = pd.Series(predictions, index=features.index)
        predictions = predictions.reindex(original_index)

        return predictions
    
    def get_feature_importance(self):
        """Get feature importance from CatBoost ranker"""
        self.logger.info("Using pre-calculated feature importance from training")
        return self._feature_importances
    
    def _compute_feature_importances(self, train_pool):
        self._feature_importances = {}
        importance_types = ['PredictionValuesChange', 'LossFunctionChange']

        for importance_type in importance_types:
            try:
                importances = self.model.get_feature_importance(train_pool, type=importance_type)
                self.logger.info("Successfully calculated PredictionValuesChange importance")
                # Create dictionary mapping feature names to importance values
                feature_names = self.model.feature_names_
                self._feature_importances = dict(zip(feature_names, importances))
            except Exception as e:
                self.logger.warning(f'Failed to calculate feature importances using {importance_type}: {str(e)}')
        
        if self._feature_importances:
            top_features = sorted(
                self._feature_importances.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            self.logger.info(f"Top 5 feature importances: {top_features}")
        else:
            self.logger.error('Failed to calculate any feature importances')
    
    def _extract_group_ids(self, features, labels=None):
        if isinstance(features, pl.DataFrame):
            features = features.to_pandas()
        if isinstance(labels, pl.Series):
            labels = labels.to_list()
        
        original_index = features.index.copy()

        if 'group_id' in features.columns:
            group_id_col = 'group_id'
        elif 'request_id' in features.columns:
            group_id_col = 'request_id'
        else:
            self.logger.error('Missing either `group_id` or `request_id` in features dataframe')
            raise ValueError('Missing either `group_id` or `request_id` in features dataframe')
        
        # Sort by group id if available to ensure CatBoost's requirement that queryIds should be grouped
        self.logger.info(f"Sorting data by {group_id_col} for grouped ranking")
        features = features.sort_values(by=group_id_col)
        group_ids = features[group_id_col].astype(str).values    
        features = features.drop(columns=[group_id_col])

        if labels is not None:
            labels = pd.Series(labels, index=features.index).loc[features.index].reset_index(drop=True)
        
        return features, labels, group_ids, original_index
    
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
                meta_filename = f"{filename}.meta"
                
                # Save the feature importances separately
                with open(meta_filename, 'wb') as f:
                    pickle.dump({'feature_importances': self._feature_importances}, f)
                
                self.logger.info(f"Saved model metadata to {meta_filename}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise ValueError(f"Error saving model: {str(e)}")
    
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
            'catboost_classifier': CatBoostClassifierModel,
            'catboost_ranker': CatBoostRankerModel,
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
        
        # Apply GPU configuration
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
        
        # Apply overrides from parameters
        if override_params:
            model_params.update(override_params)
    
        # Create and return model instance
        self.logger.info(f"Creating {model_type} model with params: {model_params}")
        return self.models[model_type](**model_params)
