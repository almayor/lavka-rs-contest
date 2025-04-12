```python
# ======================================================
# Recommender System Experimental Framework
# ======================================================

import json
import logging
import os
import time
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

# Uncomment as needed
# from catboost import CatBoostClassifier, Pool
# import lightgbm as lgb
# import xgboost as xgb
# from sklearn.ensemble import RandomForestClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('recommender-experiments')

# ======================================================
# Configuration Management
# ======================================================

class Config:
    """Configuration management for experiments"""
    
    def __init__(self, config_dict=None):
        """Initialize with default or provided configuration"""
        self.config = config_dict or {}
        
        # Set defaults if not provided
        self._set_defaults()
        
    def _set_defaults(self):
        """Set default configuration values"""
        defaults = {
            'data': {
                'train_path': 'train.parquet',
                'test_path': 'test.parquet',
                'sample_size': None,  # None = use all data
                'random_seed': 42
            },
            'preprocessing': {
                'remove_duplicates': True,
                'fill_nulls': True,
                'normalize_timestamps': True,
                'clean_text': False
            },
            'features': {
                'basic': ['count_purchase', 'ctr'],
                'temporal': ['recency', 'frequency', 'time_window'],
                'user': ['user_stats', 'user_preferences'],
                'product': ['product_stats', 'category_stats'],
                'advanced': ['novelty', 'serendipity']
            },
            'models': {
                'catboost': {
                    'iterations': 500,
                    'learning_rate': 0.05,
                    'depth': 6,
                    'loss_function': 'Logloss',
                    'eval_metric': 'AUC',
                    'early_stopping_rounds': 50,
                    'verbose': 100
                },
                'lightgbm': {
                    'num_iterations': 500,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'objective': 'binary',
                    'metric': 'auc',
                    'early_stopping_rounds': 50,
                    'verbose': 100
                },
                'xgboost': {
                    'n_estimators': 500,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'early_stopping_rounds': 50,
                    'verbose': 100
                },
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'random_state': 42
                }
            },
            'validation': {
                'method': 'temporal',  # 'temporal', 'kaggle', 'random'
                'n_folds': 3,
                'gap_days': 0,
                'test_size': 0.2
            },
            'metrics': ['auc', 'ndcg@10', 'map@10', 'novelty@10', 'serendipity@10'],
            'output': {
                'results_dir': 'results',
                'save_models': True,
                'save_features': False,
                'save_predictions': True
            }
        }
        
        # Update config with defaults for missing values
        for section, values in defaults.items():
            if section not in self.config:
                self.config[section] = values
            else:
                for key, value in values.items():
                    if key not in self.config[section]:
                        self.config[section][key] = value
    
    def get(self, section, key=None):
        """Get a configuration value"""
        if key is None:
            return self.config.get(section, {})
        return self.config.get(section, {}).get(key)
    
    def set(self, section, key, value):
        """Set a configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def save(self, filename='experiment_config.json'):
        """Save configuration to file"""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    @classmethod
    def load(cls, filename='experiment_config.json'):
        """Load configuration from file"""
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def __str__(self):
        """String representation of config"""
        return json.dumps(self.config, indent=4)


# ======================================================
# Data Loading and Preprocessing
# ======================================================

class DataLoader:
    """Data loading and preprocessing"""
    
    def __init__(self, config: Config):
        """Initialize with configuration"""
        self.config = config
        self.train_df = None
        self.test_df = None
    
    def load_data(self):
        """Load training and testing data"""
        logger.info("Loading data...")
        
        # Load train data
        train_path = self.config.get('data', 'train_path')
        self.train_df = pl.read_parquet(train_path)
        
        # Load test data
        test_path = self.config.get('data', 'test_path')
        self.test_df = pl.read_parquet(test_path)
        
        # Sample if needed
        sample_size = self.config.get('data', 'sample_size')
        if sample_size is not None:
            self.train_df = self.train_df.sample(
                n=sample_size, 
                seed=self.config.get('data', 'random_seed')
            )
        
        logger.info(f"Loaded train data: {len(self.train_df)} rows")
        logger.info(f"Loaded test data: {len(self.test_df)} rows")
        
        return self.train_df, self.test_df
    
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply preprocessing steps based on configuration"""
        logger.info("Preprocessing data...")
        
        # Make a copy to avoid modifying the original
        processed_df = df.clone()
        
        # Apply preprocessing steps based on config
        if self.config.get('preprocessing', 'remove_duplicates'):
            processed_df = processed_df.unique()
            logger.info("Removed duplicates")
        
        if self.config.get('preprocessing', 'fill_nulls'):
            # Fill numerical nulls with 0
            processed_df = processed_df.fill_null(0)
            logger.info("Filled null values")
        
        if self.config.get('preprocessing', 'normalize_timestamps'):
            # Ensure timestamps are in a consistent format
            if 'timestamp' in processed_df.columns:
                # Convert to datetime if it's not already
                if processed_df['timestamp'].dtype != pl.Datetime:
                    processed_df = processed_df.with_columns(
                        pl.col('timestamp').cast(pl.Datetime)
                    )
                logger.info("Normalized timestamps")
        
        if self.config.get('preprocessing', 'clean_text'):
            # Apply text cleaning to relevant columns
            text_columns = [col for col in processed_df.columns 
                           if any(substr in col for substr in ['name', 'description', 'text'])]
            
            for col in text_columns:
                if col in processed_df.columns:
                    # Simple text cleaning example
                    processed_df = processed_df.with_columns(
                        pl.col(col).str.strip().str.to_lowercase()
                    )
            logger.info("Cleaned text columns")
        
        return processed_df
    
    def create_validation_splits(self):
        """Create training/validation splits based on config"""
        validation_method = self.config.get('validation', 'method')
        
        if validation_method == 'temporal':
            return self._create_temporal_splits()
        elif validation_method == 'kaggle':
            return self._create_kaggle_split()
        elif validation_method == 'random':
            return self._create_random_splits()
        else:
            raise ValueError(f"Unknown validation method: {validation_method}")
    
    def _create_temporal_splits(self):
        """Create time-based validation folds"""
        n_folds = self.config.get('validation', 'n_folds')
        gap_days = self.config.get('validation', 'gap_days')
        
        # Ensure data is sorted by timestamp
        df = self.train_df.sort('timestamp')
        
        # Calculate time range and fold duration
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        time_range = max_time - min_time
        fold_duration = time_range / (n_folds + 1)  # +1 to leave last fold for final validation
        
        folds = []
        for i in range(n_folds):
            # Calculate time boundaries
            train_end_time = min_time + fold_duration * (i + 1)
            
            # Add gap if specified
            if gap_days > 0:
                val_start_time = train_end_time + timedelta(days=gap_days)
            else:
                val_start_time = train_end_time
                
            val_end_time = val_start_time + fold_duration
            
            # Create train and validation sets
            train_df = df.filter(pl.col('timestamp') < train_end_time)
            val_df = df.filter((pl.col('timestamp') >= val_start_time) & 
                              (pl.col('timestamp') < val_end_time))
            
            folds.append((train_df, val_df))
        
        logger.info(f"Created {len(folds)} temporal validation folds")
        return folds
    
    def _create_kaggle_split(self):
        """Use Kaggle's predefined train/validation split"""
        # For Kaggle competitions, you might have a separate validation file
        # Here we'll simulate by taking the most recent data as validation
        
        df = self.train_df.sort('timestamp')
        
        # Use the most recent X% as validation
        test_size = self.config.get('validation', 'test_size')
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df[:split_idx]
        val_df = df[split_idx:]
        
        logger.info(f"Created Kaggle-style split: {len(train_df)} train, {len(val_df)} validation")
        return [(train_df, val_df)]
    
    def _create_random_splits(self):
        """Create random validation splits (not recommended for time series)"""
        n_folds = self.config.get('validation', 'n_folds')
        test_size = self.config.get('validation', 'test_size')
        seed = self.config.get('data', 'random_seed')
        
        df = self.train_df.clone()
        folds = []
        
        for i in range(n_folds):
            # Shuffle and split
            shuffled = df.sample(fraction=1.0, seed=seed + i)
            split_idx = int(len(shuffled) * (1 - test_size))
            
            train_df = shuffled[:split_idx]
            val_df = shuffled[split_idx:]
            
            folds.append((train_df, val_df))
        
        logger.info(f"Created {len(folds)} random validation folds")
        return folds


# ======================================================
# Feature Factory with Decorator Pattern
# ======================================================

class FeatureFactory:
    """Feature generation with selective feature creation"""
    
    # Class-level registry of feature generators
    _feature_registry = {}
    
    @classmethod
    def register(cls, feature_name, depends_on=None, category=None):
        """Decorator to register a method as a feature generator"""
        depends_on = depends_on or []
        
        def decorator(func):
            cls._feature_registry[feature_name] = {
                'func': func,
                'depends_on': depends_on,
                'category': category
            }
            
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                return func(self, *args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def __init__(self):
        """Initialize feature factory"""
        self.features = {}  # Cache for generated features
        self.config = None
    
    def set_config(self, config):
        """Set configuration"""
        self.config = config
    
    def generate_features(self, history_df, target_df, requested_features):
        """Generate only the requested features and their dependencies"""
        logger.info(f"Generating features: {', '.join(requested_features)}")
        
        # Reset cache for new request
        self.features = {}
        
        # Expand feature groups if needed
        expanded_features = self._expand_feature_groups(requested_features)
        
        # Generate each requested feature (and dependencies)
        for feature_name in expanded_features:
            self._generate_feature(feature_name, history_df, target_df)
        
        # Return only the requested features
        return {f: self.features[f] for f in expanded_features if f in self.features}
    
    def _expand_feature_groups(self, requested_features):
        """Expand feature group names into individual features"""
        if not self.config:
            return requested_features
            
        expanded = []
        feature_groups = self.config.get('features')
        
        for feature in requested_features:
            if feature in feature_groups:
                # This is a feature group
                expanded.extend(feature_groups[feature])
            else:
                # This is an individual feature
                expanded.append(feature)
                
        return list(set(expanded))  # Remove duplicates
    
    def _generate_feature(self, feature_name, history_df, target_df):
        """Generate a single feature, handling dependencies"""
        # Return from cache if already generated
        if feature_name in self.features:
            return self.features[feature_name]
        
        # Check if feature exists
        if feature_name not in self.__class__._feature_registry:
            logger.warning(f"Feature '{feature_name}' is not registered")
            return None
        
        # Get feature info
        feature_info = self.__class__._feature_registry[feature_name]
        generator_func = feature_info['func']
        dependencies = feature_info['depends_on']
        
        # Generate dependencies first
        for dep in dependencies:
            self._generate_feature(dep, history_df, target_df)
        
        # Generate this feature
        logger.debug(f"Generating feature: {feature_name}")
        feature_df = generator_func(self, history_df, target_df)
        
        # Cache and return
        self.features[feature_name] = feature_df
        return feature_df
    
    def join_features(self, base_df=None, common_keys=None):
        """Join all generated features into a single dataframe"""
        if not self.features:
            logger.warning("No features to join")
            return None
        
        if common_keys is None:
            common_keys = ['user_id', 'product_id']
            
        # Start with base_df or the first feature
        if base_df is not None:
            result = base_df
        else:
            result = list(self.features.values())[0]
        
        # Join the rest
        for feature_name, feature_df in self.features.items():
            if feature_df is not result:  # Don't join with itself
                try:
                    result = result.join(
                        feature_df, on=common_keys, how='left'
                    )
                except Exception as e:
                    logger.error(f"Error joining feature {feature_name}: {e}")
                    
        # Fill null values
        result = result.fill_null(0)
        return result
    
    # Feature generation methods with decorators
    
    @register('count_purchase', category='basic')
    def generate_count_purchase(self, history_df, target_df):
        """Count purchases by user-product pairs"""
        return history_df.filter(
            pl.col('action_type') == "AT_Purchase"
        ).group_by(
            'user_id', 'product_id'
        ).agg(
            pl.len().alias('count_purchase')
        )
    
    @register('ctr', category='basic')
    def generate_ctr(self, history_df, target_df):
        """Calculate CTR (Click-Through Rate) for products"""
        actions = history_df.group_by(
            'action_type', 'product_id'
        ).agg(
            pl.len()
        )
        
        clicks = actions.filter(pl.col('action_type') == "AT_Click")
        views = actions.filter(pl.col('action_type') == "AT_View")
        
        return clicks.join(
            views, on='product_id'
        ).with_columns(
            ctr=pl.col('len') / pl.col('len_right')
        ).select(
            'product_id', 'ctr'
        )
    
    @register('recency', category='temporal')
    def generate_recency(self, history_df, target_df):
        """Generate recency features"""
        latest_time = history_df['timestamp'].max()
        
        return history_df.group_by(['user_id', 'product_id']).agg(
            pl.max('timestamp').alias('last_interaction')
        ).with_columns(
            days_since_interaction=(latest_time - pl.col('last_interaction')) / (24 * 60 * 60)
        )
    
    @register('frequency', category='temporal', depends_on=['recency', 'count_purchase'])
    def generate_frequency(self, history_df, target_df):
        """Calculate interaction frequency based on recency and count"""
        recency = self.features['recency']
        count_purchase = self.features['count_purchase']
        
        return recency.join(
            count_purchase, on=['user_id', 'product_id'], how='left'
        ).with_columns(
            interaction_frequency=pl.col('count_purchase') / pl.col('days_since_interaction').clip(1, None)
        ).fill_null(0)
    
    @register('time_window', category='temporal')
    def generate_time_window_features(self, history_df, target_df):
        """Generate features from different time windows"""
        latest_time = history_df['timestamp'].max()
        window_sizes = [1, 7, 30]  # days
        
        dfs = []
        for window in window_sizes:
            cutoff_time = latest_time - timedelta(days=window)
            window_data = history_df.filter(pl.col('timestamp') >= cutoff_time)
            
            window_features = window_data.group_by(['user_id', 'product_id']).agg([
                pl.len().alias(f'interactions_last_{window}d'),
                pl.sum(pl.col('action_type') == 'AT_Purchase').alias(f'purchases_last_{window}d'),
                pl.sum(pl.col('action_type') == 'AT_View').alias(f'views_last_{window}d')
            ])
            
            dfs.append(window_features)
        
        # Join all window features
        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, on=['user_id', 'product_id'], how='outer')
        
        return result.fill_null(0)
    
    @register('user_stats', category='user')
    def generate_user_stats(self, history_df, target_df):
        """Generate user-level statistics"""
        return history_df.group_by('user_id').agg([
            pl.len().alias('user_total_interactions'),
            pl.sum(pl.col('action_type') == 'AT_Purchase').alias('user_total_purchases'),
            pl.sum(pl.col('action_type') == 'AT_View').alias('user_total_views'),
            pl.n_unique('product_id').alias('user_unique_products')
        ])
    
    @register('user_preferences', category='user')
    def generate_user_preferences(self, history_df, target_df):
        """Generate user category preferences"""
        if 'product_category' not in history_df.columns:
            logger.warning("product_category column not found, skipping user_preferences")
            return pl.DataFrame()
        
        # Calculate user preference for each category
        return history_df.filter(
            pl.col('action_type') == 'AT_Purchase'
        ).group_by(['user_id', 'product_category']).agg(
            pl.len().alias('category_purchases')
        ).pivot(
            index='user_id',
            columns='product_category',
            values='category_purchases'
        ).fill_null(0)
    
    @register('product_stats', category='product')
    def generate_product_stats(self, history_df, target_df):
        """Generate product-level statistics"""
        return history_df.group_by('product_id').agg([
            pl.len().alias('product_total_interactions'),
            pl.sum(pl.col('action_type') == 'AT_Purchase').alias('product_total_purchases'),
            pl.sum(pl.col('action_type') == 'AT_View').alias('product_total_views'),
            pl.n_unique('user_id').alias('product_unique_users')
        ])
    
    @register('category_stats', category='product')
    def generate_category_stats(self, history_df, target_df):
        """Generate category-level statistics"""
        if 'product_category' not in history_df.columns:
            logger.warning("product_category column not found, skipping category_stats")
            return pl.DataFrame()
            
        category_stats = history_df.group_by('product_category').agg([
            pl.len().alias('category_total_interactions'),
            pl.sum(pl.col('action_type') == 'AT_Purchase').alias('category_total_purchases'),
            pl.n_unique('product_id').alias('category_unique_products')
        ])
        
        # Join to products
        product_categories = history_df.select(
            'product_id', 'product_category'
        ).unique()
        
        return product_categories.join(
            category_stats, on='product_category'
        )
    
    @register('novelty', category='advanced')
    def generate_novelty(self, history_df, target_df):
        """Calculate product novelty based on popularity"""
        total_users = history_df['user_id'].n_unique()
        
        product_popularity = history_df.filter(
            pl.col('action_type') == 'AT_Purchase'
        ).group_by('product_id').agg(
            unique_users=pl.n_unique('user_id')
        ).with_columns(
            novelty_score=1 - (pl.col('unique_users') / total_users)
        )
        
        return product_popularity.select('product_id', 'novelty_score')
    
    @register('serendipity', category='advanced', depends_on=['novelty'])
    def generate_serendipity(self, history_df, target_df):
        """Calculate serendipity potential"""
        # Find user purchase history
        user_purchases = history_df.filter(
            pl.col('action_type') == 'AT_Purchase'
        ).group_by(['user_id', 'product_id']).agg(
            pl.lit(1).alias('has_purchased')
        )
        
        # Join with novelty
        novelty = self.features['novelty']
        
        # Create all user-product combinations from target
        user_product_pairs = target_df.select('user_id', 'product_id').unique()
        
        # Join purchase history
        pairs_with_history = user_product_pairs.join(
            user_purchases, on=['user_id', 'product_id'], how='left'
        ).join(
            novelty, on='product_id', how='left'
        ).with_columns([
            pl.col('has_purchased').fill_null(0),
            pl.col('novelty_score').fill_null(0.5)
        ])
        
        # Calculate serendipity score
        return pairs_with_history.with_columns(
            serendipity_score=(1 - pl.col('has_purchased')) * pl.col('novelty_score')
        )


# ======================================================
# Model Factory
# ======================================================

class Model:
    """Base model class"""
    
    def __init__(self, name, params=None):
        self.name = name
        self.params = params or {}
        self.model = None
    
    def train(self, train_features, train_labels):
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
    
    def __init__(self, params=None):
        super().__init__('catboost', params)
        
        from catboost import CatBoostClassifier
        
        default_params = {
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'early_stopping_rounds': 50,
            'verbose': 100
        }
        
        self.params = {**default_params, **(params or {})}
        self.model = CatBoostClassifier(**self.params)
    
    def train(self, train_features, train_labels, eval_set=None):
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
            self.model.fit(train_pool, eval_set=eval_pool)
        else:
            # Train without evaluation
            self.model.fit(train_pool)
        
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
    
    def __init__(self, params=None):
        super().__init__('lightgbm', params)
        
        import lightgbm as lgb
        
        default_params = {
            'num_iterations': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'objective': 'binary',
            'metric': 'auc',
            'early_stopping_rounds': 50,
            'verbose': 100
        }
        
        self.params = {**default_params, **(params or {})}
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
    
    def __init__(self, config):
        self.config = config
        self.models = {
            'catboost': CatBoostModel,
            'lightgbm': LightGBMModel,
            # Add more models as needed
        }
    
    def create_model(self, model_type):
        """Create a model instance based on type"""
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Get model parameters from config
        model_params = self.config.get('models', model_type)
        
        # Create and return model instance
        return self.models[model_type](model_params)


# ======================================================
# Experiment Framework
# ======================================================

class Experiment:
    """Experiment framework for running and tracking recommender experiments"""
    
    def __init__(self, name, config):
        """Initialize experiment with configuration"""
        self.name = name
        self.config = config
        self.data_loader = DataLoader(config)
        self.feature_factory = FeatureFactory()
        self.feature_factory.set_config(config)
        self.model_factory = ModelFactory(config)
        self.results = {}
        
        # Create output directory if needed
        results_dir = config.get('output', 'results_dir')
        os.makedirs(results_dir, exist_ok=True)
        
        # Setup experiment-specific logging
        self.logger = logging.getLogger(f'experiment-{name}')
        
        # Add file handler for experiment
        log_file = os.path.join(results_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)
    
    def run(self, feature_sets, model_type):
        """Run a complete experiment with given feature sets and model"""
        self.logger.info(f"Starting experiment: {self.name}")
        self.logger.info(f"Feature sets: {feature_sets}")
        self.logger.info(f"Model type: {model_type}")
        
        start_time = time.time()
        
        # Load and preprocess data
        train_df, test_df = self.data_loader.load_data()
        train_df = self.data_loader.preprocess(train_df)
        test_df = self.data_loader.preprocess(test_df)
        
        # Create validation splits
        validation_splits = self.data_loader.create_validation_splits()
        
        # Run cross-validation
        cv_results = self._run_cross_validation(
            validation_splits, feature_sets, model_type
        )
        
        # Train final model
        final_model = self._train_final_model(
            train_df, feature_sets, model_type
        )
        
        # Create test predictions
        test_predictions = self._predict_test(
            final_model, train_df, test_df, feature_sets
        )
        
        # Save results
        experiment_results = {
            'name': self.name,
            'feature_sets': feature_sets,
            'model_type': model_type,
            'cv_results': cv_results,
            'test_predictions': test_predictions,
            'runtime': time.time() - start_time
        }
        
        self.results = experiment_results
        
        # Save results to file
        self._save_results()
        
        self.logger.info(f"Experiment completed in {time.time() - start_time:.2f} seconds")
        return experiment_results
    
    def _run_cross_validation(self, validation_splits, feature_sets, model_type):
        """Run cross-validation on validation splits"""
        cv_results = []
        
        for fold_idx, (train_df, val_df) in enumerate(validation_splits):
            self.logger.info(f"Processing fold {fold_idx+1}/{len(validation_splits)}")
            
            # Split each fold into history and target periods for feature generation
            train_history = train_df.filter(
                pl.col('timestamp') < train_df['timestamp'].max() - timedelta(days=7)
            )
            train_target = train_df.filter(
                pl.col('timestamp') >= train_df['timestamp'].max() - timedelta(days=7)
            )
            
            # Generate features
            train_features = self.feature_factory.generate_features(
                train_history, train_target, feature_sets
            )
            train_features_df = self.feature_factory.join_features()
            
            # Setup target variable
            train_target = train_df.filter(
                pl.col('action_type').is_in(["AT_View", "AT_CartUpdate"])
            ).with_columns(
                target=pl.when(pl.col('action_type') == "AT_View").then(0).otherwise(1)
            )['target']
            
            # Generate validation features
            val_features = self.feature_factory.generate_features(
                train_df, val_df, feature_sets
            )
            val_features_df = self.feature_factory.join_features()
            
            # Setup validation target
            val_target = val_df.filter(
                pl.col('action_type').is_in(["AT_View", "AT_CartUpdate"])
            ).with_columns(
                target=pl.when(pl.col('action_type') == "AT_View").then(0).otherwise(1)
            )['target']
            
            # Create and train model
            model = self.model_factory.create_model(model_type)
            model.train(
                train_features_df, 
                train_target,
                eval_set=(val_features_df, val_target)
            )
            
            # Make predictions
            val_preds = model.predict(val_features_df)
            
            # Calculate metrics
            fold_metrics = self._calculate_metrics(val_target, val_preds)
            
            # Add fold results
            cv_results.append({
                'fold': fold_idx,
                'metrics': fold_metrics,
                'feature_importance': model.get_feature_importance()
            })
            
            self.logger.info(f"Fold {fold_idx+1} metrics: {fold_metrics}")
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in cv_results[0]['metrics'].keys():
            avg_metrics[metric] = np.mean([r['metrics'][metric] for r in cv_results])
        
        cv_summary = {
            'folds': cv_results,
            'average_metrics': avg_metrics
        }
        
        self.logger.info(f"Cross-validation average metrics: {avg_metrics}")
        return cv_summary
    
    def _train_final_model(self, train_df, feature_sets, model_type):
        """Train final model on all training data"""
        self.logger.info("Training final model on all data")
        
        # Split data for feature generation
        latest_cutoff = train_df['timestamp'].max() - timedelta(days=7)
        history_df = train_df.filter(pl.col('timestamp') < latest_cutoff)
        target_df = train_df.filter(pl.col('timestamp') >= latest_cutoff)
        
        # Generate features
        self.feature_factory.generate_features(
            history_df, target_df, feature_sets
        )
        features_df = self.feature_factory.join_features()
        
        # Setup target variable
        target = train_df.filter(
            pl.col('action_type').is_in(["AT_View", "AT_CartUpdate"])
        ).with_columns(
            target=pl.when(pl.col('action_type') == "AT_View").then(0).otherwise(1)
        )['target']
        
        # Create and train model
        model = self.model_factory.create_model(model_type)
        model.train(features_df, target)
        
        # Save model if configured
        if self.config.get('output', 'save_models'):
            model_path = os.path.join(
                self.config.get('output', 'results_dir'),
                f"{self.name}_final_model.model"
            )
            model.save(model_path)
            self.logger.info(f"Saved final model to {model_path}")
        
        return model
    
    def _predict_test(self, model, train_df, test_df, feature_sets):
        """Generate predictions for test data"""
        self.logger.info("Generating test predictions")
        
        # Generate features for test data using all training data as history
        self.feature_factory.generate_features(
            train_df, test_df, feature_sets
        )
        test_features_df = self.feature_factory.join_features()
        
        # Make predictions
        test_predictions = model.predict(test_features_df)
        
        # Create submission dataframe
        submission_df = test_df.select('index', 'request_id')
        submission_df = submission_df.with_columns(
            predict=pl.Series(test_predictions)
        ).sort(
            'predict', descending=True
        ).select(
            'index', 'request_id'
        )
        
        # Save predictions if configured
        if self.config.get('output', 'save_predictions'):
            submission_path = os.path.join(
                self.config.get('output', 'results_dir'),
                f"{self.name}_submission.csv"
            )
            submission_df.write_csv(submission_path)
            self.logger.info(f"Saved submission to {submission_path}")
        
        return submission_df
    
    def _calculate_metrics(self, true_labels, predictions):
        """Calculate evaluation metrics"""
        metrics_dict = {}
        
        # Basic classification metrics
        metrics_dict['auc'] = roc_auc_score(true_labels, predictions)
        metrics_dict['logloss'] = log_loss(true_labels, predictions)
        
        # Calculate ranking metrics if specified
        if 'ndcg@10' in self.config.get('metrics'):
            metrics_dict['ndcg@10'] = ndcg_score(
                [true_labels], [predictions], k=10
            )
        
        # Add more metrics as needed
        
        return metrics_dict
    
    def _save_results(self):
        """Save experiment results to file"""
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        results_path = os.path.join(
            self.config.get('output', 'results_dir'),
            f"{self.name}_results.json"
        )
        
        # Convert numpy values to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pl.DataFrame):
                return "DataFrame (not serialized)"
            return obj
        
        # Deep copy and convert results
        serializable_results = {}
        for k, v in self.results.items():
            if isinstance(v, dict):
                serializable_results[k] = {
                    sk: convert_to_serializable(sv) for sk, sv in v.items()
                }
            else:
                serializable_results[k] = convert_to_serializable(v)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        self.logger.info(f"Saved results to {results_path}")


# ======================================================
# Metrics and Evaluation Utilities
# ======================================================

class RankingMetrics:
    """Ranking metrics for recommender systems"""
    
    @staticmethod
    def map_at_k(true_relevance, predicted_scores, k=10):
        """
        Calculate MAP@K (Mean Average Precision at K)
        
        Parameters:
        -----------
        true_relevance: List of lists of binary relevance values
        predicted_scores: List of lists of predicted scores
        k: Cutoff for evaluation
        
        Returns:
        --------
        MAP@K score
        """
        def ap_at_k(y_true, y_pred, k):
            """Calculate AP@K for a single query"""
            if np.sum(y_true) == 0:
                return 0.0
                
            sorted_indices = np.argsort(y_pred)[::-1]
            top_k_indices = sorted_indices[:k]
            y_true_k = np.array(y_true)[top_k_indices]

            cumulative_precision = 0.0
            relevant_seen = 0
            
            for i in range(len(y_true_k)):
                if y_true_k[i]:
                    relevant_seen += 1
                    precision_at_i = relevant_seen / (i + 1)
                    cumulative_precision += precision_at_i

            return cumulative_precision / max(1, np.sum(y_true))
        
        # Calculate AP@K for each query
        aps = [ap_at_k(y_true, y_pred, k) 
              for y_true, y_pred in zip(true_relevance, predicted_scores)]
        
        # Return mean of AP@K values
        return np.mean(aps)
    
    @staticmethod
    def ndcg_at_k(true_relevance, predicted_scores, k=10):
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain at K)
        
        Parameters:
        -----------
        true_relevance: List of lists of binary or graded relevance values
        predicted_scores: List of lists of predicted scores
        k: Cutoff for evaluation
        
        Returns:
        --------
        NDCG@K score
        """
        return ndcg_score(true_relevance, predicted_scores, k=k)
    
    @staticmethod
    def novelty_at_k(recommendations, popularity_df, k=10):
        """
        Calculate Novelty@K
        
        Parameters:
        -----------
        recommendations: DataFrame with user_id, product_id, and predicted scores
        popularity_df: DataFrame with product_id and popularity
        k: Cutoff for evaluation
        
        Returns:
        --------
        Novelty@K score
        """
        # Join recommendations with popularity
        recs_with_pop = recommendations.join(
            popularity_df, on='product_id', how='left'
        ).fill_null(0)
        
        # Group by user and get top-K recommendations
        novelty_scores = []
        
        for user_id, user_recs in recs_with_pop.group_by('user_id'):
            # Sort by predicted score and take top K
            top_k = user_recs.sort('predict', descending=True).head(k)
            
            # Calculate novelty as 1 - popularity
            novelty = (1 - top_k['popularity']).mean()
            novelty_scores.append(novelty)
        
        # Return mean novelty across all users
        return np.mean(novelty_scores)
    
    @staticmethod
    def serendipity_at_k(recommendations, user_history, k=10):
        """
        Calculate Serendipity@K
        
        Parameters:
        -----------
        recommendations: DataFrame with user_id, product_id, and predicted scores
        user_history: DataFrame with user purchase history
        k: Cutoff for evaluation
        
        Returns:
        --------
        Serendipity@K score
        """
        # Join recommendations with user history
        recs_with_history = recommendations.join(
            user_history, on=['user_id', 'product_id'], how='left'
        ).with_columns(
            has_purchased=pl.col('has_purchased').fill_null(0)
        )
        
        # Group by user and calculate serendipity
        serendipity_scores = []
        
        for user_id, user_recs in recs_with_history.group_by('user_id'):
            # Sort by predicted score and take top K
            top_k = user_recs.sort('predict', descending=True).head(k)
            
            # Calculate serendipity as prediction score * (1 - has_purchased)
            serendipity = (top_k['predict'] * (1 - top_k['has_purchased'])).mean()
            serendipity_scores.append(serendipity)
        
        # Return mean serendipity across all users
        return np.mean(serendipity_scores)


# ======================================================
# Visualization Utilities
# ======================================================

class Visualizer:
    """Utilities for visualizing experiment results"""
    
    @staticmethod
    def plot_metrics_comparison(experiments, metric='ndcg@10', figsize=(10, 6)):
        """
        Plot comparison of metrics across experiments
        
        Parameters:
        -----------
        experiments: Dictionary of experiment results
        metric: Metric to compare
        figsize: Figure size (width, height)
        
        Returns:
        --------
        Matplotlib figure
        """
        # Extract metric values
        exp_names = []
        metric_values = []
        
        for exp_name, exp_results in experiments.items():
            exp_names.append(exp_name)
            metric_values.append(exp_results['cv_results']['average_metrics'][metric])
        
        # Create plot
        plt.figure(figsize=figsize)
        bars = plt.bar(exp_names, metric_values)
        
        # Add labels and title
        plt.xlabel('Experiment')
        plt.ylabel(metric)
        plt.title(f'Comparison of {metric} Across Experiments')
        plt.xticks(rotation=45, ha='right')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_feature_importance(feature_importance, top_n=20, figsize=(12, 8)):
        """
        Plot feature importance
        
        Parameters:
        -----------
        feature_importance: Dictionary of feature importance values
        top_n: Number of top features to display
        figsize: Figure size (width, height)
        
        Returns:
        --------
        Matplotlib figure
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)
        
        # Take top N features
        top_features = sorted_features[:top_n]
        
        # Unpack feature names and importance values
        feature_names, importance_values = zip(*top_features)
        
        # Create horizontal bar plot
        plt.figure(figsize=figsize)
        plt.barh(range(len(feature_names)), importance_values, align='center')
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_metrics_heatmap(experiments, metrics=None, figsize=(12, 8)):
        """
        Plot heatmap of metrics across experiments
        
        Parameters:
        -----------
        experiments: Dictionary of experiment results
        metrics: List of metrics to include (defaults to all)
        figsize: Figure size (width, height)
        
        Returns:
        --------
        Matplotlib figure
        """
        # Extract experiment names and metrics
        exp_names = list(experiments.keys())
        
        if metrics is None:
            # Get metrics from first experiment
            first_exp = next(iter(experiments.values()))
            metrics = list(first_exp['cv_results']['average_metrics'].keys())
        
        # Create data matrix
        data = np.zeros((len(exp_names), len(metrics)))
        
        for i, exp_name in enumerate(exp_names):
            for j, metric in enumerate(metrics):
                data[i, j] = experiments[exp_name]['cv_results']['average_metrics'][metric]
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(data, annot=True, fmt='.4f', cmap='YlGnBu',
                   xticklabels=metrics, yticklabels=exp_names)
        plt.xlabel('Metrics')
        plt.ylabel('Experiments')
        plt.title('Metrics Comparison Heatmap')
        plt.tight_layout()
        
        return plt.gcf()


# ======================================================
# Main Execution Functions
# ======================================================

def run_single_experiment(experiment_name, feature_sets, model_type, config=None):
    """Run a single experiment with specific configuration"""
    if config is None:
        config = Config()
    
    # Create and run experiment
    experiment = Experiment(experiment_name, config)
    results = experiment.run(feature_sets, model_type)
    
    # Print summary
    print(f"\nExperiment: {experiment_name}")
    print(f"Feature sets: {feature_sets}")
    print(f"Model: {model_type}")
    print("\nCross-validation metrics:")
    for metric, value in results['cv_results']['average_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nRuntime: {results['runtime']:.2f} seconds")
    
    return results


def run_experiment_grid(base_name, feature_sets_list, model_types, config=None):
    """Run a grid of experiments with different feature sets and models"""
    if config is None:
        config = Config()
    
    results = {}
    
    for features in feature_sets_list:
        for model in model_types:
            # Create experiment name
            feature_str = "_".join(features)
            exp_name = f"{base_name}_{feature_str}_{model}"
            
            # Run experiment
            print(f"\n{'='*50}")
            print(f"Running experiment: {exp_name}")
            print(f"{'='*50}")
            
            exp_results = run_single_experiment(exp_name, features, model, config)
            results[exp_name] = exp_results
    
    # Create visualizations
    visualizer = Visualizer()
    
    # Plot metrics comparison
    for metric in config.get('metrics'):
        if any(metric in exp['cv_results']['average_metrics'] for exp in results.values()):
            fig = visualizer.plot_metrics_comparison(results, metric)
            plt.savefig(os.path.join(config.get('output', 'results_dir'), 
                                    f"{base_name}_{metric}_comparison.png"))
            plt.close(fig)
    
    # Plot metrics heatmap
    fig = visualizer.plot_metrics_heatmap(results)
    plt.savefig(os.path.join(config.get('output', 'results_dir'), 
                            f"{base_name}_metrics_heatmap.png"))
    plt.close(fig)
    
    return results


def train_final_submission(experiment_name, feature_sets, model_type, config=None):
    """Train final model on all data and create submission"""
    if config is None:
        config = Config()
    
    print(f"\n{'='*50}")
    print(f"Training final submission model: {experiment_name}")
    print(f"Feature sets: {feature_sets}")
    print(f"Model: {model_type}")
    print(f"{'='*50}")
    
    # Create experiment
    experiment = Experiment(f"{experiment_name}_final", config)
    
    # Load and preprocess data
    train_df, test_df = experiment.data_loader.load_data()
    train_df = experiment.data_loader.preprocess(train_df)
    test_df = experiment.data_loader.preprocess(test_df)
    
    # Train final model
    model = experiment._train_final_model(train_df, feature_sets, model_type)
    
    # Generate predictions
    submission = experiment._predict_test(model, train_df, test_df, feature_sets)
    
    # Save submission
    submission_path = os.path.join(
        config.get('output', 'results_dir'),
        f"{experiment_name}_final_submission.csv"
    )
    submission.write_csv(submission_path)
    
    print(f"\nFinal submission saved to: {submission_path}")
    return submission_path


# ======================================================
# Example Usage
# ======================================================

if __name__ == "__main__":
    # Create configuration
    config = Config()
    
    # Example 1: Run a single experiment
    result = run_single_experiment(
        "basic_experiment",
        ["basic", "temporal"],
        "catboost",
        config
    )
    
    # Example 2: Run a grid of experiments
    feature_combinations = [
        ["basic"],
        ["basic", "temporal"],
        ["basic", "temporal", "user"],
        ["basic", "temporal", "user", "product"]
    ]
    
    model_types = ["catboost", "lightgbm"]
    
    grid_results = run_experiment_grid(
        "grid_experiments",
        feature_combinations,
        model_types,
        config
    )
    
    # Example 3: Train final submission with best configuration
    submission_path = train_final_submission(
        "best_model",
        ["basic", "temporal", "user", "product"],
        "catboost",
        config
    )
```
