import time
import os
import pickle
import hashlib
import json
import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm
from sklearn.metrics import ndcg_score

from .config import Config
from .data_loader import DataLoader
from .feature_factory import FeatureFactory
from .model_factory import ModelFactory, Model
from .time_splitter import TimeSplitter
from .cached_feature_generator import CachedFeatureGenerator
from .custom_logging import get_logger
from .metrics import RankingMetrics

class EnhancedTrainingPipeline:
    """
    Enhanced training pipeline that leverages sliding window data splits with all 
    historical data for each target period.
    """
    
    def __init__(self, config: Config, data_loader: DataLoader, 
                feature_factory: FeatureFactory, model_factory: ModelFactory):
        self.config = config
        self.data_loader = data_loader
        self.feature_factory = CachedFeatureGenerator(feature_factory)
        self.model_factory = model_factory
        self.time_splitter = TimeSplitter(config)
        self.logger = get_logger(self.__class__.__name__)
    
    def get_model_cache_path(self, config_dict):
        """Generate a cache path for the trained model based on configuration"""
        # Create a cache directory if it doesn't exist
        cache_dir = self.config.get('cache.directory', 'model_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a hash of the config for the cache key
        config_str = json.dumps(config_dict, sort_keys=True).encode('utf-8')
        config_hash = hashlib.md5(config_str).hexdigest()[:12]
        
        # Return the full cache path
        return os.path.join(cache_dir, f"model_{config_hash}.pkl")
    
    def try_load_cached_model(self, cache_key_dict):
        """Try to load a previously cached model"""
        cache_path = self.get_model_cache_path(cache_key_dict)
        
        if os.path.exists(cache_path):
            try:
                self.logger.info(f"Found cached model at {cache_path}, attempting to load...")
                with open(cache_path, 'rb') as f:
                    model = pickle.load(f)
                self.logger.info(f"Successfully loaded cached model")
                return model
            except Exception as e:
                self.logger.warning(f"Failed to load cached model: {str(e)}")
        
        return None
    
    def save_model_to_cache(self, model, cache_key_dict):
        """Save a trained model to cache"""
        cache_path = self.get_model_cache_path(cache_key_dict)
        
        try:
            self.logger.info(f"Saving model to cache at {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f"Successfully saved model to cache")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to save model to cache: {str(e)}")
            return False
    
    def train_with_full_history(self, provided_train_df=None) -> Model:
        """
        Train a model using all time splits, respecting temporal boundaries to prevent data leakage.
        This approach:
        1. Processes each split individually to generate features with proper history context
        2. Combines the features from all splits while preserving temporal integrity
        3. Trains a single model on all pre-processed features
        
        Args:
            provided_train_df: Optional pre-loaded training dataframe. If provided,
                              this will be used instead of loading data from scratch.
        
        Returns:
            Model: The trained model
        """
        # Set up training data
        start_time = time.time()
        self.logger.info("Starting full history time-aware training approach")
        
        # Create a cache key from our configuration and data properties
        cache_enabled = self.config.get('model_caching.enabled', True)
        
        if cache_enabled:
            # Prepare cache key information (key factors that determine model output)
            cache_key_dict = {
                'model_type': self.config.get('model.type'),
                'model_config': self.config.get(f'model.config.{self.config.get("model.type")}'),
                'features': self.config.get('features'),
                'training': {
                    'target_days': self.config.get('training.target_days', 1),
                    'step_days': self.config.get('training.step_days', 7),
                    'max_splits': self.config.get('training.max_splits', 10),
                    'validation_days': self.config.get('training.validation_days')
                }
            }
            
            # If training data is provided, include info about it in the cache key
            if provided_train_df is not None:
                try:
                    min_time = provided_train_df['timestamp'].min().isoformat()
                    max_time = provided_train_df['timestamp'].max().isoformat()
                    record_count = len(provided_train_df)
                    cache_key_dict['data_signature'] = f"{min_time}_{max_time}_{record_count}"
                except:
                    # If we can't get this info, just use the number of rows
                    cache_key_dict['data_signature'] = f"rows_{len(provided_train_df)}"
            
            # Try to load a cached model first
            cached_model = self.try_load_cached_model(cache_key_dict)
            if cached_model:
                self.logger.info("Using cached model - skipping training")
                return cached_model
        
        if provided_train_df is not None:
            # Use the provided training data (important for simulation)
            self.logger.info("Using provided training dataframe")
            train_df = provided_train_df
        else:
            # Load all training data if none provided
            self.logger.info("Loading training data from data loader")
            train_df, _ = self.data_loader.load_data()
            
        # Log data time range
        if len(train_df) > 0:
            min_time = train_df['timestamp'].min()
            max_time = train_df['timestamp'].max()
            self.logger.info(f"Training data range: {min_time} to {max_time} ({len(train_df)} records)")
        
        # Prepare for splitting
        target_days = self.config.get('training.target_days', 1)
        step_days = self.config.get('training.step_days', 7)
        max_splits = self.config.get('training.max_splits', 10)
        validation_days = self.config.get('training.validation_days')
        
        self.logger.info(f"Creating time splits with target_days={target_days}, "
                         f"step_days={step_days}, max_splits={max_splits}")
        
        if validation_days:
            self.logger.info(f"Using {validation_days} days after each target window for validation")
        else:
            self.logger.info("Validation is disabled")
        
        # Get features and model settings
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        model_type = self.config.get("model.type")
        
        self.logger.info(f"Using model type: {model_type}")
        self.logger.info(f"Features: {feature_names}")
        self.logger.info(f"Target: {target_name}")
        
        # First collect all the splits to know the total number for progress bar
        self.logger.info("Collecting time splits...")
        splits = list(self.time_splitter.create_sliding_window_splits(
            train_df, 
            target_days=target_days, 
            step_days=step_days, 
            max_splits=max_splits,
            validation_days=validation_days
        ))
        
        self.logger.info(f"Total number of splits: {len(splits)}")
        total_records = sum(len(train_df) for _, train_df, _, _ in splits)
        self.logger.info(f"Total records across all splits: {total_records}")
        
        # Collect features for each split individually, respecting temporal boundaries
        all_train_features = []
        all_train_targets = []
        all_train_request_ids = []
        all_validation_features = []
        all_validation_targets = []
        all_validation_request_ids = []
        cat_columns_set = set()
        
        # Process each split individually with tqdm progress bar, respecting temporal boundaries
        for i, (history_df, train_df, val_history_df, val_df) in enumerate(tqdm(splits, desc="Processing splits")):
            split_start_time = time.time()
            self.logger.info(f"\nProcessing split {i+1}/{len(splits)}")
            
            # Report sizes and time ranges
            history_min = history_df.select(pl.col("timestamp").min()).item()
            history_max = history_df.select(pl.col("timestamp").max()).item()
            train_min = train_df.select(pl.col("timestamp").min()).item() if len(train_df) > 0 else None
            train_max = train_df.select(pl.col("timestamp").max()).item() if len(train_df) > 0 else None
            
            self.logger.info(f"  History: {history_min} to {history_max} ({len(history_df)} records)")
            self.logger.info(f"  Train: {train_min} to {train_max} ({len(train_df)} records)")
            
            # Generate features for training data using only available history for this split
            # This prevents data leakage by ensuring we only use history that would have been
            # available at that point in time
            feature_start_time = time.time()
            train_features, train_target, cat_columns, train_request_ids = self.feature_factory.generate_batch(
                history_df, train_df, feature_names, target_name
            )
            
            # Track all categorical columns we encounter
            if cat_columns:
                cat_columns_set.update(cat_columns)
            
            # Add these features to our collection 
            all_train_features.append(train_features)
            all_train_targets.append(train_target)
            if train_request_ids is not None:
                all_train_request_ids.append(train_request_ids)
            
            feature_time = time.time() - feature_start_time
            self.logger.info(f"  Train feature generation completed in {feature_time:.2f} seconds")
            self.logger.info(f"  Train features shape: {train_features.shape}")
            
            # Generate validation features if we have validation data for this split
            if len(val_df) > 0:
                val_min = val_df.select(pl.col("timestamp").min()).item()
                val_max = val_df.select(pl.col("timestamp").max()).item()
                self.logger.info(f"  Validation: {val_min} to {val_max} ({len(val_df)} records)")
                
                val_feature_start_time = time.time()
                
                # Use the proper history for validation features (this includes the training data
                # for this split, since in a real scenario we would have trained on this data
                # before validating)
                val_features, val_target, _, val_request_ids = self.feature_factory.generate_batch(
                    val_history_df, val_df, feature_names, target_name
                )
                
                # Add validation features to our collection
                all_validation_features.append(val_features)
                all_validation_targets.append(val_target)
                if val_request_ids is not None:
                    all_validation_request_ids.append(val_request_ids)
                
                val_feature_time = time.time() - val_feature_start_time
                self.logger.info(f"  Validation feature generation completed in {val_feature_time:.2f} seconds")
                self.logger.info(f"  Validation features shape: {val_features.shape}")
                
                # Report validation target distribution
                if val_target is not None:
                    val_pos_count = val_target.sum()
                    val_total = len(val_target)
                    val_pos_ratio = val_pos_count / val_total if val_total > 0 else 0
                    self.logger.info(f"  Validation target distribution: {val_pos_count}/{val_total} positive examples ({val_pos_ratio:.2%})")
            
            # Count training target distribution
            if train_target is not None:
                pos_count = train_target.sum()
                total = len(train_target)
                pos_ratio = pos_count / total if total > 0 else 0
                self.logger.info(f"  Training target distribution: {pos_count}/{total} positive examples ({pos_ratio:.2%})")
            
            self.logger.info(f"  Split {i+1} processed in {time.time() - split_start_time:.2f} seconds")
        
        # Combine all the features
        self.logger.info("Combining features from all splits...")
        
        # Combine training features
        if len(all_train_features) > 0:
            # Convert features to pandas dataframes if they aren't already
            if isinstance(all_train_features[0], pl.DataFrame):
                train_features_df_list = [f.to_pandas() for f in all_train_features]
            else:
                train_features_df_list = all_train_features
                
            # Ensure we have the same columns in all dataframes (important for consistent concatenation)
            all_columns = set()
            for df in train_features_df_list:
                all_columns.update(df.columns)
                
            # Add missing columns to each dataframe
            for i, df in enumerate(train_features_df_list):
                for col in all_columns:
                    if col not in df.columns:
                        train_features_df_list[i][col] = None
            
            # Concatenate the dataframes
            train_features_df = pd.concat(train_features_df_list, axis=0)
            
            # Combine targets
            if isinstance(all_train_targets[0], pl.Series):
                train_target_series = pl.concat(all_train_targets)
            else:
                train_target_series = pl.Series(np.concatenate(all_train_targets))
            
            # Combine request IDs if we have them
            train_request_ids_combined = None
            if all_train_request_ids:
                train_request_ids_combined = pl.concat(all_train_request_ids) if isinstance(all_train_request_ids[0], pl.Series) else pl.Series(np.concatenate(all_train_request_ids))
            
            self.logger.info(f"Combined training features shape: {train_features_df.shape}")
            self.logger.info(f"Combined training target shape: {len(train_target_series)}")
        else:
            self.logger.warning("No training features found!")
            return None
        
        # Combine validation features if we have any
        eval_set = None
        val_request_ids_combined = None
        if all_validation_features:
            # Convert features to pandas dataframes if they aren't already
            if isinstance(all_validation_features[0], pl.DataFrame):
                val_features_df_list = [f.to_pandas() for f in all_validation_features]
            else:
                val_features_df_list = all_validation_features
            
            # Ensure we have the same columns in all dataframes
            all_columns = set(train_features_df.columns)  # Use same columns as training data
            for df in val_features_df_list:
                all_columns.update(df.columns)
                
            # Add missing columns to each dataframe
            for i, df in enumerate(val_features_df_list):
                for col in all_columns:
                    if col not in df.columns:
                        val_features_df_list[i][col] = None
            
            # Concatenate the dataframes
            val_features_df = pd.concat(val_features_df_list, axis=0)
            
            # Combine targets
            if isinstance(all_validation_targets[0], pl.Series):
                val_target_series = pl.concat(all_validation_targets)
            else:
                val_target_series = pl.Series(np.concatenate(all_validation_targets))
            
            # Combine request IDs if we have them
            if all_validation_request_ids:
                val_request_ids_combined = pl.concat(all_validation_request_ids) if isinstance(all_validation_request_ids[0], pl.Series) else pl.Series(np.concatenate(all_validation_request_ids))
            
            # Create evaluation set
            eval_set = (val_features_df, val_target_series)
            
            self.logger.info(f"Combined validation features shape: {val_features_df.shape}")
            self.logger.info(f"Combined validation target shape: {len(val_target_series)}")
            
            # Report validation target distribution
            val_pos_count = val_target_series.sum()
            val_total = len(val_target_series)
            val_pos_ratio = val_pos_count / val_total if val_total > 0 else 0
            self.logger.info(f"Validation target distribution: {val_pos_count}/{val_total} positive examples ({val_pos_ratio:.2%})")
        
        # Report training target distribution
        pos_count = train_target_series.sum()
        total = len(train_target_series)
        pos_ratio = pos_count / total if total > 0 else 0
        self.logger.info(f"Training target distribution: {pos_count}/{total} positive examples ({pos_ratio:.2%})")
        
        # Create and train a single model on the combined features
        self.logger.info("Training model on the combined features...")
        model = self.model_factory.create_model()
        
        # Convert cat_columns_set to list
        cat_columns_list = list(cat_columns_set) if cat_columns_set else None
        
        # Handle categorical columns properly
        if cat_columns_list:
            self.logger.info(f"Ensuring categorical columns are properly formatted for CatBoost")
            for col in cat_columns_list:
                if col in train_features_df.columns:
                    train_features_df[col] = train_features_df[col].astype(str)
                    
                if eval_set is not None and col in val_features_df.columns:
                    val_features_df[col] = val_features_df[col].astype(str)
        
        # Train the model
        train_start_time = time.time()
        model.train(
            train_features_df,
            train_target_series,
            cat_columns=cat_columns_list,
            eval_set=eval_set
        )
        
        train_time = time.time() - train_start_time
        self.logger.info(f"Model training completed in {train_time:.2f} seconds")
        
        # Calculate validation metrics if we have validation data
        if eval_set is not None and val_request_ids_combined is not None:
            try:
                # Make predictions on validation set
                val_preds = model.predict(val_features_df)
                
                # Calculate nDCG@10
                try:
                    val_ndcg = RankingMetrics.ndcg_at_k(val_target_series, pl.Series(val_preds), val_request_ids_combined, k=10)
                    self.logger.info(f"Validation nDCG@10: {val_ndcg:.4f}")
                except Exception as e:
                    self.logger.warning(f"Could not calculate validation nDCG@10: {str(e)}")
            except Exception as e:
                self.logger.warning(f"Error calculating validation metrics: {str(e)}")
        
        # Calculate training metrics
        try:
            if train_request_ids_combined is not None:
                # Make predictions on training data
                train_preds = model.predict(train_features_df)
                
                # Calculate nDCG@10
                try:
                    train_ndcg = RankingMetrics.ndcg_at_k(train_target_series, pl.Series(train_preds), train_request_ids_combined, k=10)
                    self.logger.info(f"Training nDCG@10: {train_ndcg:.4f}")
                except Exception as e:
                    self.logger.warning(f"Could not calculate training nDCG@10: {str(e)}")
        except Exception as e:
            self.logger.warning(f"Error calculating training metrics: {str(e)}")
        
        total_time = time.time() - start_time
        self.logger.info(f"Full history time-aware training completed in {total_time:.2f} seconds")
        
        # Save model to cache if caching is enabled
        if cache_enabled:
            self.save_model_to_cache(model, cache_key_dict)
        
        return model