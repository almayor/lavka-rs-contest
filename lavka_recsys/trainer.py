import time
import os
import pickle
import hashlib
import json
import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

from .config import Config
from .data_loader import DataLoader
from .feature_factory import FeatureFactory
from .model_factory import ModelFactory, Model
from .time_splitter import TimeSplitter, SplitType
from .custom_logging import get_logger
from .metrics import RankingMetrics

class Trainer:
    """
    Unified trainer class that handles all model training strategies.
    """
    
    def __init__(self, config: Config, data_loader: DataLoader, 
                feature_factory: FeatureFactory, model_factory: ModelFactory):
        """Initialize the trainer with required components"""
        self.config = config
        self.data_loader = data_loader
        self.feature_factory = feature_factory
        self.model_factory = model_factory
        self.time_splitter = TimeSplitter(config)
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize caching configuration
        self.model_caching_enabled = config.get('model_caching.enabled', True)
        
    def get_model_cache_path(self, config_dict: Dict) -> str:
        """Generate a cache path for the trained model based on configuration"""
        # Create a cache directory if it doesn't exist
        cache_dir = self.config.get('output.model_cache_dir', 'results/model_cache')
        
        try:
            os.makedirs(cache_dir, exist_ok=True)
            self.logger.info(f"Using model cache directory: {cache_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create model cache directory {cache_dir}: {str(e)}")
            # Fallback to a default directory
            cache_dir = 'model_cache'
            os.makedirs(cache_dir, exist_ok=True)
            self.logger.info(f"Using fallback model cache directory: {cache_dir}")
        
        # Create a hash of the config for the cache key
        config_str = json.dumps(config_dict, sort_keys=True).encode('utf-8')
        config_hash = hashlib.md5(config_str).hexdigest()[:12]
        
        # Return the full cache path
        return os.path.join(cache_dir, f"model_{config_hash}.pkl")
    
    def try_load_cached_model(self, cache_key_dict: Dict) -> Optional[Model]:
        """
        Try to load a previously cached model.
        
        Args:
            cache_key_dict: Dictionary with model configuration for cache key
            
        Returns:
            Model object if successfully loaded, None otherwise
        """
        if not self.model_caching_enabled:
            return None
            
        cache_path = self.get_model_cache_path(cache_key_dict)
        
        if os.path.exists(cache_path):
            try:
                start_time = time.time()
                self.logger.info(f"Found cached model at {cache_path}, attempting to load...")
                with open(cache_path, 'rb') as f:
                    model = pickle.load(f)
                load_time = time.time() - start_time
                self.logger.info(f"Successfully loaded cached model in {load_time:.2f} seconds")
                
                # Make sure data is loaded for evaluation
                if self.data_loader.train_df is None or self.data_loader.train_df.is_empty():
                    self.logger.info("Cached model loaded but no training data, loading data...")
                    self.data_loader.load_data()
                    
                return model
            except Exception as e:
                self.logger.warning(f"Failed to load cached model: {str(e)}")
                
                # Remove corrupted cache file if it exists
                try:
                    os.remove(cache_path)
                    self.logger.info(f"Removed corrupted cache file: {cache_path}")
                except:
                    pass
        else:
            self.logger.info(f"No cached model found at {cache_path}")
        
        return None
    
    def save_model_to_cache(self, model: Model, cache_key_dict: Dict) -> bool:
        """
        Save a trained model to cache.
        
        Args:
            model: Trained model to save
            cache_key_dict: Dictionary with model configuration for cache key
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        if not self.model_caching_enabled:
            return False
            
        cache_path = self.get_model_cache_path(cache_key_dict)
        
        try:
            start_time = time.time()
            self.logger.info(f"Saving model to cache at {cache_path}")
            
            # Create a temporary file first, then rename to avoid partial writes
            temp_path = f"{cache_path}.tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Replace the existing file with the new one
            if os.path.exists(cache_path):
                os.remove(cache_path)
            os.rename(temp_path, cache_path)
            
            save_time = time.time() - start_time
            self.logger.info(f"Successfully saved model to cache in {save_time:.2f} seconds")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to save model to cache: {str(e)}")
            
            # Clean up temporary file if it exists
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
                
            return False
    
    def create_cache_key_dict(self, split_type: SplitType, model_params: Optional[Dict] = None, 
                             provided_train_df: Optional[pl.DataFrame] = None) -> Dict:
        """
        Create a dictionary for model cache key based on configuration.
        
        Args:
            split_type: Type of split used for training
            model_params: Optional model parameters (for tuning)
            provided_train_df: Optional training data signature 
            
        Returns:
            Dict: Dictionary for cache key
        """
        # Base cache dictionary with all key configuration elements
        cache_key_dict = {
            'model_type': self.config.get('model.type'),
            'model_config': model_params or self.config.get(f'model.config.{self.config.get("model.type")}'),
            'features': self.config.get('features'),
            'training': {
                'split_type': split_type.value,
                'history_days': self.config.get('training.history_days'),
                'target_days': self.config.get('training.target_days', 1),
                'step_days': self.config.get('training.step_days', 7),
                'max_splits': self.config.get('training.max_splits', 10),
                'validation_days': self.config.get('training.validation_days')
            }
        }
        
        # Add a signature for the training data if provided
        if provided_train_df is not None:
            try:
                min_time = provided_train_df['timestamp'].min().isoformat()
                max_time = provided_train_df['timestamp'].max().isoformat()
                record_count = len(provided_train_df)
                cache_key_dict['data_signature'] = f"{min_time}_{max_time}_{record_count}"
            except:
                # If we can't get this info, just use the number of rows
                cache_key_dict['data_signature'] = f"rows_{len(provided_train_df)}"
        
        return cache_key_dict
    
    def train(self, 
            split_type: SplitType = None,
            model_params: Optional[Dict] = None,
            provided_train_df: Optional[pl.DataFrame] = None) -> Model:
        """
        Train a model using the specified strategy.
        
        Args:
            split_type: Type of split to use (standard, fixed_window, expanding_window)
            model_params: Optional model parameters to use (for tuning)
            provided_train_df: Optional pre-loaded training dataframe
            
        Returns:
            Model: The trained model
        """
        # Use configuration value if not provided
        if split_type is None:
            split_type_str = self.config.get('training.split_type', 'standard')
            if split_type_str == 'fixed_window':
                split_type = SplitType.FIXED_WINDOW
            elif split_type_str == 'expanding_window':
                split_type = SplitType.EXPANDING_WINDOW
            else:  # Default to standard
                split_type = SplitType.STANDARD
        
        start_time = time.time()
        self.logger.info(f"Starting training with {split_type.value} strategy")
        
        # Create cache key and try to load cached model if enabled
        cache_key_dict = self.create_cache_key_dict(split_type, model_params, provided_train_df)
        cached_model = self.try_load_cached_model(cache_key_dict)
        if cached_model:
            self.logger.info("Using cached model - skipping training")
            return cached_model
        
        # Prepare training data
        if provided_train_df is not None:
            self.logger.info("Using provided training dataframe")
            train_df = provided_train_df
        else:
            self.logger.info("Loading training data from data loader")
            train_df, _ = self.data_loader.load_data()
            
        # Get configuration parameters
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        history_days = self.config.get('training.history_days')
        target_days = self.config.get('training.target_days', 1)
        step_days = self.config.get('training.step_days', 7)
        max_splits = self.config.get('training.max_splits', 10)
        validation_days = self.config.get('training.validation_days')
        
        self.logger.info(f"Training configuration:")
        self.logger.info(f"  Split type: {split_type.value}")
        self.logger.info(f"  Target days: {target_days}")
        if split_type == SplitType.FIXED_WINDOW:
            self.logger.info(f"  History days: {history_days}")
        if split_type != SplitType.STANDARD:
            self.logger.info(f"  Step days: {step_days}")
            self.logger.info(f"  Max splits: {max_splits}")
        if validation_days:
            self.logger.info(f"  Validation days: {validation_days}")
        
        # Create time splits based on the specified strategy
        splits = list(self.time_splitter.create_splits(
            train_df,
            split_type=split_type,
            history_days=history_days,
            target_days=target_days, 
            step_days=step_days,
            max_splits=max_splits,
            validation_days=validation_days
        ))
        
        self.logger.info(f"Created {len(splits)} splits")
        
        if not splits:
            self.logger.error("No valid splits were created")
            raise ValueError("No valid splits were created")
        
        # For single standard split, use a simpler approach
        if split_type == SplitType.STANDARD:
            return self._train_with_single_split(splits[0], feature_names, target_name, model_params)
        else:
            return self._train_with_multiple_splits(splits, feature_names, target_name, model_params, split_type)
    
    def _train_with_single_split(self, 
                                split_data: Tuple, 
                                feature_names: List[str], 
                                target_name: str,
                                model_params: Optional[Dict] = None) -> Model:
        """
        Train a model using a single split.
        
        Args:
            split_data: Tuple of (history_df, train_df, val_history_df, val_df)
            feature_names: List of feature names to use
            target_name: Name of the target column
            model_params: Optional model parameters to use
            
        Returns:
            Model: The trained model
        """
        self.logger.info("Training with single split")
        
        # Extract split components
        history_df, train_df, val_history_df, val_df = split_data
        
        # Generate features for training
        train_features, train_target, cat_columns, train_request_ids = self.feature_factory.generate_batch(
            history_df, train_df, feature_names, target_name
        )
        
        # Generate validation features if available
        eval_set = None
        val_request_ids = None
        if len(val_df) > 0:
            val_features, val_target, _, val_request_ids = self.feature_factory.generate_batch(
                val_history_df, val_df, feature_names, target_name
            )
            eval_set = (val_features, val_target, val_request_ids)
            self.logger.info(f"Using {len(val_df)} validation records for evaluation")
        
        # Create and train model
        model = self.model_factory.create_model(model_params)
        self.logger.info(f"Training model with {len(train_df)} records")
        
        train_start_time = time.time()
        model.train(
            train_features, 
            train_target,
            cat_columns=cat_columns,
            train_request_ids=train_request_ids,
            eval_set=eval_set
        )
        train_time = time.time() - train_start_time
        self.logger.info(f"Model training completed in {train_time:.2f} seconds")
        
        # Log validation metrics if available
        if eval_set is not None:
            val_preds = model.predict(val_features)
            if isinstance(val_preds, np.ndarray):
                val_preds = pl.Series(val_preds)
            
            try:
                val_auc = roc_auc_score(val_target, val_preds)
                self.logger.info(f"Validation AUC: {val_auc:.4f}")
                
                # Calculate nDCG if request IDs are available
                if 'val_request_ids' in locals() and val_request_ids is not None:
                    try:
                        val_ndcg = RankingMetrics.ndcg_at_k(val_target, val_preds, val_request_ids, k=10)
                        self.logger.info(f"Validation nDCG@10: {val_ndcg:.4f}")
                    except Exception as e:
                        self.logger.warning(f"Could not calculate validation nDCG@10: {str(e)}")
            except Exception as e:
                self.logger.warning(f"Error calculating validation metrics: {str(e)}")
        
        # Save model to cache if enabled
        cache_key_dict = self.create_cache_key_dict(SplitType.STANDARD, model_params)
        self.save_model_to_cache(model, cache_key_dict)
        
        return model
    
    def _train_with_multiple_splits(self, 
                                  splits: List[Tuple], 
                                  feature_names: List[str], 
                                  target_name: str,
                                  model_params: Optional[Dict] = None,
                                  split_type: SplitType = SplitType.EXPANDING_WINDOW) -> Model:
        """
        Train a model using multiple time splits.
        
        Args:
            splits: List of split tuples (history_df, train_df, val_history_df, val_df)
            feature_names: List of feature names to use
            target_name: Name of the target column
            model_params: Optional model parameters to use
            
        Returns:
            Model: The trained model
        """
        self.logger.info(f"Training with {len(splits)} splits")
        
        # Collect features from all splits
        all_train_features = []
        all_train_targets = []
        all_train_request_ids = []
        all_validation_features = []
        all_validation_targets = []
        all_validation_request_ids = []
        cat_columns_set = set()
        
        # Process each split with a progress bar
        for i, (history_df, train_df, val_history_df, val_df) in enumerate(tqdm(splits, desc="Processing splits")):
            split_start_time = time.time()
            self.logger.info(f"\nProcessing split {i+1}/{len(splits)}")
            
            # Report split information
            history_min = history_df.select(pl.col("timestamp").min()).item() if len(history_df) > 0 else None
            history_max = history_df.select(pl.col("timestamp").max()).item() if len(history_df) > 0 else None
            train_min = train_df.select(pl.col("timestamp").min()).item() if len(train_df) > 0 else None
            train_max = train_df.select(pl.col("timestamp").max()).item() if len(train_df) > 0 else None
            
            self.logger.info(f"  History: {history_min} to {history_max} ({len(history_df)} records)")
            self.logger.info(f"  Train: {train_min} to {train_max} ({len(train_df)} records)")
            
            # Generate features for this split
            feature_start_time = time.time()
            train_features, train_target, cat_columns, train_request_ids = self.feature_factory.generate_batch(
                history_df, train_df, feature_names, target_name
            )
            
            # Track categorical columns
            if cat_columns:
                cat_columns_set.update(cat_columns)
            
            # Add features to collection
            all_train_features.append(train_features)
            all_train_targets.append(train_target)
            if train_request_ids is not None:
                all_train_request_ids.append(train_request_ids)
            
            feature_time = time.time() - feature_start_time
            self.logger.info(f"  Train feature generation: {feature_time:.2f}s, shape: {train_features.shape}")
            
            # Generate validation features if available
            if len(val_df) > 0:
                val_min = val_df.select(pl.col("timestamp").min()).item()
                val_max = val_df.select(pl.col("timestamp").max()).item()
                self.logger.info(f"  Validation: {val_min} to {val_max} ({len(val_df)} records)")
                
                val_feature_start_time = time.time()
                val_features, val_target, _, val_request_ids = self.feature_factory.generate_batch(
                    val_history_df, val_df, feature_names, target_name
                )
                
                # Add to collection
                all_validation_features.append(val_features)
                all_validation_targets.append(val_target)
                if val_request_ids is not None:
                    all_validation_request_ids.append(val_request_ids)
                
                val_feature_time = time.time() - val_feature_start_time
                self.logger.info(f"  Validation feature generation: {val_feature_time:.2f}s, shape: {val_features.shape}")
            
            self.logger.info(f"  Split {i+1} processed in {time.time() - split_start_time:.2f} seconds")
        
        # Combine features from all splits
        self.logger.info("Combining features from all splits")
        
        # Combine training features
        if not all_train_features:
            self.logger.error("No training features generated from any split")
            raise ValueError("No training features generated from any split")
            
        # Convert features to pandas DataFrames for consistent handling
        train_features_df = self._combine_features(all_train_features)
        
        # Combine targets
        if isinstance(all_train_targets[0], pl.Series):
            train_target_series = pl.concat(all_train_targets)
        else:
            train_target_series = pl.Series(np.concatenate(all_train_targets))
        
        # Combine request IDs if available
        train_request_ids_combined = None
        if all_train_request_ids:
            if isinstance(all_train_request_ids[0], pl.Series):
                train_request_ids_combined = pl.concat(all_train_request_ids)
            else:
                train_request_ids_combined = pl.Series(np.concatenate(all_train_request_ids))
        
        self.logger.info(f"Combined training features shape: {train_features_df.shape}")
        self.logger.info(f"Combined training target shape: {len(train_target_series)}")
        
        # Combine validation features if available
        eval_set = None
        val_request_ids_combined = None
        
        if all_validation_features:
            # Combine validation features
            val_features_df = self._combine_features(all_validation_features)
            
            # Combine validation targets
            if isinstance(all_validation_targets[0], pl.Series):
                val_target_series = pl.concat(all_validation_targets)
            else:
                val_target_series = pl.Series(np.concatenate(all_validation_targets))
            
            # Combine validation request IDs if available
            if all_validation_request_ids:
                if isinstance(all_validation_request_ids[0], pl.Series):
                    val_request_ids_combined = pl.concat(all_validation_request_ids)
                else:
                    val_request_ids_combined = pl.Series(np.concatenate(all_validation_request_ids))
            
            # Create evaluation set
            eval_set = (val_features_df, val_target_series)
            
            self.logger.info(f"Combined validation features shape: {val_features_df.shape}")
            self.logger.info(f"Combined validation target shape: {len(val_target_series)}")
        
        # Convert categorical columns set to list
        cat_columns_list = list(cat_columns_set) if cat_columns_set else None
        
        # Handle categorical columns properly
        if cat_columns_list:
            self.logger.info(f"Ensuring categorical columns are properly formatted")
            for col in cat_columns_list:
                if col in train_features_df.columns:
                    train_features_df[col] = train_features_df[col].astype(str)
                    
                if eval_set is not None and col in val_features_df.columns:
                    val_features_df[col] = val_features_df[col].astype(str)
        
        # Create and train the model
        self.logger.info("Training model on combined features")
        model = self.model_factory.create_model(model_params)
        
        # Always set up eval_set with request IDs
        if eval_set is not None and val_request_ids_combined is not None:
            eval_set = (eval_set[0], eval_set[1], val_request_ids_combined)
        
        train_start_time = time.time()
        model.train(
            train_features_df,
            train_target_series,
            cat_columns=cat_columns_list,
            train_request_ids=train_request_ids_combined,
            eval_set=eval_set
        )
        train_time = time.time() - train_start_time
        self.logger.info(f"Model training completed in {train_time:.2f} seconds")
        
        # Calculate and log metrics
        if eval_set is not None and val_request_ids_combined is not None:
            try:
                # Predict on validation set
                val_preds = model.predict(val_features_df)
                if isinstance(val_preds, np.ndarray):
                    val_preds = pl.Series(val_preds)
                
                # Calculate AUC
                val_auc = roc_auc_score(val_target_series, val_preds)
                self.logger.info(f"Validation AUC: {val_auc:.4f}")
                
                # Calculate nDCG@10
                try:
                    val_ndcg = RankingMetrics.ndcg_at_k(val_target_series, val_preds, val_request_ids_combined, k=10)
                    self.logger.info(f"Validation nDCG@10: {val_ndcg:.4f}")
                except Exception as e:
                    self.logger.warning(f"Could not calculate validation nDCG@10: {str(e)}")
            except Exception as e:
                self.logger.warning(f"Error calculating validation metrics: {str(e)}")
        
        # Save model to cache if enabled
        cache_key_dict = self.create_cache_key_dict(split_type, model_params)
        self.save_model_to_cache(model, cache_key_dict)
        
        return model
    
    def _combine_features(self, feature_list: List) -> pd.DataFrame:
        """
        Combine features from multiple splits into a single DataFrame.
        
        Args:
            feature_list: List of feature dataframes
            
        Returns:
            pd.DataFrame: Combined features
        """
        # Convert to pandas DataFrames if not already
        if isinstance(feature_list[0], pl.DataFrame):
            df_list = [f.to_pandas() for f in feature_list]
        else:
            df_list = feature_list
            
        # Get all unique columns
        all_columns = set()
        for df in df_list:
            all_columns.update(df.columns)
            
        # Add missing columns to each DataFrame
        for i, df in enumerate(df_list):
            for col in all_columns:
                if col not in df.columns:
                    df_list[i][col] = None
        
        # Concatenate all DataFrames
        return pd.concat(df_list, axis=0)
        
    def train_with_full_history(self, provided_train_df=None) -> Model:
        """
        Legacy method for backwards compatibility.
        Now redirects to train with EXPANDING_WINDOW type.
        
        Args:
            provided_train_df: Optional pre-loaded training dataframe
            
        Returns:
            Model: The trained model
        """
        self.logger.info("Using expanding window strategy (formerly full_history)")
        
        return self.train(
            split_type=SplitType.EXPANDING_WINDOW,
            provided_train_df=provided_train_df
        )

    def create_validation_split(self, df: pl.DataFrame = None, 
                                    target_days: int = None, 
                                    validation_days: int = None) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Create a validation split for evaluation metrics.
        
        Args:
            df: DataFrame to split (uses data_loader.train_df if None)
            target_days: Number of days for target window (from config if None)
            validation_days: Number of days for validation window (from config if None)
            
        Returns:
            Tuple of (history_df, train_df, val_history_df, val_df)
        """
        # Use provided df or load from data_loader
        if df is None:
            df = self.data_loader.train_df
            
        # Check if dataframe is empty
        if df is None or df.is_empty():
            self.logger.error("Cannot create validation split: DataFrame is empty or None")
            raise ValueError("DataFrame is empty or None")
            
        # Get configuration parameters if not provided
        if target_days is None:
            target_days = self.config.get('training.target_days', 1)
            
        if validation_days is None:
            validation_days = self.config.get('training.validation_days')
            
        # Create standard split
        try:
            splits = list(self.time_splitter.create_splits(
                df, split_type=SplitType.STANDARD,
                target_days=target_days, validation_days=validation_days
            ))
            
            if not splits:
                self.logger.error("No valid validation splits were created")
                raise ValueError("No valid validation splits were created")
                
            return splits[0]
            
        except Exception as e:
            self.logger.error(f"Error creating validation split: {str(e)}")
            
            # Fallback: if we can't create a proper time-based split, create a simple random split
            # This shouldn't be used in production but helps prevent crashes during evaluation
            self.logger.warning("Using fallback random split for validation")
            
            # Sort by timestamp to ensure we're still respecting chronological order
            df = df.sort('timestamp')
            
            # Calculate split point (80% train, 20% validation)
            split_idx = int(len(df) * 0.8)
            if split_idx == 0:
                self.logger.error("DataFrame too small to split")
                raise ValueError("DataFrame too small to split")
                
            # Create splits
            history_train_df = df.slice(0, split_idx)
            val_df = df.slice(split_idx, len(df))
            
            # Since we don't have a clean temporal separation, use the whole history_train for both
            history_df = history_train_df
            train_df = history_train_df
            val_history_df = df  # Use all data for validation history (slight data leakage but prevents crashes)
            
            self.logger.warning(f"Created fallback split: history/train={len(history_df)}, val={len(val_df)} records")
            
            return (history_df, train_df, val_history_df, val_df)

    def evaluate_model(self, model: Model, validation_data: Optional[Tuple] = None) -> Tuple[float, Dict]:
        """
        Evaluate a model on validation data.
        
        Args:
            model: The model to evaluate
            validation_data: Optional tuple of (history_df, train_df, val_history_df, val_df)
                             If not provided, will create a validation split
                             
        Returns:
            Tuple of (score, metrics_dict)
        """
        # Get feature configuration
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        
        # Create validation split if not provided
        if validation_data is None:
            try:
                # Explicitly load data if it's not already loaded
                if self.data_loader.train_df is None or self.data_loader.train_df.is_empty():
                    self.logger.info("Training data not loaded, loading now...")
                    self.data_loader.load_data()
                
                # Check again after loading
                if self.data_loader.train_df is None or self.data_loader.train_df.is_empty():
                    self.logger.error("Cannot create validation split: Train data is still empty after loading")
                    return 0.0, {"error": "Train data is empty even after explicit loading"}
                
                self.logger.info(f"Loaded {len(self.data_loader.train_df)} rows for validation")
                validation_data = self.create_validation_split()
            except Exception as e:
                self.logger.warning(f"Failed to create validation split: {str(e)}")
                return 0.0, {"error": str(e)}
                
        # Unpack validation data
        history_df, train_df, val_history_df, val_df = validation_data
        
        # Check if we have validation data
        if len(val_df) == 0:
            self.logger.warning("No validation data available")
            return 0.0, {"error": "No validation data available"}
            
        # Generate validation features
        val_features, val_target, _, val_request_ids = self.feature_factory.generate_batch(
            val_history_df, val_df, feature_names, target_name
        )
        
        # Check if we need to pass request_ids to the predict method
        model_type = self.config.get('model.type')
        predict_kwargs = {}
        
        # For ranking models, pass request_ids if available
        if model_type == 'catboost_ranker' and val_request_ids is not None:
            predict_kwargs['request_ids'] = val_request_ids
        
        # Predict on validation data
        val_preds = model.predict(val_features, **predict_kwargs)
        if isinstance(val_preds, np.ndarray):
            val_preds = pl.Series(val_preds)
            
        # Calculate metrics
        metrics = {}
        
        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(val_target, val_preds)
            metrics["auc"] = auc
            self.logger.info(f"Validation AUC: {auc:.6f}")
        except Exception as e:
            self.logger.warning(f"Failed to calculate AUC: {str(e)}")
            
        # nDCG@10 if request IDs are available
        if val_request_ids is not None:
            try:
                from .metrics import RankingMetrics
                ndcg = RankingMetrics.ndcg_at_k(val_target, val_preds, val_request_ids, k=10)
                metrics["ndcg@10"] = ndcg
                self.logger.info(f"Validation nDCG@10: {ndcg:.6f}")
            except Exception as e:
                self.logger.warning(f"Failed to calculate nDCG@10: {str(e)}")
                
        # Return the primary score (AUC) and all metrics
        primary_score = metrics.get("auc", 0.0)
        return primary_score, metrics

# Legacy class name for backwards compatibility
EnhancedTrainingPipeline = Trainer