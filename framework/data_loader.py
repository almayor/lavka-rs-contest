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

from .config import Config
from .custom_logging import get_logger

class DataLoader:
    """Data loading and preprocessing"""
    
    def __init__(self, config: Config):
        """
        Initialize with configuration
        Args:
            config (Config): Configuration object containing paths and settings.
        """
        self.config = config
        self.train_df = None
        self.test_df = None
        self.logger = get_logger(self.__class__.__name__)
    
    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Load training and testing data.
        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: Tuple containing training and testing DataFrames.
        """
        self.logger.debug("Loading data...")
        
        # Load train data
        train_path = self.config.get('data.train_path')
        self.train_df = pl.read_parquet(train_path)
        
        # Load test data
        test_path = self.config.get('data.test_path')
        self.test_df = pl.read_parquet(test_path)
        
        # Sample if needed
        sample_size = self.config.get('data.sample_size')
        if sample_size is not None:
            self.train_df = self.train_df.sample(
                n=sample_size, 
                seed=self.config.get('data.random_seed')
            )
        
        self.logger.info(f"Loaded train data: {len(self.train_df)} rows")        
        self.train_df = self._preprocess(self.train_df)
        self.logger.info(f"Loaded test data: {len(self.test_df)} rows")
        self.test_df = self._preprocess(self.test_df)
        return self.train_df, self.test_df
    
    def _preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply preprocessing steps based on configuration. 
        """
        self.logger.debug("Preprocessing data...")
        
        # Make a copy to avoid modifying the original
        processed_df = df.clone()
        
        # Apply preprocessing steps based on config
        if self.config.get('preprocessing.normalize_timestamps'):
            # Ensure timestamps are in a consistent format
            if 'timestamp' in processed_df.columns:
                # Convert to datetime if it's not already
                if processed_df['timestamp'].dtype != pl.Datetime:
                    processed_df.with_columns(
                        (pl.col("timestamp") * 1_000).cast(pl.Datetime("ms")).alias("timestamp")
                    )
                self.logger.info("Normalized timestamps")
        
        if self.config.get('preprocessing.clean_text'):
            # Apply text cleaning to relevant columns
            text_columns = []
            
            for col in text_columns:
                if col in processed_df.columns:
                    # Simple text cleaning example
                    processed_df = processed_df.with_columns(
                        pl.col(col).str.strip().str.to_lowercase()
                    )
            self.logger.info("Cleaned text columns")
        
        return processed_df
    
    def create_fixed_splits(self, target_n_days: int):
        """
        Create a split into history, training and validation where both
        training and validation targets encompass a fixed number of days.
        Args:
            target_ndats (int): Number of days to use for targetting
        """
        pass
        

    def create_validation_splits(self, n_folds: None|int = None) -> List[Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]]:
        """
        Create validation splits based on the configuration.
        Args:
            n_folds (Union[int, None]): Number of folds (if None, taken from config)
        Returns:
            List[Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]]: List of tuples containing
                history, training, and validation DataFrames.
        """
        n_folds = n_folds or self.config.get('validation.n_folds')
        validation_method = self.config.get('validation.method')
        
        if validation_method == 'temporal':
            return self._create_temporal_splits(n_folds)
        else:
            raise ValueError(f"Unknown validation method: {validation_method}")
    
    def _create_temporal_splits(self, n_folds: int):
        """
        Create time-based validation folds.
        Args:
            n_folds (int): Number of folds
        Returns:
            List[Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]]: List of tuples containing
                history, training, and validation DataFrames.
        """        
        # Ensure data is sorted by timestamp
        df = self.train_df.sort('timestamp')
        
        # Calculate time range and fold duration
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        time_range = max_time - min_time
        fold_duration = time_range / (n_folds + 2)  # +2 as we need separate folds for feature generation and training
        
        folds = []
        for i in range(n_folds):
            # Calculate time boundaries
            history_start_time = min_time + fold_duration * i
            train_start_time = history_start_time + fold_duration 
            val_start_time = train_start_time + fold_duration
            val_end_time = val_start_time + fold_duration
            
            # Create train and validation sets
            history_df = df.filter((pl.col('timestamp') >= history_start_time) &
                                   (pl.col('timestamp') < train_start_time))
            train_df = df.filter((pl.col('timestamp') >= train_start_time) &
                                 (pl.col('timestamp') < val_start_time))
            val_df = df.filter((pl.col('timestamp') >= val_start_time) & 
                              (pl.col('timestamp') < val_end_time))
            
            history_df = self._clean_history(history_df)
            folds.append((history_df, train_df, val_df))
        
        self.logger.info(f"Created {len(folds)} temporal validation folds")
        return folds

    
    def _clean_history(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean training history based on configuration"""
        if self.config.get('history_cleaning.remove_lurkers'):
            n_old = df.height
            valid_users = (
                df.filter(pl.col("action_type") != "AT_View")
                .select("user_id")
                .unique()
            )
            df = df.join(valid_users, on="user_id", how="inner")

            n_new = df.height
            total_user_count = df.select("user_id").n_unique()
            invalid_user_count = total_user_count - valid_users.height
            self.logger.info(
                f'Removed {invalid_user_count} users who only watch (lurkers); ' +
                f'rows reduced from {n_old} to {n_new}'
            )

        return df
