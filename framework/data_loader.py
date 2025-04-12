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
        """Initialize with configuration"""
        self.config = config
        self.train_df = None
        self.test_df = None
        self.logger = get_logger(self.__class__.__name__)
    
    def load_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Load training and testing data"""
        self.logger.debug("Loading data...")
        
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
        
        self.logger.info(f"Loaded train data: {len(self.train_df)} rows")
        self.logger.info(f"Loaded test data: {len(self.test_df)} rows")
        
        self.train_df = self._preprocess(self.train_df)
        self.test_df = self._preprocess(self.test_df)
        return self.train_df, self.test_df
    
    def _preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply preprocessing steps based on configuration"""
        self.logger.debug("Preprocessing data...")
        
        # Make a copy to avoid modifying the original
        processed_df = df.clone()
        
        # Apply preprocessing steps based on config
        if self.config.get('preprocessing', 'remove_duplicates'):
            processed_df = processed_df.unique()
            self.logger.info("Removed duplicates")
        
        if self.config.get('preprocessing', 'normalize_timestamps'):
            # Ensure timestamps are in a consistent format
            if 'timestamp' in processed_df.columns:
                # Convert to datetime if it's not already
                if processed_df['timestamp'].dtype != pl.Datetime:
                    processed_df = processed_df.with_columns(
                        pl.col('timestamp').cast(pl.Datetime)
                    )
                self.logger.info("Normalized timestamps")
        
        if self.config.get('preprocessing', 'clean_text'):
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
    
    def create_validation_splits(self):
        """Create training/validation splits based on config"""
        validation_method = self.config.get('validation', 'method')
        
        if validation_method == 'temporal':
            return self._create_temporal_splits()
        else:
            raise ValueError(f"Unknown validation method: {validation_method}")
    
    def _create_temporal_splits(self):
        """Create time-based validation folds"""
        n_folds = self.config.get('validation', 'n_folds')
        
        # Ensure data is sorted by timestamp
        df = self.train_df.sort('timestamp')
        
        # Calculate time range and fold duration
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        time_range = max_time - min_time
        fold_duration = time_range / (n_folds + 1)  # +1 to leave first fold for train feature generation
        
        folds = []
        for i in range(n_folds):
            # Calculate time boundaries
            train_start_time = min_time + fold_duration
            train_end_time = train_start_time + fold_duration
            val_start_time = train_end_time
            val_end_time = val_start_time + fold_duration
            
            # Create train and validation sets
            history_df = df.filter(pl.col('timestamp') < train_start_time)
            train_df = df.filter((pl.col('timestamp') >= train_start_time) &
                                 (pl.col('timestamp') < train_end_time))
            val_df = df.filter((pl.col('timestamp') >= val_start_time) & 
                              (pl.col('timestamp') < val_end_time))
            
            folds.append((history_df, train_df, val_df))
        
        self.logger.info(f"Created {len(folds)} temporal validation folds")
        return folds

    
    def _clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean training data based on configuration"""
        #TODO – remove duplicated actions
        #TODO – remove users who only watch
        return df
