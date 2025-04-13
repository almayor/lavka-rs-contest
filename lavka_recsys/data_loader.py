import polars as pl
from typing import Tuple, List
from datetime import datetime

from .config import Config
from .custom_logging import get_logger

class DataLoader:
    """Data loading and processing"""
    
    def __init__(self, config: Config):
        """Initialize with configuration"""
        self.config = config
        self.train_df = None
        self.test_df = None
        self.logger = get_logger(self.__class__.__name__)
    
    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load training and testing data."""
        self.logger.debug("Loading data...")
        
        # Load train data
        train_path = self.config.get('data.train_path')
        self.train_df = pl.read_parquet(train_path)
        
        # Load test data
        test_path = self.config.get('data.test_path')
        self.test_df = pl.read_parquet(test_path)

        # Preprocess data
        self.logger.debug("Preprocessing data...")
        self._preprocess_data()
        
        # Sample if needed (for faster debugging)
        sample_size = self.config.get('data.sample_size')
        if sample_size is not None:
            self.train_df = self.train_df.sample(
                n=sample_size, 
                seed=42
            )
        
        self.logger.info(f"Loaded train data: {len(self.train_df)} rows")        
        self.logger.info(f"Loaded test data: {len(self.test_df)} rows")
        
        return self.train_df, self.test_df
    
    def create_validation_split(
            self,
            validation_days: int = 14,
            train_days: int = 40
        ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Create a single validation split using the last X days of data.
        Args:
            validation_days (int): Number of days for validation.
            train_days (int): Number of days for training.
        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: 
                Tuple of (history_df, train_df, val_df)
        """
        self.logger.debug("Creating validation split...")
        
        # Ensure data is sorted by timestamp
        df = self.train_df.sort('timestamp')
        
        # Use last N days for validation
        max_time = df['timestamp'].max()
        days_in_seconds = validation_days * 24 * 60 * 60
        val_start_time = max_time - days_in_seconds
        
        # Use previous M days for training
        train_window_seconds = train_days * 24 * 60 * 60
        train_start_time = val_start_time - train_window_seconds
        
        # Split data
        history_df = df.filter(pl.col('timestamp') < train_start_time)
        train_df = df.filter(
            (pl.col('timestamp') >= train_start_time) & 
            (pl.col('timestamp') < val_start_time)
        )
        val_df = df.filter(pl.col('timestamp') >= val_start_time)
        
        # Log the split details
        history_end = datetime.fromtimestamp(train_start_time)
        train_end = datetime.fromtimestamp(val_start_time)
        val_end = datetime.fromtimestamp(max_time)
        
        self.logger.info(f"Validation split: history until {history_end}")
        self.logger.info(f"Training: {history_end} to {train_end}")
        self.logger.info(f"Validation: {train_end} to {val_end}")
        self.logger.info(f"Split data: {len(history_df)} history, {len(train_df)} train, {len(val_df)} validation rows")
        
        return history_df, train_df, val_df
    
    def create_final_split(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Create a final split for training the production model.
        Uses all data until the last 14 days for history, and the last 14 days for training.
        
        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: Tuple of (history_df, train_df)
        """
        self.logger.debug("Creating final split for production model...")
        
        # Ensure data is sorted by timestamp
        df = self.train_df.sort('timestamp')
        
        # Use last 14 days for final training
        max_time = df['timestamp'].max()
        days_in_seconds = 14 * 24 * 60 * 60
        split_time = max_time - days_in_seconds
        
        # Split data
        history_df = df.filter(pl.col('timestamp') < split_time)
        train_df = df.filter(pl.col('timestamp') >= split_time)
        
        # Log the split details
        split_date = datetime.fromtimestamp(split_time)
        
        self.logger.info(f"Final split: history until {split_date}")
        self.logger.info(f"Final training: {split_date} to {datetime.fromtimestamp(max_time)}")
        self.logger.info(f"Final split data: {len(history_df)} history, {len(train_df)} train rows")
        
        return history_df, train_df
    
    def _preprocess_data(self):
        """
        Preprocess data for training and testing by doing necessary data type conversions
        and transformations.
        """
        self.train_df.with_columns(
            pl.col('timestamp').mul(1000).cast(pl.Datetime("ms"))
        )
        self.logger.debug("Converted timestamp to datetime")