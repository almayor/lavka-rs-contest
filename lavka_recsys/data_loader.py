import polars as pl
from typing import Tuple, List
from datetime import datetime, timedelta

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
        sample_fraction = self.config.get('data.sample_fraction')
        if sample_fraction is not None:
            self.train_df = self.train_df.sample(
                fraction=sample_fraction,
                with_replacement=False,
                seed=42
            )
        
        self.logger.info(f"Loaded train data: {len(self.train_df)} rows")        
        self.logger.info(f"Loaded test data: {len(self.test_df)} rows")
        
        return self.train_df, self.test_df
    
    def create_validation_split(
            self,
            validation_days: int = None,
            target_days: int = None
        ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Create a single validation split using TimeSplitter with STANDARD split type.
        
        Args:
            validation_days (int): Optional override for validation days
            target_days (int): Optional override for target days
            
        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: 
                Tuple of (history_df, train_df, val_df) where train_df is the target window
        """
        self.logger.debug("Creating validation split using TimeSplitter...")
        
        # Get configuration values if not provided
        if validation_days is None:
            validation_days = self.config.get('training.validation_days')
        if target_days is None:
            target_days = self.config.get('training.target_days', 7)
            
        # Create time splitter
        from .time_splitter import TimeSplitter, SplitType
        time_splitter = TimeSplitter(self.config)
        
        # Get the most recent standard split
        splits = list(time_splitter.create_splits(
            self.train_df,
            split_type=SplitType.STANDARD,
            target_days=target_days,
            validation_days=validation_days
        ))
        
        if not splits:
            self.logger.warning("No valid splits were created, falling back to simple split")
            # Fallback to a simple split
            df = self.train_df.sort('timestamp')
            max_time = df['timestamp'].max()
            history_df = df.filter(pl.col('timestamp') < (max_time - pl.duration(days=target_days)))
            train_df = df.filter(pl.col('timestamp') >= (max_time - pl.duration(days=target_days)))
            val_df = pl.DataFrame(schema=df.schema)  # Empty dataframe with same schema
            return history_df, train_df, val_df
        
        # Extract the components from the first (and only) split
        history_df, train_df, val_history_df, val_df = splits[0]
        
        # Log the split information
        history_end = history_df['timestamp'].max() if len(history_df) > 0 else None
        train_start = train_df['timestamp'].min() if len(train_df) > 0 else None
        train_end = train_df['timestamp'].max() if len(train_df) > 0 else None
        val_start = val_df['timestamp'].min() if len(val_df) > 0 else None
        val_end = val_df['timestamp'].max() if len(val_df) > 0 else None
        
        # Format the datetimes
        def format_time(dt):
            if dt is not None:
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            return "unknown"
            
        history_end_str = format_time(history_end)
        train_start_str = format_time(train_start)
        train_end_str = format_time(train_end)
        val_start_str = format_time(val_start)
        val_end_str = format_time(val_end)
        
        self.logger.info(f"Validation split: history until {history_end_str}")
        self.logger.info(f"Training: {train_start_str} to {train_end_str} ({len(train_df)} records)")
        if len(val_df) > 0:
            self.logger.info(f"Validation: {val_start_str} to {val_end_str} ({len(val_df)} records)")
        else:
            self.logger.info("No validation data available")
        self.logger.info(f"Split data: {len(history_df)} history, {len(train_df)} train, {len(val_df)} validation rows")
        
        return history_df, train_df, val_df
    
    def create_final_split(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Create a final split for training the production model using TimeSplitter.
        Uses the configured target_days for the final training window.
        
        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: Tuple of (history_df, train_df)
        """
        self.logger.debug("Creating final split for production model using TimeSplitter...")
        
        # Get configuration values
        target_days = self.config.get('training.target_days', 30)
            
        # Create time splitter
        from .time_splitter import TimeSplitter, SplitType
        time_splitter = TimeSplitter(self.config)
        
        # Get the most recent standard split without validation
        splits = list(time_splitter.create_splits(
            self.train_df,
            split_type=SplitType.STANDARD,
            target_days=target_days,
            validation_days=None
        ))
        
        if not splits:
            self.logger.warning("No valid splits were created, falling back to simple split")
            # Fallback to a simple split
            df = self.train_df.sort('timestamp')
            max_time = df['timestamp'].max()
            history_df = df.filter(pl.col('timestamp') < (max_time - pl.duration(days=target_days)))
            train_df = df.filter(pl.col('timestamp') >= (max_time - pl.duration(days=target_days)))
            return history_df, train_df
        
        # Extract the components from the first (and only) split
        # We only need history_df and train_df
        history_df, train_df, _, _ = splits[0]
        
        # Get actual datetime values from the dataframes for logging
        # This approach is safer than trying to format the expressions directly
        split_time_actual = history_df['timestamp'].max() if len(history_df) > 0 else None
        max_time_actual = train_df['timestamp'].max() if len(train_df) > 0 else None
        
        # Format the datetimes for logging
        if split_time_actual is not None:
            split_time_str = split_time_actual.strftime("%Y-%m-%d %H:%M:%S")
        else:
            split_time_str = "unknown"
            
        if max_time_actual is not None:
            max_time_str = max_time_actual.strftime("%Y-%m-%d %H:%M:%S")
        else:
            max_time_str = "unknown"
        
        self.logger.info(f"Final split: history until {split_time_str}")
        self.logger.info(f"Final training: {split_time_str} to {max_time_str}")
        self.logger.info(f"Final split data: {len(history_df)} history, {len(train_df)} train rows")
        
        return history_df, train_df
        
    def create_kaggle_simulation_split(
            self, 
            test_days: int = 30, 
            train_days: int = 30,
            validation_ratio: float = 0.2
        ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Create a data split that simulates the Kaggle test scenario for evaluation purposes.
        
        This splits the data into four parts:
        1. History: All data before the training period (for feature generation)
        2. Train: A period of data for model training
        3. Validation: Portion of training data reserved for validation (based on validation_ratio)
        4. Test: Final period that simulates the Kaggle test set
        
        Args:
            test_days: Number of days to reserve for testing
            train_days: Number of days to use for training before the test period
            validation_ratio: Portion of training data to use for validation (0-1)
            
        Returns:
            Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]: Tuple containing:
                - History data (for feature generation)
                - Training data (for model training)
                - Validation data (for model validation)
                - Test data (for evaluation, with ground truth labels)
        """
        self.logger.debug("Creating Kaggle simulation split for model evaluation...")
        
        # Ensure data is sorted by timestamp
        df = self.train_df.sort('timestamp')
        
        # Find the cutoff date for test
        max_date = df['timestamp'].max()
        test_start_date = max_date - pl.duration(days=test_days)
        
        # Define training period before test
        train_start_date = test_start_date - pl.duration(days=train_days)
        
        # Extract test data
        test_df = df.filter(pl.col('timestamp') >= test_start_date)
        
        # Extract training data
        all_train_df = df.filter(
            (pl.col('timestamp') >= train_start_date) & 
            (pl.col('timestamp') < test_start_date)
        )
        
        # Extract history data (everything before training)
        history_df = df.filter(pl.col('timestamp') < train_start_date)
        
        # Split training data into train and validation chronologically
        if validation_ratio > 0 and validation_ratio < 1:
            # Sort training data by time
            all_train_df = all_train_df.sort('timestamp')
            
            # Calculate split point - use last X% for validation (chronological split)
            split_idx = int(len(all_train_df) * (1 - validation_ratio))
            
            # Split into train and validation
            train_df = all_train_df.slice(0, split_idx)
            val_df = all_train_df.slice(split_idx)
        else:
            # No validation split requested
            train_df = all_train_df
            val_df = pl.DataFrame(schema=all_train_df.schema)  # Empty DataFrame with same schema
        
        # Get actual datetime values for logging
        history_end = history_df['timestamp'].max() if len(history_df) > 0 else None
        train_start = train_df['timestamp'].min() if len(train_df) > 0 else None
        train_end = train_df['timestamp'].max() if len(train_df) > 0 else None
        val_start = val_df['timestamp'].min() if len(val_df) > 0 else None
        val_end = val_df['timestamp'].max() if len(val_df) > 0 else None
        test_start = test_df['timestamp'].min() if len(test_df) > 0 else None
        test_end = test_df['timestamp'].max() if len(test_df) > 0 else None
        
        # Format timestamps
        def format_time(dt):
            if dt is not None:
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            return "unknown"
        
        self.logger.info(f"Kaggle simulation split:")
        self.logger.info(f"  History: until {format_time(history_end)} ({len(history_df)} rows)")
        self.logger.info(f"  Training: {format_time(train_start)} to {format_time(train_end)} ({len(train_df)} rows)")
        
        if len(val_df) > 0:
            self.logger.info(f"  Validation: {format_time(val_start)} to {format_time(val_end)} ({len(val_df)} rows)")
        else:
            self.logger.info(f"  Validation: None (validation_ratio = {validation_ratio})")
            
        self.logger.info(f"  Test: {format_time(test_start)} to {format_time(test_end)} ({len(test_df)} rows)")
        
        return history_df, train_df, val_df, test_df
    
    def _preprocess_data(self):
        """
        Preprocess data for training and testing by doing necessary data type conversions
        and transformations.
        """
        # Convert timestamp column to datetime
        self.train_df = self.train_df.with_columns(
            pl.col('timestamp').mul(1000).cast(pl.Datetime("ms"))
        )
        
        # Do the same for test data if it exists
        if self.test_df is not None:
            self.test_df = self.test_df.with_columns(
                pl.col('timestamp').mul(1000).cast(pl.Datetime("ms"))
            )
            
        self.logger.debug("Converted timestamp to datetime format")
