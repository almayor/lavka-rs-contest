import os
import polars as pl
from typing import Tuple, List, Generator, Optional
import hashlib
import pickle
from datetime import timedelta
from enum import Enum

from .config import Config
from .custom_logging import get_logger

class SplitType(Enum):
    """Enumeration of different types of time-based data splits."""
    STANDARD = "standard"  # Creates a single split with default time windows
    FIXED_WINDOW = "fixed_window"  # Multiple splits with fixed size history window
    EXPANDING_WINDOW = "expanding_window"  # Multiple splits with expanding history window

class TimeSplitter:
    """
    Unified time series splitter that supports multiple splitting strategies
    while maintaining consistent interface and preventing data leakage.
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration"""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    def create_splits(
            self, df: pl.DataFrame, 
            split_type: SplitType = SplitType.STANDARD,
            history_days: Optional[int] = None, 
            target_days: Optional[int] = None, 
            step_days: Optional[int] = None, 
            max_splits: Optional[int] = None,
            validation_days: Optional[int] = None
        ) -> Generator[Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame], None, None]:
        """
        Create time-based splits using the specified strategy.
        
        Args:
            df: Complete DataFrame containing all data
            split_type: Type of split to create (standard, fixed_window, expanding_window)
            history_days: Number of days to use for history (required for fixed_window)
            target_days: Number of days to use for target
            step_days: Number of days to step forward for each new split
            max_splits: Maximum number of splits to generate (None = no maximum)
            validation_days: Number of days to use for validation after target period
                           (None = no validation)
        
        Returns:
            Generator yielding tuples of (history_df, train_df, val_history_df, val_df)
        """
        # Use configuration values if not provided
        target_days = target_days or self.config.get('training.target_days', 1)
        step_days = step_days or self.config.get('training.step_days', 7)
        max_splits = max_splits or self.config.get('training.max_splits')
        validation_days = validation_days or self.config.get('training.validation_days')
        
        # Choose the appropriate split implementation based on split_type
        if split_type == SplitType.STANDARD:
            return self.create_standard_splits(
                df, target_days, validation_days
            )
        elif split_type == SplitType.EXPANDING_WINDOW:
            return self.create_expanding_window_splits(
                df, target_days, step_days, max_splits, validation_days
            )
        elif split_type == SplitType.FIXED_WINDOW:
            # Ensure we have history_days for fixed window
            if history_days is None:
                history_days = self.config.get('training.history_days')
                if history_days is None:
                    raise ValueError("history_days is required for fixed window splits")
                    
            return self.create_fixed_window_splits(
                df, history_days, target_days, step_days, max_splits, validation_days
            )
        else:
            raise ValueError(f"Unknown split type: {split_type}")
    
    def create_standard_splits(
            self, df: pl.DataFrame, 
            target_days: int = 1,
            validation_days: Optional[int] = None
        ) -> Generator[Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame], None, None]:
        """
        Yield a single history-target-validation split using the most recent data.
        
        Args:
            df: Complete DataFrame containing all data
            target_days: Number of days to use for target
            validation_days: Number of days to use for validation after target period
                            (None = no validation)
        
        Yields:
            Tuple of (history_df, train_df, val_history_df, val_df)
            - history_df: All data up to target_start for feature generation
            - train_df: Data in the target period (target_start to target_end)
            - val_history_df: All data up to validation_start (history + train) for feature generation
            - val_df: Data in the validation period (validation_start to validation_end)
        """
        # Sort by timestamp to ensure chronological order
        df = df.sort('timestamp')
        
        # Working with concrete values - get min/max timestamps as Python datetime objects
        min_timestamp = df.select(pl.col("timestamp").min()).item()
        max_timestamp = df.select(pl.col("timestamp").max()).item()
        
        # Log the actual data range for debugging
        self.logger.info(f"Data range: {min_timestamp} to {max_timestamp}")
        
        # For standard split, we work backwards from the end of the data
        # If validation is enabled, leave room for validation
        if validation_days and validation_days > 0:
            target_end = max_timestamp - timedelta(days=validation_days)
            self.logger.info(f"Using target end: {target_end} to allow for validation window")
        else:
            target_end = max_timestamp
            self.logger.info(f"Using target end: {target_end} (dataset end)")
        
        # Calculate target start
        target_start = target_end - timedelta(days=target_days)
        
        # Create dataframes with simple timestamp comparisons
        history_df = df.filter(pl.col('timestamp') < target_start)
        target_df = df.filter(
            (pl.col('timestamp') >= target_start) & 
            (pl.col('timestamp') < target_end)
        )
        
        # Create validation dataframe if validation is enabled
        val_df = pl.DataFrame()
        val_history_df = pl.concat([history_df, target_df], how='vertical')
        
        if validation_days and validation_days > 0:
            val_start = target_end
            val_end = val_start + timedelta(days=validation_days)
            
            # Create validation split
            val_df = df.filter(
                (pl.col('timestamp') >= val_start) & 
                (pl.col('timestamp') < val_end)
            )
            
            # Only use validation if we have at least 10 records (prevent tiny validation sets)
            if len(val_df) < 10:
                self.logger.warning(f"Not enough validation data: only {len(val_df)} records (minimum 10 required)")
                val_df = pl.DataFrame()
        
        # Log details
        history_min = history_df.select(pl.col("timestamp").min()).item() if len(history_df) > 0 else None
        history_max = history_df.select(pl.col("timestamp").max()).item() if len(history_df) > 0 else None
        target_min = target_df.select(pl.col("timestamp").min()).item() if len(target_df) > 0 else None
        target_max = target_df.select(pl.col("timestamp").max()).item() if len(target_df) > 0 else None
        
        self.logger.info(f"Standard split:")
        self.logger.info(f"  History: {history_min} to {history_max} ({len(history_df)} records)")
        self.logger.info(f"  Train: {target_min} to {target_max} ({len(target_df)} records)")
        
        if len(val_df) > 0:
            val_min = val_df.select(pl.col("timestamp").min()).item()
            val_max = val_df.select(pl.col("timestamp").max()).item()
            self.logger.info(f"  Validation: {val_min} to {val_max} ({len(val_df)} records)")
        else:
            self.logger.info("  No validation data available")
        
        # Yield the split data (single yield for standard split)
        yield (history_df, target_df, val_history_df, val_df)
    
    def create_expanding_window_splits(
            self, df: pl.DataFrame, 
            target_days: int = 1, 
            step_days: int = 1, 
            max_splits: Optional[int] = None,
            validation_days: Optional[int] = None  # Separate time window for validation after target
        ) -> Generator[Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame], None, None]:
        """
        Yield multiple history-target-validation pairs using all history data up to each target period,
        with a dedicated validation period after the target period.
        
        Args:
            df: Complete DataFrame containing all data
            target_days: Number of days to use for target
            step_days: Number of days to step forward for each new split
            max_splits: Maximum number of splits to generate (None = no maximum)
            validation_days: Number of days to use for validation after target period
                            (None = no validation)
        
        Yields:
            Tuples of (history_df, train_df, val_history_df, val_df)
            - history_df: All data up to target_start for feature generation
            - train_df: Data in the target period (target_start to target_end)
            - val_history_df: All data up to validation_start (history + train) for feature generation
            - val_df: Data in the validation period (validation_start to validation_end)
        """
        # Sort by timestamp to ensure chronological order
        df = df.sort('timestamp')
        
        # Working with concrete values - get min/max timestamps as Python datetime objects
        min_timestamp = df.select(pl.col("timestamp").min()).item()
        max_timestamp = df.select(pl.col("timestamp").max()).item()
        
        # Log the actual data range for debugging
        self.logger.info(f"Data range: {min_timestamp} to {max_timestamp}")
        
        # Start from a point that allows for validation after the last training period
        # If validation is enabled, we need to leave room at the end for validation
        if validation_days and validation_days > 0:
            # Start validation_days before max_timestamp to leave room for validation
            # This ensures first split has enough data for validation
            current_end = max_timestamp - timedelta(days=validation_days)
            self.logger.info(f"Adjusted start point to allow validation window: {current_end} "
                           f"({validation_days} days before dataset end)")
        else:
            current_end = max_timestamp
            
        split_count = 0
        
        # Keep going until we've reached the beginning of data or max_splits
        while True:
            # Calculate target window boundaries
            target_end = current_end
            target_start = target_end - timedelta(days=target_days)
            
            # We've gone too far back if target_start is before min_timestamp
            if target_start < min_timestamp:
                self.logger.info("Reached the beginning of the data, stopping")
                break
            
            # Create dataframes with simple timestamp comparisons
            history_df = df.filter(pl.col('timestamp') < target_start)
            target_df = df.filter(
                (pl.col('timestamp') >= target_start) & 
                (pl.col('timestamp') < target_end)
            )
            
            # Create validation dataframe if validation is enabled
            val_df = pl.DataFrame()
            val_history_df = pl.concat([history_df, target_df], how='vertical')
            
            # Calculate validation window boundaries (if validation is enabled)
            # Note: We only create validation if we have enough data
            if validation_days and validation_days > 0:
                val_start = target_end
                val_end = val_start + timedelta(days=validation_days)
                
                # Only create validation split if there's data available (and enough of it)
                temp_val_df = df.filter(
                    (pl.col('timestamp') >= val_start) & 
                    (pl.col('timestamp') < val_end)
                )
                
                # Only use validation if we have at least 10 records (prevent tiny validation sets)
                if len(temp_val_df) >= 10:
                    val_df = temp_val_df
                else:
                    self.logger.info(f"Skipping validation for this split - only {len(temp_val_df)} validation records available (minimum 10 required)")
            
            # Skip empty splits (need both history and target)
            if len(history_df) > 0 and len(target_df) > 0:
                # Get actual timestamps for logging
                history_min = history_df.select(pl.col("timestamp").min()).item()
                history_max = history_df.select(pl.col("timestamp").max()).item()
                target_min = target_df.select(pl.col("timestamp").min()).item() 
                target_max = target_df.select(pl.col("timestamp").max()).item()
                
                # Log the split details
                self.logger.info(f"Expanding window split {split_count+1}:")
                self.logger.info(f"  History: {history_min} to {history_max} ({len(history_df)} records)")
                self.logger.info(f"  Train: {target_min} to {target_max} ({len(target_df)} records)")
                
                # Log validation details if available
                if len(val_df) > 0:
                    val_min = val_df.select(pl.col("timestamp").min()).item()
                    val_max = val_df.select(pl.col("timestamp").max()).item()
                    self.logger.info(f"  Validation: {val_min} to {val_max} ({len(val_df)} records)")
                else:
                    self.logger.info("  No validation data available")
                
                # Yield the split data
                yield (history_df, target_df, val_history_df, val_df)
                split_count += 1
            else:
                self.logger.info(f"Skipping empty split with {len(history_df)} history records and {len(target_df)} target records")
            
            # Move back by step_days (simple Python datetime math)
            # For next iteration, end of target window becomes end of current target minus step
            current_end = target_end - timedelta(days=step_days)
            
            # Check if we've reached the maximum number of splits
            if max_splits is not None and split_count >= max_splits:
                self.logger.info(f"Reached maximum number of splits ({max_splits}), stopping")
                break
                
    def create_fixed_window_splits(
            self, df: pl.DataFrame, 
            history_days: int,  # Required for fixed window
            target_days: int = 1, 
            step_days: int = 1, 
            max_splits: Optional[int] = None,
            validation_days: Optional[int] = None
        ) -> Generator[Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame], None, None]:
        """
        Yield multiple history-target-validation pairs using fixed-size history window.
        
        Args:
            df: Complete DataFrame containing all data
            history_days: Fixed number of days to use for history
            target_days: Number of days to use for target
            step_days: Number of days to step forward for each new split
            max_splits: Maximum number of splits to generate (None = no maximum)
            validation_days: Number of days to use for validation after target period
                           (None = no validation)
        
        Yields:
            Tuples of (history_df, train_df, val_history_df, val_df)
        """
        # Ensure history_days is provided
        if history_days is None:
            raise ValueError("history_days is required for fixed window splits")
            
        # Sort by timestamp to ensure chronological order
        df = df.sort('timestamp')
        
        # Get min/max timestamps
        min_timestamp = df.select(pl.col("timestamp").min()).item()
        max_timestamp = df.select(pl.col("timestamp").max()).item()
        
        # Log the actual data range for debugging
        self.logger.info(f"Data range: {min_timestamp} to {max_timestamp}")
        
        # Adjust end point if validation is needed
        if validation_days and validation_days > 0:
            current_end = max_timestamp - timedelta(days=validation_days)
            self.logger.info(f"Adjusted start point to allow validation window: {current_end}")
        else:
            current_end = max_timestamp
            
        split_count = 0
        
        # Keep going until we've reached the beginning of data or max_splits
        while True:
            # Calculate target window boundaries
            target_end = current_end
            target_start = target_end - timedelta(days=target_days)
            
            # Calculate history window boundaries (fixed size)
            history_end = target_start
            history_start = history_end - timedelta(days=history_days)
            
            # We've gone too far back if history_start is before min_timestamp
            if history_start < min_timestamp:
                self.logger.info("Reached the beginning of the data, stopping")
                break
            
            # Create dataframes with timestamp comparisons
            history_df = df.filter(
                (pl.col('timestamp') >= history_start) & 
                (pl.col('timestamp') < history_end)
            )
            target_df = df.filter(
                (pl.col('timestamp') >= target_start) & 
                (pl.col('timestamp') < target_end)
            )
            
            # Create validation dataframe if validation is enabled
            val_df = pl.DataFrame()
            val_history_df = pl.concat([history_df, target_df], how='vertical')
            
            if validation_days and validation_days > 0:
                val_start = target_end
                val_end = val_start + timedelta(days=validation_days)
                
                temp_val_df = df.filter(
                    (pl.col('timestamp') >= val_start) & 
                    (pl.col('timestamp') < val_end)
                )
                
                if len(temp_val_df) >= 10:  # Minimum size check
                    val_df = temp_val_df
                else:
                    self.logger.info(f"Skipping validation for this split - only {len(temp_val_df)} validation records")
            
            # Skip empty splits (need both history and target)
            if len(history_df) > 0 and len(target_df) > 0:
                # Get actual timestamps for logging
                history_min = history_df.select(pl.col("timestamp").min()).item()
                history_max = history_df.select(pl.col("timestamp").max()).item()
                target_min = target_df.select(pl.col("timestamp").min()).item() 
                target_max = target_df.select(pl.col("timestamp").max()).item()
                
                # Log the split details
                self.logger.info(f"Fixed window split {split_count+1}:")
                self.logger.info(f"  History: {history_min} to {history_max} ({len(history_df)} records)")
                self.logger.info(f"  Train: {target_min} to {target_max} ({len(target_df)} records)")
                
                if len(val_df) > 0:
                    val_min = val_df.select(pl.col("timestamp").min()).item()
                    val_max = val_df.select(pl.col("timestamp").max()).item()
                    self.logger.info(f"  Validation: {val_min} to {val_max} ({len(val_df)} records)")
                
                # Yield the split data
                yield (history_df, target_df, val_history_df, val_df)
                split_count += 1
            else:
                self.logger.info(f"Skipping empty split with {len(history_df)} history records and {len(target_df)} target records")
            
            # Move back by step_days
            current_end = target_end - timedelta(days=step_days)
            
            # Check if we've reached the maximum number of splits
            if max_splits is not None and split_count >= max_splits:
                self.logger.info(f"Reached maximum number of splits ({max_splits}), stopping")
                break