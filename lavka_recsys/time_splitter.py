import os
import polars as pl
from typing import Tuple, List, Generator, Optional
import hashlib
import pickle
from datetime import timedelta

from .config import Config
from .custom_logging import get_logger

class TimeSplitter:
    """
    Splits time-series data into multiple history-target pairs using a generator approach with all
    available history up to the target date.
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration"""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    def create_sliding_window_splits(
            self, df: pl.DataFrame, 
            history_days: Optional[int] = None,  # Not used but kept for API consistency
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
            history_days: Not used, all available history is used up to target start
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
            val_history_df = history_df.clone()
            
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
                    # History for validation includes all data up to validation start
                    # (i.e., history + target data)
                    val_history_df = df.filter(pl.col('timestamp') < val_start)
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
                self.logger.info(f"Split {split_count+1}:")
                self.logger.info(f"  History: {history_min} to {history_max} ({len(history_df)} records)")
                self.logger.info(f"  Train: {target_min} to {target_max} ({len(target_df)} records)")
                
                # Log validation details if available
                if len(val_df) > 0:
                    val_min = val_df.select(pl.col("timestamp").min()).item()
                    val_max = val_df.select(pl.col("timestamp").max()).item()
                    self.logger.info(f"  Validation: {val_min} to {val_max} ({len(val_df)} records)")
                    self.logger.info(f"  Validation history: up to {val_history_df.select(pl.col('timestamp').max()).item()} ({len(val_history_df)} records)")
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