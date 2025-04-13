import polars as pl
from typing import Generator, Tuple
from datetime import datetime

from .config import Config
from .custom_logging import get_logger

class HistoryGenerator:
    """Handles history generation based on temporal constraints"""
    
    def __init__(self, config: Config):
        """Initialize with configuration"""
        self.config = config
        self.method = config.get('history_generation.method')
        self.logger = get_logger(self.__class__.__name__)

    def get_history(
            self, full_data: pl.DataFrame, target_data: pl.DataFrame
        ) -> Generator[tuple[pl.DataFrame, pl.DataFrame], None, None]:
        """
        Get appropriate historical data for feature generation
        based on the configured method
        
        Args:
            full_data (pl.DataFrame): The full dataset
            target_data (pl.DataFrame): The target dataset to generate features for
        Returns:
            generator: A generator yielding tuples of historical data and target data
        """
        if self.method == 'basic':
            return self._get_basic_history(full_data, target_data)
        elif self.method == 'sliding_window':
            return self._get_sliding_window_history(full_data, target_data)
        elif self.method == 'all_history':
            return self._get_all_history(full_data, target_data)
        elif self.method == 'time_window':
            return self._get_time_window_history(full_data, target_data)
        else:
            raise ValueError(f"Unknown feature generation method: {self.method}")
    
    def _get_basic_history(self, full_data, target_data):
        """
        Basic history uses all data up to the first timestamp in the target data.
        Args:
            full_data (pl.DataFrame): The full dataset
            target_data (pl.DataFrame): The target dataset to generate features for
        Returns:
            generator: A generator yielding a single tuple of historical data and target data
        """
        full_data = full_data.sort('timestamp')
        target_min_time = target_data['timestamp'].min()
        history_data = full_data.filter(pl.col('timestamp') < target_min_time)
        
        self.logger.info(
            f"Basic history: using data before {datetime.fromtimestamp(target_min_time)} "
            f"({len(history_data)} rows)"
        )
        
        yield (history_data, target_data)
    
    def _get_all_history(self, full_data, target_data):
        """
        Use all available historical data for each target instance.
        This is ideal for final prediction, using all data up to each prediction point.
        
        Args:
            full_data (pl.DataFrame): The full dataset
            target_data (pl.DataFrame): The target dataset to generate features for
        Returns:
            generator: A generator yielding a single tuple of historical data and target data
        """
        full_data = full_data.sort('timestamp')
        
        # For final prediction, we use all data that comes before each target instance
        # We need to iterate through target_data and create history for each row
        
        # Group target data by timestamp to process together
        target_by_time = target_data.group_by('timestamp').agg([
            pl.col('*').first().alias('unique_timestamps')
        ])
        
        timestamps = target_by_time['timestamp'].sort().to_list()
        
        self.logger.info(f"Processing {len(timestamps)} unique target timestamps")
        
        for timestamp in timestamps:
            # Get target rows for this timestamp
            current_targets = target_data.filter(pl.col('timestamp') == timestamp)
            
            # Get history for this timestamp (all data before this timestamp)
            history_data = full_data.filter(pl.col('timestamp') < timestamp)
            
            self.logger.debug(
                f"Yielding history up to {datetime.fromtimestamp(timestamp)} "
                f"({len(history_data)} rows) for {len(current_targets)} target rows"
            )
            
            yield (history_data, current_targets)
    
    def _get_sliding_window_history(self, full_data, target_data):
        """
        Sliding window history uses a configurable time window to generate features.
        This gives a more realistic evaluation during training.
        
        Args:
            full_data (pl.DataFrame): The full dataset
            target_data (pl.DataFrame): The target dataset to generate features for
        Returns:
            generator: A generator yielding tuples of historical data and target data
        """
        full_data = full_data.sort('timestamp')
        
        # Get window size in days (default 30 days)
        window_days = self.config.get('history_generation.window_days', 30)
        window_seconds = window_days * 24 * 60 * 60
        
        # Group target data by timestamp to process together
        target_by_time = target_data.group_by('timestamp').agg([
            pl.col('*').first().alias('unique_timestamps')
        ])
        
        timestamps = target_by_time['timestamp'].sort().to_list()
        
        self.logger.info(
            f"Sliding window history: using {window_days}-day window "
            f"for {len(timestamps)} unique target timestamps"
        )
        
        for timestamp in timestamps:
            # Get target rows for this timestamp
            current_targets = target_data.filter(pl.col('timestamp') == timestamp)
            
            # Calculate window start time
            window_start = timestamp - window_seconds
            
            # Get history for this window
            history_data = full_data.filter(
                (pl.col('timestamp') >= window_start) & 
                (pl.col('timestamp') < timestamp)
            )
            
            self.logger.debug(
                f"Yielding {window_days}-day history window "
                f"{datetime.fromtimestamp(window_start)} - {datetime.fromtimestamp(timestamp)} "
                f"({len(history_data)} rows) for {len(current_targets)} target rows"
            )
            
            yield (history_data, current_targets)
    
    def _get_time_window_history(self, full_data, target_data):
        """
        Time window history uses a configurable time window but includes all history
        before that window as well. This is useful for getting both long-term and
        short-term patterns.
        
        Args:
            full_data (pl.DataFrame): The full dataset
            target_data (pl.DataFrame): The target dataset to generate features for
        Returns:
            generator: A generator yielding tuples of historical data and target data
        """
        full_data = full_data.sort('timestamp')
        
        # Get window parameters
        recent_window_days = self.config.get('history_generation.recent_window_days', 7)
        mid_window_days = self.config.get('history_generation.mid_window_days', 30)
        long_window_days = self.config.get('history_generation.long_window_days', 90)
        
        # Convert to seconds
        recent_window_seconds = recent_window_days * 24 * 60 * 60
        mid_window_seconds = mid_window_days * 24 * 60 * 60
        long_window_seconds = long_window_days * 24 * 60 * 60
        
        # Group target data by timestamp to process together
        target_by_time = target_data.group_by('timestamp').agg([
            pl.col('*').first().alias('unique_timestamps')
        ])
        
        timestamps = target_by_time['timestamp'].sort().to_list()
        
        self.logger.info(
            f"Time window history with {recent_window_days}-day recent window, "
            f"{mid_window_days}-day mid window, {long_window_days}-day long window "
            f"for {len(timestamps)} unique target timestamps"
        )
        
        for timestamp in timestamps:
            # Get target rows for this timestamp
            current_targets = target_data.filter(pl.col('timestamp') == timestamp)
            
            # Calculate window start times
            recent_start = timestamp - recent_window_seconds
            mid_start = timestamp - mid_window_seconds
            long_start = timestamp - long_window_seconds
            
            # Add column to identify which window each row belongs to
            # This allows creating window-specific features later
            history_data = full_data.filter(pl.col('timestamp') < timestamp).with_columns([
                pl.when(pl.col('timestamp') >= recent_start)
                  .then(pl.lit('recent'))
                  .when(pl.col('timestamp') >= mid_start)
                  .then(pl.lit('mid'))
                  .when(pl.col('timestamp') >= long_start)
                  .then(pl.lit('long'))
                  .otherwise(pl.lit('older'))
                  .alias('history_window')
            ])
            
            self.logger.debug(
                f"Yielding multi-window history up to {datetime.fromtimestamp(timestamp)} "
                f"({len(history_data)} rows) for {len(current_targets)} target rows"
            )
            
            yield (history_data, current_targets)