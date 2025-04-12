import polars as pl
from typing import Generator

from .config import Config

class HistoryGenerator:
    """Handles history generation based on temporal constraints"""
    
    def __init__(self, config: Config):
        self.config = config
        self.method = config.get('history_generation.method')

    def get_history(
            self, full_data: pl.DataFrame, target_data: pl.DataFrame
        ) -> Generator[tuple[pl.DataFrame, pl.DataFrame]]:
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
        yield (history_data, target_data)
    
    def _get_sliding_window_history(self, full_data, target_data):
        """
        Sliding window history uses a configurable time window to generate features.
        Args:
            full_data (pl.DataFrame): The full dataset
            target_data (pl.DataFrame): The target dataset to generate features for
        Returns:
            generator: A generator yielding tuples of historical data and target data
        """
        #TODO
        raise NotImplementedError