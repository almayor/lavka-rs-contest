import polars as pl

class HistoryGenerator:
    """Handles history generation based on temporal constraints"""
    
    def __init__(self, config):
        self.config = config
        self.method = config.get('history_generation', 'method')

    def get_history(self, full_data, target_data):
        """
        Get appropriate historical data for feature generation
        based on the configured method
        
        Parameters:
        -----------
        full_data: Full dataset
        target_data: Target dataset for which to generate features
        
        Returns:
        --------
        Tuple of (history_df, target_df)
        """
        if self.method == 'basic':
            return self._get_basic_history(full_data, target_data)
        elif self.method == 'sliding_window':
            return self._get_sliding_window_history(full_data, target_data)
        else:
            raise ValueError(f"Unknown feature generation method: {self.method}")
    
    def _get_basic_history(self, full_data, target_data):
        """
        Basic history uses all data up to the first timestamp in the target data
        """
        full_data = full_data.sort('timestamp')
        target_min_time = target_data['timestamp'].min()
        history_data = full_data.filter(pl.col('timestamp') < target_min_time)
        yield (history_data, target_data)
    
    def _get_sliding_window_history(self, full_data, target_data):
        #TODO
        raise NotImplementedError