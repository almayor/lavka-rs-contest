import os
import pickle
import hashlib
import polars as pl
from typing import Tuple, List, Dict, Any, Optional

from .config import Config
from .feature_factory import FeatureFactory
from .custom_logging import get_logger

class CachedFeatureGenerator:
    """Feature generator that caches computed features to avoid redundant calculations."""
    
    def __init__(self, feature_factory: FeatureFactory, cache_dir: str = "feature_cache"):
        self.feature_factory = feature_factory
        self.cache_dir = cache_dir
        self.logger = get_logger(self.__class__.__name__)
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, history_df: pl.DataFrame, target_df: pl.DataFrame, 
                      feature_names: List[str], target_name: str) -> str:
        """Generate a unique cache key based on the input parameters."""
        # Use history and target time ranges in the key
        history_range = f"{history_df['timestamp'].min()}_to_{history_df['timestamp'].max()}"
        target_range = f"{target_df['timestamp'].min()}_to_{target_df['timestamp'].max()}"
        
        # Include feature names in the key
        feature_str = "_".join(sorted(feature_names))
        
        # Create a hash for the key
        key_parts = [history_range, target_range, feature_str, target_name]
        key_str = "_".join(key_parts)
        cache_key = hashlib.md5(key_str.encode()).hexdigest()
        
        return cache_key
    
    def _get_cache_path(self, cache_key: str, suffix: str) -> str:
        """Get the file path for cached features."""
        return os.path.join(self.cache_dir, f"{cache_key}_{suffix}.pkl")
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if features are already cached."""
        features_path = self._get_cache_path(cache_key, "features")
        targets_path = self._get_cache_path(cache_key, "targets")
        meta_path = self._get_cache_path(cache_key, "meta")
        
        return (os.path.exists(features_path) and 
                os.path.exists(targets_path) and 
                os.path.exists(meta_path))
    
    def _save_to_cache(self, cache_key: str, features: pl.DataFrame, targets: pl.Series, 
                       cat_columns: List[str], request_ids: pl.Series) -> None:
        """Save generated features to cache."""
        features_path = self._get_cache_path(cache_key, "features")
        targets_path = self._get_cache_path(cache_key, "targets")
        meta_path = self._get_cache_path(cache_key, "meta")
        
        # Save features as Parquet file for efficiency
        features.write_parquet(features_path)
        
        # Save targets as pickle (might be a Series)
        with open(targets_path, 'wb') as f:
            pickle.dump(targets, f)
        
        # Save metadata (categorical columns and request IDs)
        with open(meta_path, 'wb') as f:
            pickle.dump({'cat_columns': cat_columns, 'request_ids': request_ids}, f)
    
    def _load_from_cache(self, cache_key: str) -> Tuple[pl.DataFrame, pl.Series, List[str], pl.Series]:
        """Load cached features."""
        features_path = self._get_cache_path(cache_key, "features")
        targets_path = self._get_cache_path(cache_key, "targets")
        meta_path = self._get_cache_path(cache_key, "meta")
        
        # Load features from Parquet
        features = pl.read_parquet(features_path)
        
        # Load targets from pickle
        with open(targets_path, 'rb') as f:
            targets = pickle.load(f)
        
        # Load metadata
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        return features, targets, meta['cat_columns'], meta['request_ids']
    
    def generate_batch(self, history_df: pl.DataFrame, target_df: pl.DataFrame, 
                      feature_names: Optional[List[str]] = None, 
                      target_name: Optional[str] = None) -> Tuple[pl.DataFrame, pl.Series, List[str], pl.Series]:
        """Generate features with caching."""
        if feature_names is None or target_name is None:
            # Get from feature factory's config if not provided
            config = self.feature_factory.config
            feature_names = feature_names or config.get("features")
            target_name = target_name or config.get("target")
        
        cache_key = self._get_cache_key(history_df, target_df, feature_names, target_name)
        
        if self._is_cached(cache_key):
            self.logger.info(f"Loading features from cache: {cache_key}")
            return self._load_from_cache(cache_key)
        
        self.logger.info(f"Generating features (not found in cache): {cache_key}")
        features, targets, cat_columns, request_ids = self.feature_factory.generate_batch(
            history_df, target_df, feature_names, target_name
        )
        
        self.logger.info(f"Saving features to cache: {cache_key}")
        self._save_to_cache(cache_key, features, targets, cat_columns, request_ids)
        
        return features, targets, cat_columns, request_ids
        
    def generate_features_only(self, history_df: pl.DataFrame, target_df: pl.DataFrame, 
                             feature_names: Optional[List[str]] = None) -> Tuple[pl.DataFrame, List[str], pl.Series]:
        """
        Generate only features (without targets) for test data prediction.
        This method is specifically designed for generating features for test data where 
        target values are not available.
        
        Args:
            history_df: Historical data for feature generation context
            target_df: Target dataframe with records to predict
            feature_names: List of feature names to generate
            
        Returns:
            Tuple containing:
            - Features dataframe
            - List of categorical column names
            - Request IDs for each row
        """
        if feature_names is None:
            # Get from feature factory's config if not provided
            config = self.feature_factory.config
            feature_names = feature_names or config.get("features")
        
        # Use a special cache key for test features
        cache_key = f"test_{self._get_cache_key(history_df, target_df, feature_names, 'no_target')}"
        
        if self._is_cached(cache_key):
            self.logger.info(f"Loading test features from cache: {cache_key}")
            cached_data = self._load_from_cache(cache_key)
            return cached_data[0], cached_data[2], cached_data[3]  # features, cat_columns, request_ids
        
        self.logger.info(f"Generating test features (not found in cache): {cache_key}")
        
        # Get request IDs
        request_ids = target_df['request_id']
        
        # Generate only features using the feature factory
        features, cat_columns = self.feature_factory.generate_features(
            history_df, target_df, feature_names
        )
        
        # Create a dummy target for cache consistency
        dummy_target = pl.Series(name="target", values=[None] * len(features), dtype=pl.Int64)
        
        self.logger.info(f"Saving test features to cache: {cache_key}")
        self._save_to_cache(cache_key, features, dummy_target, cat_columns, request_ids)
        
        return features, cat_columns, request_ids