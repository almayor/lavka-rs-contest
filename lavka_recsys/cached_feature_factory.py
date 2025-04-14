import os
import hashlib
import pickle
import time
import json
import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set

from .custom_logging import get_logger
from .feature_factory import FeatureFactory

class CachedFeatureFactory:
    """
    Feature factory with built-in caching capabilities.
    Enhances FeatureFactory by providing transparent feature caching.
    """
    
    def __init__(self, feature_factory: FeatureFactory = None, config=None):
        """
        Initialize with an existing feature factory or create a new one
        
        Args:
            feature_factory: An existing FeatureFactory instance (optional)
            config: Configuration for creating a new FeatureFactory if one isn't provided
        """
        # Create logger
        self.logger = get_logger(self.__class__.__name__)
        
        # Set up feature factory
        if feature_factory is not None:
            self.feature_factory = feature_factory
            self.config = feature_factory.config
        elif config is not None:
            from .feature_factory import FeatureFactory
            self.feature_factory = FeatureFactory(config)
            self.config = config
        else:
            raise ValueError("Either feature_factory or config must be provided")
        
        # Set up caching configuration
        self.cache_enabled = self.config.get('feature_caching.enabled', True)
        self.cache_dir = self.config.get('output.feature_cache_dir', 'feature_cache')
        
        # Create cache directory if it doesn't exist
        if self.cache_enabled:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                self.logger.info(f"Using feature cache directory: {self.cache_dir}")
            except Exception as e:
                self.logger.error(f"Failed to create feature cache directory {self.cache_dir}: {str(e)}")
                # Fall back to a default directory
                self.cache_dir = 'feature_cache'
                os.makedirs(self.cache_dir, exist_ok=True)
                self.logger.info(f"Using fallback feature cache directory: {self.cache_dir}")
    
    def _generate_cache_key(self, history_df: pl.DataFrame, target_df: pl.DataFrame, 
                          feature_names: List[str], target_name: Optional[str] = None) -> str:
        """
        Generate a cache key based on the input data and requested features.
        
        Args:
            history_df: Historical data for feature generation
            target_df: Target data to generate features for
            feature_names: List of requested features
            target_name: Optional target column name
            
        Returns:
            str: Cache key string
        """
        # Create a cache key from important attributes of the input data
        cache_items = []
        
        # Add feature names to the key (sorted for consistency)
        cache_items.append(f"features={','.join(sorted(feature_names))}")
        
        # Add target name if provided
        if target_name:
            cache_items.append(f"target={target_name}")
        
        # Add data characteristics that uniquely identify this dataset
        # For history_df, use time range and row count
        if len(history_df) > 0:
            min_time = history_df['timestamp'].min().strftime("%Y%m%d%H%M%S")
            max_time = history_df['timestamp'].max().strftime("%Y%m%d%H%M%S")
            cache_items.append(f"history_range={min_time}-{max_time}")
            cache_items.append(f"history_rows={len(history_df)}")
        else:
            cache_items.append("history_empty=True")
            
        # For target_df, use time range and row count plus a sample of request_ids
        # This helps distinguish between different target datasets with same time range
        if len(target_df) > 0:
            min_time = target_df['timestamp'].min().strftime("%Y%m%d%H%M%S")
            max_time = target_df['timestamp'].max().strftime("%Y%m%d%H%M%S")
            cache_items.append(f"target_range={min_time}-{max_time}")
            cache_items.append(f"target_rows={len(target_df)}")
            
            # Sample a few request_ids as part of the fingerprint
            if 'request_id' in target_df.columns:
                request_samples = target_df.select('request_id').sample(n=min(20, len(target_df)), seed=42)
                request_hash = hashlib.md5(str(request_samples).encode()).hexdigest()[:8]
                cache_items.append(f"request_hash={request_hash}")
        else:
            cache_items.append("target_empty=True")
        
        # Join all items to create a key string
        key_string = "|".join(cache_items)
        
        # Create a hash as the actual key
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get the file path for a given cache key.
        
        Args:
            cache_key: Cache key string
            
        Returns:
            str: Path to the cache file
        """
        return os.path.join(self.cache_dir, f"features_{cache_key}.pkl")
    
    def _save_to_cache(self, cache_key: str, data: Tuple) -> bool:
        """
        Save feature data to cache.
        
        Args:
            cache_key: Cache key string
            data: Tuple of (features, target, cat_columns, request_ids)
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        if not self.cache_enabled:
            return False
            
        cache_path = self._get_cache_path(cache_key)
        
        try:
            start_time = time.time()
            # Create a temporary file first to avoid corruption
            temp_path = f"{cache_path}.tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)
                
            # Rename to the final path
            if os.path.exists(cache_path):
                os.remove(cache_path)
            os.rename(temp_path, cache_path)
            
            save_time = time.time() - start_time
            self.logger.info(f"Features saved to cache in {save_time:.2f}s: {cache_path}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to save features to cache: {str(e)}")
            
            # Clean up temporary file if it exists
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
                
            return False
    
    def _load_from_cache(self, cache_key: str) -> Optional[Tuple]:
        """
        Load feature data from cache.
        
        Args:
            cache_key: Cache key string
            
        Returns:
            Optional[Tuple]: Tuple of (features, target, cat_columns, request_ids) if found, None otherwise
        """
        if not self.cache_enabled:
            return None
            
        cache_path = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_path):
            try:
                start_time = time.time()
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                load_time = time.time() - start_time
                self.logger.info(f"Features loaded from cache in {load_time:.2f}s: {cache_path}")
                return data
            except Exception as e:
                self.logger.warning(f"Failed to load features from cache: {str(e)}")
                
                # Remove corrupted cache file
                try:
                    os.remove(cache_path)
                    self.logger.info(f"Removed corrupted cache file: {cache_path}")
                except:
                    pass
        
        return None
    
    def generate_batch(self, history_df: pl.DataFrame, target_df: pl.DataFrame, 
                      feature_names: List[str] = None, target_name: Optional[str] = None) -> Tuple:
        """
        Generate features for a batch of data with automatic caching.
        
        Args:
            history_df: Historical data for feature generation
            target_df: Target data to generate features for
            feature_names: List of features to generate (defaults to all in config)
            target_name: Optional target column name
            
        Returns:
            Tuple: (features, target, cat_columns, request_ids)
        """
        # Use all features from config if none specified
        if feature_names is None:
            feature_names = self.config.get('features', [])
            
        # Generate cache key for this request
        cache_key = self._generate_cache_key(history_df, target_df, feature_names, target_name)
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # If not in cache, generate features using the feature factory
        self.logger.info(f"Generating features for {len(target_df)} records with {len(feature_names)} feature sets")
        start_time = time.time()
        
        # Call the feature factory to generate features
        features, target, cat_columns, request_ids = self.feature_factory.generate_batch(
            history_df, target_df, feature_names, target_name
        )
        
        # Log performance
        generation_time = time.time() - start_time
        self.logger.info(f"Feature generation completed in {generation_time:.2f}s")
        
        # Save to cache for future use
        self._save_to_cache(cache_key, (features, target, cat_columns, request_ids))
        
        return features, target, cat_columns, request_ids
    
    def generate_features_only(self, history_df: pl.DataFrame, target_df: pl.DataFrame, 
                             feature_names: List[str] = None) -> Tuple:
        """
        Generate only features (no target) for a batch of data with automatic caching.
        
        Args:
            history_df: Historical data for feature generation
            target_df: Target data to generate features for
            feature_names: List of features to generate (defaults to all in config)
            
        Returns:
            Tuple: (features, cat_columns, request_ids)
        """
        # Use all features from config if none specified
        if feature_names is None:
            feature_names = self.config.get('features', [])
            
        # Generate cache key for this request
        cache_key = self._generate_cache_key(history_df, target_df, feature_names)
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            # Extract just the features, cat_columns, and request_ids (no target)
            features, _, cat_columns, request_ids = cached_data
            return features, cat_columns, request_ids
        
        # If not in cache, generate features using the feature factory
        self.logger.info(f"Generating features for {len(target_df)} records with {len(feature_names)} feature sets")
        start_time = time.time()
        
        # Call the feature factory to generate features
        features, cat_columns, request_ids = self.feature_factory.generate_features_only(
            history_df, target_df, feature_names
        )
        
        # Log performance
        generation_time = time.time() - start_time
        self.logger.info(f"Feature generation completed in {generation_time:.2f}s")
        
        # Save to cache for future use (with None as target)
        self._save_to_cache(cache_key, (features, None, cat_columns, request_ids))
        
        return features, cat_columns, request_ids
    
    def register_feature_selector(self, feature_selector):
        """
        Register a feature selector with the underlying feature factory.
        
        Args:
            feature_selector: Feature selector to register
        """
        self.feature_factory.register_feature_selector(feature_selector)
        self.logger.info("Feature selector registered with the feature factory")
        
    # Provide direct access to the underlying feature factory's methods
    def __getattr__(self, name):
        """Pass through any other attributes to the underlying feature factory"""
        return getattr(self.feature_factory, name)