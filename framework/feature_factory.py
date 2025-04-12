from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

from .custom_logging import get_logger
from .config import Config

class FeatureFactory:
    """Feature generation with selective feature creation"""
    
    # Class-level registry of feature generators
    _feature_registry = {}
    _possible_targets = set()
    
    @classmethod
    def register(cls, feature_name: str,
                 depends_on: str | List[str] | None = None):
        """
        Decorator to register a method as a feature generator and to specify its dependencies. The method
        must accept two arguments: history_df and target_df, and return a tuple of (target_df, categorical_col_names).
        Args:
            feature_name (str): Name of the feature to be generated.
            depends_on (str | List[str] | None): List of features that this feature depends on.
        """
        depends_on = depends_on or []
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        
        def decorator(func):
            cls._feature_registry[feature_name] = {
                'func': func,
                'depends_on': depends_on,
            }
            
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                return func(self, *args, **kwargs)
            
            return wrapper
        
        return decorator

    @classmethod
    def register_target(cls, feature_name: str,
                 depends_on: str | List[str] | None = None):
        """
        Decorator to register a method as a target generator and to specify its dependencies. The method
        must accept two arguments: history_df and target_df, and return a pl.Series of the target. Rows where
        the target is null will be filtered out.
        Args:
            feature_name (str): Name of the target to be generated.
            depends_on (str | List[str] | None): List of features that this target depends on.
        """
        depends_on = depends_on or []
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        
        def decorator(func):
            cls._feature_registry[feature_name] = {
                'func': func,
                'depends_on': depends_on,
            }
            cls._possible_targets.add(feature_name)

            
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                # Return an empty list for categorical columns
                # for compatibility with the feature generation
                return func(self, *args, **kwargs), []
            
            return wrapper
        
        return decorator

    def __init__(self, config: Config):
        """Initialize feature factory"""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    def generate_batch(
            self, history_df, target_df, requested_features=None, requested_target=None
        ) -> Tuple[pl.DataFrame, pl.Series, List[str], pl.Series]:
        """
        Generate features and target for a batch of requests.
        Args:
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            requested_features (str | List[str] | None): Features to generate (if None, config is used).
            requested_target (str | None): Target to generate (if None, config is used).
        Returns:
            Tuple[pl.DataFrame, pl.Series, List[str], pl.Series]: Tuple containing:
                - Generated features (pl.DataFrame)
                - Target (pl.Series)
                - Categorical column names in the generated features (List[str])
                - Request IDs per row (pl.Series)
        """
        request_ids = target_df['request_id']
        features, cat_col_names = self.generate_features(history_df, target_df, requested_features)
        target = self.generate_target(history_df, target_df, requested_target)
        mask = ~target.is_null()
        return (
            features.filter(mask),
            target.filter(mask),
            cat_col_names,
            request_ids.filter(mask),
        )

    def generate_features(
            self, history_df: pl.DataFrame, target_df: pl.DataFrame, requested_features: List[str] | None = None
        ) -> Tuple[pl.DataFrame, List[str]]:
        """
        Generate only the requested features and their dependencies
        Args:
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            requested_features (List[str] | None): Features to generate (if None, config is used).
        Returns:
            Tuple[pl.DataFrame, List[str]]: Tuple containing:
                - Generated features (pl.DataFrame)
                - Categorical column names in the generated features (List[str])
        """
        if requested_features is None:
            requested_features = self.config.get("features")
        self.logger.info(f"Generating features: {', '.join(requested_features)}")
                
        # Generate each requested feature (and dependencies)
        all_col_names, all_cat_col_names = set(), set()
        for feature_name in requested_features:
            target_df, col_names, cat_col_names = self._generate_feature(
                feature_name, history_df, target_df
            )
            all_col_names.update(col_names)
            all_cat_col_names.update(cat_col_names)
        
        self.logger.info("Joined features")
        self.logger.info(f"All column names: {all_col_names}")
        self.logger.info(f"All categorical column names: {all_cat_col_names}")
        return target_df.select(all_col_names), all_cat_col_names
    
    def generate_target(self, history_df, target_df, requested_target: str | None = None) -> pl.Series:
        """
        Generate the target variable.
        Args:
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            requested_target (str | None): Target to generate (if None, config is used).
        Returns:
            pl.Series: Generated target (pl.Series)
        """
        if requested_target is None:
            requested_target = self.config.get("target")
        if requested_target not in self.__class__._possible_targets:
            raise ValueError(f"Unknown target {requested_target}")

        return self._generate_feature(requested_target, history_df, target_df)
    
    def _generate_feature(
            self, feature_name: str, history_df: pl.DataFrame, target_df: pl.DataFrame,
            generated_features=None
        ) -> Tuple[pl.DataFrame, List[str]]:
        """
        Generate a single feature, handling dependencies
        Args:
            feature_name (str): Name of the feature to be generated.
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            generated_features (set | None): Set of already generated features (for caching).
        Returns:
            Tuple[pl.DataFrame, List[str]]: Tuple containing:
                - Generated features (pl.DataFrame)
                - Categorical column names in the generated features (List[str])
        """
        # Return from cache if already generated
        if generated_features and feature_name in generated_features:
            return target_df
        
        # Check if feature exists
        if feature_name not in self.__class__._feature_registry:
            msg = f"Feature '{feature_name}' is not registered"
            self.logger.error(msg)
            raise ValueError(msg)
        
        # Get feature info
        feature_info = self.__class__._feature_registry[feature_name]
        generator_func = feature_info['func']
        dependencies = feature_info['depends_on']
        
        # Generate dependencies first
        generated_features = generated_features or set()
        for dep in dependencies:
            self._generate_feature(dep, history_df, target_df, generated_features)
        
        # Generate this feature
        self.logger.debug(f"Generating feature: {feature_name}")
        old_columns = target_df.columns
        features, cat_col_names = generator_func(history_df, target_df)
    
        if isinstance(features, pl.DataFrame):
            new_col_names = list(set(target_df.columns).difference(old_columns))
        else:
            new_col_names = [feature_name]
        self.logger.debug(f"New column names: {new_col_names}")
        self.logger.debug(f"New categorical columns: {cat_col_names}")
        generated_features.add(feature_name)
        
        # Cache and return
        return target_df, new_col_names, cat_col_names
    