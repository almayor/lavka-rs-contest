from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

from .config import Config
from .custom_logging import get_logger


class FeatureFactory:
    """Feature generation with selective feature creation"""
    
    # Class-level registry of feature generators
    _feature_registry = {}
    _possible_targets = set()
    
    @classmethod
    def register(cls, feature_name: str,
                 depends_on: str | List[str] | None = None,
                 categorical_cols: List[str] | None = None):
        """
        Decorator to register a method as a feature generator and to specify its dependencies. The method
        must accept two arguments: history_df and target_df, and return a target_df with new columns.
        Args:
            feature_name (str): Name of the feature to be generated.
            categorical_cols (List[str] | None): List of categorical columns this feature produces.
            depends_on (str | List[str] | None): List of features that this feature depends on.
        """
        depends_on = depends_on or []
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            cls._feature_registry[feature_name] = {
                'func': wrapper,
                'depends_on': depends_on,
                'categorical_cols': categorical_cols or [],
            }
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
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return result, []
        
            cls._feature_registry[feature_name] = {
                'func': wrapper,
                'depends_on': depends_on,
                'categorical_cols': [],
            }
            cls._possible_targets.add(feature_name)

            return wrapper
        
        return decorator

    def __init__(self, config: Config, feature_selector: Optional[Callable] = None):
        """Initialize feature factory"""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.feature_selector = feature_selector

    def register_feature_selector(self, feature_selector: Callable):
        """
        Register a feature selector function to be applied after feature generation.
        Args:
            feature_selector (Callable): Function to select features.
        """
        self.logger.debug("Registering feature selector")
        self.feature_selector = feature_selector
        
    def set_selected_features(self, selected_features: List[str]):
        """
        Set a list of selected features to use.
        This is an alternative to using a feature selector function.
        
        Args:
            selected_features: List of feature column names to keep
        """
        self.logger.info(f"Setting {len(selected_features)} selected features")
        self.logger.debug(f"Selected features: {selected_features}")
        
        # Create a simple feature selector function that selects only these features
        def feature_selector(df: pl.DataFrame) -> pl.DataFrame:
            # Get categorical columns
            cat_cols = []
            for feature_name, feature_info in self._feature_registry.items():
                cat_cols.extend(feature_info.get('categorical_cols', []))
            
            # Select columns to keep
            columns_to_select = list(set(cat_cols) | set(selected_features))
            
            # Filter to only include columns that exist in df
            available_columns = [col for col in columns_to_select if col in df.columns]
            
            self.logger.debug(f"Applying feature selection to keep {len(available_columns)} columns")
            return df.select(available_columns)
        
        # Register this selector
        self.feature_selector = feature_selector
    
    def generate_batch(
            self, history_df, target_df, requested_features=None, requested_target=None, show_progress=True
        ) -> Tuple[pl.DataFrame, pl.Series, List[str], pl.Series]:
        """
        Generate features and target for a batch of requests.
        Args:
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            requested_features (str | List[str] | None): Features to generate (if None, config is used).
            requested_target (str | None): Target to generate (if None, config is used).
            show_progress (bool): Whether to show a progress bar during feature generation.
        Returns:
            Tuple[pl.DataFrame, pl.Series, List[str], pl.Series]: Tuple containing:
                - Generated features (pl.DataFrame)
                - Target (pl.Series)
                - Categorical column names in the generated features (List[str])
                - Request IDs per row (pl.Series)
        """
        request_ids = target_df['request_id']
        features, cat_columns = self.generate_features(history_df, target_df, requested_features, show_progress)
        target, _ = self.generate_target(history_df, target_df, requested_target)
        mask = ~target.is_null()
        if self.feature_selector:
            features = self.feature_selector(features)
        return (
            features.filter(mask),
            target.filter(mask),
            cat_columns,
            request_ids.filter(mask),
        )
        
    def generate_features_only(
            self, history_df: pl.DataFrame, target_df: pl.DataFrame, requested_features: List[str] | None = None,
            show_progress: bool = True
        ) -> Tuple[pl.DataFrame, List[str], pl.Series]:
        """
        Generate only features without target (for prediction/inference).
        Args:
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            requested_features (List[str] | None): Features to generate (if None, config is used).
            show_progress (bool): Whether to show a progress bar during feature generation.
        Returns:
            Tuple[pl.DataFrame, List[str], pl.Series]: Tuple containing:
                - Generated features (pl.DataFrame)
                - Categorical column names in the generated features (List[str])
                - Request IDs per row (pl.Series)
        """
        request_ids = target_df['request_id']
        features, cat_columns = self.generate_features(history_df, target_df, requested_features, show_progress)
        
        if self.feature_selector:
            features = self.feature_selector(features)
            
        return features, cat_columns, request_ids

    def generate_features(
            self, history_df: pl.DataFrame, target_df: pl.DataFrame, requested_features: List[str] | None = None,
            show_progress: bool = True
        ) -> pl.DataFrame:
        """
        Generate only the requested features and their dependencies
        Args:
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            requested_features (List[str] | None): Features to generate (if None, config is used).
            show_progress (bool): Whether to show a progress bar during feature generation.
        Returns:
            Tuple[pl.DataFrame, List[str]]: Tuple containing:
                - Generated features (pl.DataFrame)
                - Categorical column names in the generated features (List[str])
        """
        if requested_features is None:
            requested_features = self.config.get("features")
        if len(requested_features) != len(set(requested_features)):
            self.logger.error("Duplicate feature names in requested_features")
            raise ValueError("Duplicate feature names in requested_features")
        self.logger.info(f"Generating features: {', '.join(requested_features)}")
                
        # Generate each requested feature (and dependencies)
        all_columns, all_cat_columns = set(), set()
        
        # Create progress bar if requested
        feature_iterator = tqdm(requested_features, desc="Generating features") if show_progress else requested_features
        
        for feature_name in feature_iterator:
            if show_progress:
                feature_iterator.set_description(f"Generating feature: {feature_name}")
                
            target_df, columns = self._generate_feature(
                feature_name, history_df, target_df
            )
            all_columns.update(columns)
            cat_columns = self.__class__._feature_registry[feature_name]['categorical_cols']
            all_columns.update(cat_columns) # preserve categorical features even if they already existed
            all_cat_columns.update(cat_columns)
        
        all_cat_columns = list(all_cat_columns) if len(all_cat_columns) else None
        self.logger.info("Joined features")
        self.logger.info(f"All column names: {all_columns}")
        self.logger.info(f"All categorical column names: {all_cat_columns}")
        
        target_df = target_df.select(all_columns)
        if self.feature_selector:
            target_df = self.feature_selector(target_df)
        
        return target_df, all_cat_columns
    
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

        feature, _ = self._generate_feature(requested_target, history_df, target_df)
        return feature
    
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
            Tuple[pl.DataFrame, List[str], List[str]]: Tuple containing:
                - Generated features (pl.DataFrame)
                - Newly added column names
        """
        # Return from cache if already generated
        if generated_features and feature_name in generated_features:
            return target_df
        
        # Check if feature exists
        if feature_name not in self.__class__._feature_registry:
            self.logger.error(f"Feature '{feature_name}' is not registered")
            self.logger.error(f"Available features: {list(self.__class__._feature_registry.keys())}")
            raise ValueError(f"Feature '{feature_name}' is not registered")
        
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
        features = generator_func(history_df, target_df)
    
        if isinstance(features, pl.DataFrame):
            new_columns = list(set(features.columns).difference(old_columns))
        else:
            new_columns = [feature_name]
        self.logger.debug(f"New column names: {new_columns}")
        generated_features.add(feature_name)
        
        # Cache and return
        return features, new_columns
