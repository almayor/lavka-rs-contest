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
    _category_registry = {}
    
    @classmethod
    def register(cls, feature_name: str, join_on: str | list[str],
                 depends_on: str | List[str] | None = None, category: str | None = None):
        """Decorator to register a method as a feature generator"""
        depends_on = depends_on or []
        
        def decorator(func):
            cls._feature_registry[feature_name] = {
                'func': func,
                'join_on': join_on,
                'depends_on': depends_on,
            }
            if category is not None:
                cls._category_registry.setdefault(category, []).append(feature_name)
            
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                return func(self, *args, **kwargs)
            
            return wrapper
        
        return decorator

    def __init__(self, config: Config):
        """Initialize feature factory"""
        self.config = None
        self.logger = get_logger(self.__class__.__name__)
    
    def generate_features(self, history_df, target_df, requested_features):
        """Generate only the requested features and their dependencies"""
        self.logger.info(f"Generating features: {', '.join(requested_features)}")
        
        # Expand feature groups if needed
        expanded_features = self._expand_feature_groups(requested_features)
        
        # Generate each requested feature (and dependencies)
        for feature_name in expanded_features:
            self._generate_feature(feature_name, history_df, target_df)
        
        # Return only the requested features
        feature_data = [self.features[f] for f in expanded_features if f in self.features]
        result_df = target_df
        for df, feature_name in zip(feature_data, requested_features):
            join_on = self.__class__._feature_registry[feature_name]["join_on"]
            try:
                result_df = result_df.join(
                    df, on=join_on, how='left'
                )
            except Exception as e:
                self.logger.error(f"Error joining feature {feature_name}: {e}")
                raise e
        
        self.logger.debug("Joined features")
        return result_df
    
    def _expand_feature_groups(self, requested_features):
        """Expand feature group names into individual features"""
        if not self.config:
            return requested_features
            
        expanded = []
        
        for feature in requested_features:
            if feature in self.__class__._category_registry:
                # This is a feature group
                expanded.extend(self.__class__._category_registry[feature])
            else:
                # This is an individual feature
                expanded.append(feature)
                
        return list(set(expanded))  # Remove duplicates
    
    def _generate_feature(self, feature_name, history_df, target_df):
        """Generate a single feature, handling dependencies"""
        # Return from cache if already generated
        if feature_name in self.features:
            return self.features[feature_name]
        
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
        for dep in dependencies:
            self._generate_feature(dep, history_df, target_df)
        
        # Generate this feature
        self.logger.debug(f"Generating feature: {feature_name}")
        feature_df = generator_func(self, history_df, target_df)
        
        # Cache and return
        self.features[feature_name] = feature_df
        return feature_df
    