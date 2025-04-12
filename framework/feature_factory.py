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
    _target_registry = {}
    
    @classmethod
    def register(cls, feature_name: str, join_on: str | list[str],
                 depends_on: str | List[str] | None = None):
        """Decorator to register a method as a feature generator"""
        depends_on = depends_on or []
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        
        def decorator(func):
            cls._feature_registry[feature_name] = {
                'func': func,
                'join_on': join_on,
                'depends_on': depends_on,
            }
            
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                return func(self, *args, **kwargs)
            
            return wrapper
        
        return decorator

    @classmethod
    def register_target(cls, target_name: str, depends_on: str | list[str] | None = None):
        """Decorator to register a method as a target generator"""
        depends_on = depends_on or []
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        
        def decorator(func):
            cls._target_registry[target_name] = {
                'func': func,
                'depends_on': depends_on,
            }
            
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                return func(self, *args, **kwargs)
            
            return wrapper
        
        return decorator

    def __init__(self, config: Config):
        """Initialize feature factory"""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    def generate_features(self, history_df, target_df, requested_features=None):
        """Generate only the requested features and their dependencies"""
        if requested_features is None:
            requested_features = self.config.get("features")
        self.logger.info(f"Generating features: {', '.join(requested_features)}")

        self.features = {}
                
        # Generate each requested feature (and dependencies)
        for feature_name in requested_features:
            self._generate_feature(feature_name, history_df, target_df)
        
        # Return only the requested features
        feature_data = [self.features[f] for f in requested_features if f in self.features]
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
    
    def generate_target(self, history_df, target_df, requested_target: str | None = None):
        """Generate the target feature"""
        if requested_target is None:
            requested_target = self.config.get("target")
        if requested_target not in self.__class__._target_registry:
            raise ValueError(f"Unknown target {requested_target}")

        self.features = {}
        self._generate_feature(requested_target, history_df, target_df)
        return self.features[requested_target]
    
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
        feature_df = generator_func(history_df, target_df)
        
        # Cache and return
        self.features[feature_name] = feature_df
        return feature_df
    