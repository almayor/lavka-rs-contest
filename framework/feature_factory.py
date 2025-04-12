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
        """Decorator to register a method as a feature generator"""
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
        """Decorator to register a method as a feature generator"""
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
                return func(self, *args, **kwargs)
            
            return wrapper
        
        return decorator

    def __init__(self, config: Config):
        """Initialize feature factory"""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    def generate_batch(self, history_df, target_df, requested_features=None, requested_target=None):
        """Generate both features and target"""
        features = self.generate_features(history_df, target_df, requested_features)
        target = self.generate_target(history_df, target_df, requested_target)
        mask = ~target.is_null()
        return features.filter(mask), target.filter(mask)

    def generate_features(self, history_df, target_df, requested_features=None):
        """Generate only the requested features and their dependencies"""
        if requested_features is None:
            requested_features = self.config.get("features")
        self.logger.info(f"Generating features: {', '.join(requested_features)}")
                
        # Generate each requested feature (and dependencies)
        for feature_name in requested_features:
            target_df = self._generate_feature(feature_name, history_df, target_df)
        
        self.logger.debug("Joined features")
        return target_df.select(requested_features)
    
    def generate_target(self, history_df, target_df, requested_target: str | None = None):
        """Generate the target feature"""
        if requested_target is None:
            requested_target = self.config.get("target")
        if requested_target not in self.__class__._possible_targets:
            raise ValueError(f"Unknown target {requested_target}")

        return self._generate_feature(requested_target, history_df, target_df)
    
    def _generate_feature(self, feature_name, history_df, target_df, generated_features=None):
        """Generate a single feature, handling dependencies"""
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
        target_df = generator_func(history_df, target_df)
        generated_features.add(feature_name)
        
        # Cache and return
        return target_df
    