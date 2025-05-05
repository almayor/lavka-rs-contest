from tqdm.auto import tqdm
from typing import Iterable

import numpy as np
import pandas as pd
import pickle
import polars as pl

from ..utils.custom_logging import get_logger
from ..utils.config import Config

from .base import Model
from .catboost import CatBoostClassifierModel, CatBoostRankerModel
from .baseline import RandomModel, SingleFeatureModel

class ModelFactory:
    """Factory for creating and managing models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self._registry = {
            'catboost_classifier': CatBoostClassifierModel,
            'catboost_ranker': CatBoostRankerModel,
            'random_baseline': RandomModel,
            'single_feature': SingleFeatureModel,
            # Add more models as needed
        }
    
    def create_model(self, override_params=None) -> Model:
        """Create a model instance based on type"""
        mtype = self.config.get('model.type')
        if mtype not in self._registry:
            raise ValueError(f"Unknown model type: {mtype}")
            
        # Get model parameters from config
        mparams = self.config.get(('model', 'config', mtype), {}).copy()
        
        use_gpu = self.config.get(('model', 'use_gpu'), False)
        mparams['task_type'] = 'GPU' if use_gpu else 'CPU'
        if use_gpu:
            mparams['devices'] = self.config.get(('model', 'gpu_devices'), '0')
        mparams['thread_count'] = self.config.get(('model', 'thread_count'), -1)
        
        if override_params:
            mparams.update(override_params)
        self.logger.info(f"Creating {mtype} model with params: {mparams}")
        return self._registry[mtype](**mparams)

