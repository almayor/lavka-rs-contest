import json
import logging
import os
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.metrics import log_loss, ndcg_score, roc_auc_score
from tqdm.auto import tqdm

from .default_config import DEFAULT_CONFIG

class Config:
    """Configuration management for experiments"""
    
    def __init__(self, config_dict=None):
        """Initialize with default or provided configuration"""
        self.config = config_dict or {}
        
        # Set defaults if not provided
        self._set_defaults()
        
    def _set_defaults(self):
        """Set default configuration values"""
        # Update config with defaults for missing values
        for section, values in DEFAULT_CONFIG.items():
            if section not in self.config:
                self.config[section] = values
            else:
                for key, value in values.items():
                    if key not in self.config[section]:
                        self.config[section][key] = value
    
    def get(self, section, key=None):
        """Get a configuration value"""
        if key is None:
            return self.config.get(section, {})
        return self.config.get(section, {}).get(key)
    
    def set(self, section, key, value):
        """Set a configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def save(self, filename='experiment_config.json'):
        """Save configuration to file"""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    @classmethod
    def load(cls, filename='experiment_config.json'):
        """Load configuration from file"""
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def __str__(self):
        """String representation of config"""
        return json.dumps(self.config, indent=4)
