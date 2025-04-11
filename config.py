class Config:
    """Configuration management for experiments"""
    
    def __init__(self, config_dict=None):
        """Initialize with default or provided configuration"""
        self.config = config_dict or {}
        
        # Set defaults if not provided
        self._set_defaults()
        
    def _set_defaults(self):
        """Set default configuration values"""
        defaults = {
            'data': {
                'train_path': 'train.parquet',
                'test_path': 'test.parquet',
                'sample_size': None,  # None = use all data
                'random_seed': 42
            },
            'preprocessing': {
                'remove_duplicates': True,
                'fill_nulls': True,
                'normalize_timestamps': True,
                'clean_text': False
            },
            'features': {
                'basic': ['count_purchase', 'ctr'],
                'temporal': ['recency', 'frequency', 'time_window'],
                'user': ['user_stats', 'user_preferences'],
                'product': ['product_stats', 'category_stats'],
                'advanced': ['novelty', 'serendipity']
            },
            'models': {
                'catboost': {
                    'iterations': 500,
                    'learning_rate': 0.05,
                    'depth': 6,
                    'loss_function': 'Logloss',
                    'eval_metric': 'AUC',
                    'early_stopping_rounds': 50,
                    'verbose': 100
                },
                'lightgbm': {
                    'num_iterations': 500,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'objective': 'binary',
                    'metric': 'auc',
                    'early_stopping_rounds': 50,
                    'verbose': 100
                },
                'xgboost': {
                    'n_estimators': 500,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'early_stopping_rounds': 50,
                    'verbose': 100
                },
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'random_state': 42
                }
            },
            'validation': {
                'method': 'temporal',  # 'temporal', 'kaggle', 'random'
                'n_folds': 3,
                'gap_days': 0,
                'test_size': 0.2
            },
            'metrics': ['auc', 'ndcg@10', 'map@10', 'novelty@10', 'serendipity@10'],
            'output': {
                'results_dir': 'results',
                'save_models': True,
                'save_features': False,
                'save_predictions': True
            }
        }
        
        # Update config with defaults for missing values
        for section, values in defaults.items():
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
