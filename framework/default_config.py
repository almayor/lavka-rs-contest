DEFAULT_CONFIG = {
    'data': {
        'train_path': 'train.parquet',
        'test_path': 'test.parquet',
        'sample_size': None,  # None = use all data
        'random_seed': 42
    },
    'preprocessing': {
        'normalize_timestamps': True,
        'clean_text': False
    },
    'cleaning': {
        'remove_duplicates': True,
        'remove_sessions_only_views': False
    },
    'feature_set': ['basic'],
    'history_generation':{
        'method': 'basic',  # 'basic' or 'rolling_window'
        'window_size': 30,
    },
    'models_config': {
        'type': 'catboost', # 'catboost' or 'lightgbm' or 'xgboost' or 'random_forest'
        'catboost': {
            'iterations': 300,
            'learning_rate': 0.01,
            'depth': 2,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'early_stopping_rounds': 50,
            # 'verbose': 100
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
        'type': 'temporal',
        'n_folds': 2,
    },
    'target': 'simple', # 'simple', 'viewed
    'metrics': ['auc', 'ndcg@10', 'map@10', 'novelty@10', 'serendipity@10'],
    'output': {
        'results_dir': 'results',
        'save_models': True,
        'save_features': False,
        'save_predictions': True
    }
}