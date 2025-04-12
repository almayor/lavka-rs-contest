import json
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

from .custom_logging import get_logger
from .data_loader import DataLoader
from .feature_factory import FeatureFactory
from .model_factory import ModelFactory


class Experiment:
    """Experiment framework for running and tracking recommender experiments"""
    
    def __init__(self, name, config):
        """Initialize experiment with configuration"""
        self.name = name
        self.config = config
        self.data_loader = DataLoader(config)
        self.feature_factory = FeatureFactory(config)
        self.model_factory = ModelFactory(config)
        self.results = {}
        self.start_time = datetime.now()
        
        # Create output directory if needed
        results_dir = config.get('output.results_dir')
        os.makedirs(results_dir, exist_ok=True)
        
        # # Setup experiment-specific logging
        # self.logger = logging.getLogger(f'experiment-{name}')
        
        # # Add file handler for experiment
        # log_file = os.path.join(results_dir, f"{name}.log")
        # file_handler = logging.FileHandler(log_file)
        # file_handler.setFormatter(logging.Formatter(
        #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        # ))
        # self.logger.addHandler(file_handler)
        self.logger = get_logger(f'experiment={name}')
        
        # Save initial configuration immediately for reproducibility
        # Even if the experiment fails, we'll have the configuration
        self._save_config()
    
    def run(self):
        """Run a complete experiment with given feature sets and model"""
        feature_set = self.config.get("features")
        target_name = self.config.get('target')
        model_type = self.config.get("models.type")

        self.logger.info(f"Starting experiment: {self.name}")
        self.logger.info(f"Feature sets: {feature_set}")
        self.logger.info(f"Model type: {model_type}")
        
        start_time = time.time()
        
        # Load and preprocess data
        train_df, test_df = self.data_loader.load_data()
        
        # Create validation splits
        validation_splits = self.data_loader.create_validation_splits()
        
        # Run cross-validation
        cv_results = self._run_cross_validation(
            validation_splits, feature_set, target_name, model_type
        )
        
        # Train final model
        final_model = self._train_final_model(
            train_df, feature_set, model_type
        )
        
        # Create test predictions
        test_predictions = self._predict_test(
            final_model, train_df, test_df, feature_set
        )
        
        # Save results
        experiment_results = {
            'name': self.name,
            'feature_sets': feature_set,
            'model_type': model_type,
            'cv_results': cv_results,
            'test_predictions': test_predictions,
            'runtime': time.time() - start_time,
        }
        
        self.results = experiment_results
        
        # Save results to file
        self._save_results()
        
        self.logger.info(f"Experiment completed in {time.time() - start_time:.2f} seconds")
        return experiment_results
    
    def _run_cross_validation(self, validation_splits, feature_sets, target_name, model_type):
        """Run cross-validation on validation splits"""
        cv_results = []
        
        for fold_idx, (train_history, train_df, val_df) in enumerate(tqdm(validation_splits, 'cv')):
            self.logger.info(f"Processing fold {fold_idx+1}/{len(validation_splits)}")
            
            train_features = self.feature_factory.generate_features(
                train_history, train_df, feature_sets
            )
            train_target = self.feature_factory.generate_target(
                train_history, train_df, target_name
            )

            val_history = pl.concat(
                [train_history, train_df],
                how='vertical'
            )
            val_features = self.feature_factory.generate_features(
                val_history, val_df, feature_sets
            )
            val_target = self.feature_factory.generate_target(
                val_history, val_df, target_name
            )

            
            # Create and train model
            model = self.model_factory.create_model(model_type)
            model.train(
                train_features, 
                train_target,
                eval_set=(val_features, val_target)
            )
            val_preds = model.predict(val_features)
            
            # Calculate metrics
            fold_metrics = self._calculate_metrics(val_target, val_preds)
            
            # Add fold results
            cv_results.append({
                'fold': fold_idx,
                'metrics': fold_metrics,
                'feature_importance': model.get_feature_importance()
            })
            
            self.logger.info(f"Fold {fold_idx+1} metrics: {fold_metrics}")
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in cv_results[0]['metrics'].keys():
            avg_metrics[metric] = np.mean([r['metrics'][metric] for r in cv_results])
        
        cv_summary = {
            'folds': cv_results,
            'average_metrics': avg_metrics
        }
        
        self.logger.info(f"Cross-validation average metrics: {avg_metrics}")
        return cv_summary
    
    def _train_final_model(self, train_df, feature_sets, model_type):
        """Train final model on all training data"""
        self.logger.info("Training final model on all data")
        
        # Split data for feature generation
        latest_cutoff = train_df['timestamp'].max() - timedelta(days=7)
        history_df = train_df.filter(pl.col('timestamp') < latest_cutoff)
        target_df = train_df.filter(pl.col('timestamp') >= latest_cutoff)
        
        # Generate features
        self.feature_factory.generate_features(
            history_df, target_df, feature_sets
        )
        features_df = self.feature_factory.join_features()
        
        # Setup target variable
        target = train_df.filter(
            pl.col('action_type').is_in(["AT_View", "AT_CartUpdate"])
        ).with_columns(
            target=pl.when(pl.col('action_type') == "AT_View").then(0).otherwise(1)
        )['target']
        
        # Create and train model
        model = self.model_factory.create_model(model_type)
        model.train(features_df, target)
        
        # Save model if configured
        if self.config.get('output.save_models'):
            model_path = os.path.join(
                self.config.get('output.results_dir'),
                f"{self.name}_final_model.model"
            )
            model.save(model_path)
            self.logger.info(f"Saved final model to {model_path}")
        
        return model
    
    def _predict_test(self, model, train_df, test_df, feature_sets):
        """Generate predictions for test data"""
        self.logger.info("Generating test predictions")
        
        # Generate features for test data using all training data as history
        self.feature_factory.generate_features(
            train_df, test_df, feature_sets
        )
        test_features_df = self.feature_factory.join_features()
        
        # Make predictions
        test_predictions = model.predict(test_features_df)
        
        # Create submission dataframe
        submission_df = test_df.select('index', 'request_id')
        submission_df = submission_df.with_columns(
            predict=pl.Series(test_predictions)
        ).sort(
            'predict', descending=True
        ).select(
            'index', 'request_id'
        )
        
        # Save predictions if configured
        if self.config.get('output.save_predictions'):
            submission_path = os.path.join(
                self.config.get('output.results_dir'),
                f"{self.name}_submission.csv"
            )
            submission_df.write_csv(submission_path)
            self.logger.info(f"Saved submission to {submission_path}")
        
        return submission_df
    
    def _calculate_metrics(self, true_labels, predictions):
        """Calculate evaluation metrics"""
        metrics_dict = {}
        
        # Basic classification metrics
        metrics_dict['auc'] = roc_auc_score(true_labels, predictions)
        metrics_dict['logloss'] = log_loss(true_labels, predictions)
        
        # Calculate ranking metrics if specified
        if 'ndcg@10' in self.config.get('metrics'):
            metrics_dict['ndcg@10'] = ndcg_score(
                [true_labels], [predictions], k=10
            )
        
        # Add more metrics as needed
        
        return metrics_dict
    
    def _save_config(self):
        """Save experiment configuration to file"""
        results_dir = self.config.get('output.results_dir')
        config_path = os.path.join(
            results_dir,
            f"{self.name}_config.json"
        )
        self.config.save(config_path)
        self.logger.info(f"Saved experiment configuration to {config_path}")
        return config_path
    
    def _save_results(self):
        """Save experiment results to file"""
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        results_dir = self.config.get('output.results_dir')
        
        # 1. Make sure configuration is saved
        config_path = os.path.join(
            results_dir,
            f"{self.name}_config.json"
        )
        
        # 2. Save results with reference to configuration
        results_path = os.path.join(
            results_dir,
            f"{self.name}_results.json"
        )
        
        # Convert numpy values to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pl.DataFrame):
                return "DataFrame (not serialized)"
            return obj
        
        # Deep copy and convert results
        serializable_results = {
            'experiment_name': self.name,
            'config_file': config_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        for k, v in self.results.items():
            if isinstance(v, dict):
                serializable_results[k] = {
                    sk: convert_to_serializable(sv) for sk, sv in v.items()
                }
            else:
                serializable_results[k] = convert_to_serializable(v)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        self.logger.info(f"Saved results to {results_path}")
