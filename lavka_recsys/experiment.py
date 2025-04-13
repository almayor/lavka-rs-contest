import os
import json
import time
import hashlib
from datetime import datetime

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss

from .custom_logging import get_logger
from .data_loader import DataLoader
from .feature_factory import FeatureFactory
from .model_factory import ModelFactory
from .hyperparameter_tuner import HyperparameterTuner

class Experiment:
    """Simplified experiment framework for recommender systems"""
    
    def __init__(self, name, config):
        """Initialize experiment with configuration"""
        # Create a short hash of the config for unique identification
        config_str = json.dumps(config.to_dict(), sort_keys=True).encode('utf-8')
        config_hash = hashlib.md5(config_str).hexdigest()[:6]
        
        self.name = f"{name}_{config_hash}"
        self.config = config
        self.logger = get_logger(f'experiment={name}')
        
        # Initialize components
        self.data_loader = DataLoader(config)
        self.feature_factory = FeatureFactory(config)
        self.model_factory = ModelFactory(config)
        self.results = {}
        
        # Create output directory if needed
        results_dir = config.get('output.results_dir')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save initial configuration
        self._save_config()
        
        # Load data
        self.data_loader.load_data()
    
    def run(self):
        """
        Run a simple experiment with fixed validation.
        Returns:
            Dict: Experiment results
        """
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        model_type = self.config.get("model.type")
        
        self.logger.info(f"Starting experiment: {self.name}")
        self.logger.info(f"Feature names: {feature_names}")
        self.logger.info(f"Model type: {model_type}")
        
        start_time = time.time()
        
        # Create validation split
        history_df, train_df, val_df = self.data_loader.create_validation_split()
        
        # Generate features
        train_features, train_target, cat_columns, _ = self.feature_factory.generate_batch(
            history_df, train_df, feature_names, target_name
        )
        
        val_history = pl.concat([history_df, train_df], how='vertical')
        val_features, val_target, _, val_request_ids = self.feature_factory.generate_batch(
            val_history, val_df, feature_names, target_name
        )
        
        # Train model
        model = self.model_factory.create_model()
        model.train(
            train_features, 
            train_target,
            cat_columns=cat_columns,
            eval_set=(val_features, val_target)
        )
        
        # Predict on validation set
        val_preds = model.predict(val_features)
        val_preds = pl.Series(val_preds)
        
        # Calculate metrics
        metrics = self._calculate_metrics(val_target, val_preds)
        feature_importance = model.get_feature_importance()
        
        # Store results
        experiment_results = {
            'name': self.name,
            'feature_names': feature_names,
            'model_type': model_type,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'runtime': time.time() - start_time,
        }
        self.results = experiment_results
        
        # Save results
        self._save_results()
        
        self.logger.info(f"Experiment metrics: {metrics}")
        self.logger.info(f"Experiment completed in {time.time() - start_time:.2f} seconds")
        
        return experiment_results
    
    def run_with_tuning(self):
        """
        Run experiment with hyperparameter tuning.
        Returns:
            Dict: Experiment results
        """
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        model_type = self.config.get("model.type")
        
        self.logger.info(f"Starting experiment with tuning: {self.name}")
        
        start_time = time.time()
        
        # Run hyperparameter tuning
        tuner = HyperparameterTuner(
            self.config, 
            self.data_loader, 
            self.feature_factory, 
            self.model_factory
        )
        best_params = tuner.tune()
        
        # Create validation split
        history_df, train_df, val_df = self.data_loader.create_validation_split()
        
        # Generate features
        train_features, train_target, cat_columns, _ = self.feature_factory.generate_batch(
            history_df, train_df, feature_names, target_name
        )
        
        val_history = pl.concat([history_df, train_df], how='vertical')
        val_features, val_target, _, val_request_ids = self.feature_factory.generate_batch(
            val_history, val_df, feature_names, target_name
        )
        
        # Train model with best parameters
        model = self.model_factory.create_model(best_params)
        model.train(
            train_features, 
            train_target,
            cat_columns=cat_columns,
            eval_set=(val_features, val_target)
        )
        
        # Predict on validation set
        val_preds = model.predict(val_features)
        val_preds = pl.Series(val_preds)
        
        # Calculate metrics
        metrics = self._calculate_metrics(val_target, val_preds)
        feature_importance = model.get_feature_importance()
        
        # Store results
        experiment_results = {
            'name': self.name,
            'feature_names': feature_names,
            'model_type': model_type,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'best_params': best_params,
            'runtime': time.time() - start_time,
        }
        self.results = experiment_results
        
        # Save results
        self._save_results()
        
        self.logger.info(f"Experiment metrics: {metrics}")
        self.logger.info(f"Experiment completed in {time.time() - start_time:.2f} seconds")
        
        return experiment_results
    
    def predict(self, best_params=None):
        """
        Train on all data and make predictions for the test set.
        Args:
            best_params (dict): Best parameters from tuning (optional)
        Returns:
            pd.DataFrame: Predictions for the test set
        """
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        
        self.logger.info(f"Training final model for prediction")
        
        start_time = time.time()
        
        # Create final split
        history_df, train_df = self.data_loader.create_final_split()
        
        # Generate features for training
        train_features, train_target, cat_columns, _ = self.feature_factory.generate_batch(
            history_df, train_df, feature_names, target_name
        )
        
        # Train final model
        model = self.model_factory.create_model(best_params)
        model.train(
            train_features, 
            train_target,
            cat_columns=cat_columns
        )
        
        # Generate features for test data
        all_history = pl.concat([history_df, train_df], how='vertical')
        test_features = self.feature_factory.generate_features(
            all_history, self.data_loader.test_df, feature_names
        )[0]
        
        # Make predictions
        test_preds = model.predict(test_features)
        
        # Create submission DataFrame
        submission_df = self.data_loader.test_df.select(['index', 'request_id']).with_columns(
            predict=pl.Series(test_preds)
        ).sort('predict', descending=True)
        
        # Save model if configured
        if self.config.get('output.save_model', True):
            model_path = os.path.join(
                self.config.get('output.results_dir', 'results'),
                f"{self.name}_model.pkl"
            )
            model.save(model_path)
            self.logger.info(f"Saved model to {model_path}")
        
        # Save predictions if configured
        if self.config.get('output.save_predictions', True):
            pred_path = os.path.join(
                self.config.get('output.results_dir', 'results'),
                f"{self.name}_predictions.csv"
            )
            submission_df.select('index', 'request_id').write_csv(pred_path)
            self.logger.info(f"Saved predictions to {pred_path}")
        
        self.logger.info(f"Prediction completed in {time.time() - start_time:.2f} seconds")
        
        return submission_df
    
    def _calculate_metrics(self, true_labels, predictions):
        """Calculate basic evaluation metrics"""
        metrics = {}
        
        # AUC
        metrics['auc'] = roc_auc_score(true_labels, predictions)
        
        # Log loss
        metrics['logloss'] = log_loss(true_labels, predictions)
        
        return metrics
    
    def _save_config(self):
        """Save experiment configuration to file"""
        results_dir = self.config.get('output.results_dir', 'results')
        config_path = os.path.join(
            results_dir,
            f"{self.name}_config.json"
        )
        self.config.save(config_path)
        self.logger.info(f"Saved configuration to {config_path}")
    
    def _save_results(self):
        """Save experiment results to file"""
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        results_dir = self.config.get('output.results_dir', 'results')
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