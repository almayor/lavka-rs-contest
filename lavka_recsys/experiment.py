import os
import json
import time
import pickle
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
from .feature_selector import FeatureSelector
from .enhanced_training import EnhancedTrainingPipeline

class Experiment:
    """Simplified experiment framework for recommender systems"""
    
    def __init__(self, name, config):
        """Initialize experiment with configuration"""
        # Create a short hash of the config for unique identification
        config_str = json.dumps(config.to_dict(), sort_keys=True).encode('utf-8')
        config_hash = hashlib.md5(config_str).hexdigest()[:6]
        
        self.name = f"{name}_{config_hash}"
        self.config = config
        self.logger = get_logger(f"{self.__class__.__name__}({self.name})")
        
        # Initialize components
        self.data_loader = DataLoader(config)
        self.model_factory = ModelFactory(config)
        self.feature_selector = FeatureSelector(config) if config.get('feature_selection.enabled', False) else None
        
        # Initialize feature factory with caching
        self.feature_factory = FeatureFactory(config)
        
        # Create cached feature generator for consistent caching across all methods
        from .cached_feature_generator import CachedFeatureGenerator
        self.cached_feature_generator = CachedFeatureGenerator(self.feature_factory)
        
        self.results = {}
        self.last_trained_model = None  # Store last trained model for reuse
        
        # Create output directory if needed
        results_dir = config.get('output.results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save initial configuration
        self._save_config()
    
    def setup(self):
        """Prepare the experiment by loading data and initializing components"""
        self.data_loader.load_data()

        # Set up feature selector if enabled
        if self.feature_selector:
            self.logger.info("Setting up feature selector")
            
            # Try to load directly from cache first
            if self.feature_selector.load_from_cache():
                self.logger.info("Successfully loaded feature selector from cache")
                # Register the loaded feature selector with the feature factory
                self.feature_factory.register_feature_selector(self.feature_selector)
                return
                
            # If cache loading failed, proceed with normal feature selection
            self.logger.info("No valid cache found. Training feature selector...")
            history_df, train_df = self.data_loader.create_final_split()
            train_features, train_target, cat_columns, _ = self.feature_factory.generate_batch(
                history_df, train_df
            )
            
            # Train the feature selector
            self.feature_selector.train(
                train_features,
                train_target,
                cat_columns=cat_columns,
                use_cache=True  # Will save to cache but not try to load
            )
            self.feature_factory.register_feature_selector(self.feature_selector)
            self.logger.info("Feature selector trained and saved to cache")
        else:
            self.logger.info("Feature selector disabled")
    
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
        
        # Generate features for training
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
    
    def run_with_full_history(self):
        """
        Run experiment using full history approach for more robust training.
        
        Returns:
            Tuple[Dict, Model]: Experiment results and the trained model
        """
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        model_type = self.config.get("model.type")
        
        self.logger.info(f"Starting full history experiment: {self.name}")
        self.logger.info(f"Feature names: {feature_names}")
        self.logger.info(f"Model type: {model_type}")
        self.logger.info(f"Will report nDCG@10 for both training and validation sets")
        
        start_time = time.time()
        
        # Get training configuration
        target_days = self.config.get('training.target_days', 1)
        step_days = self.config.get('training.step_days', 7)
        max_splits = self.config.get('training.max_splits', 10)
        validation_days = self.config.get('training.validation_days')
        
        self.logger.info(f"Training configuration:")
        self.logger.info(f"  Target days per split: {target_days}")
        self.logger.info(f"  Days between splits: {step_days}")
        self.logger.info(f"  Maximum number of splits: {max_splits}")
        if validation_days:
            self.logger.info(f"  Validation days after each target: {validation_days}")
        
        # Use the enhanced training pipeline with full history
        pipeline = EnhancedTrainingPipeline(
            self.config, 
            self.data_loader, 
            self.feature_factory, 
            self.model_factory
        )
        
        # Train model using full history approach
        self.logger.info("Starting model training with full history...")
        model = pipeline.train_with_full_history()
        
        # Create final validation split for evaluation
        self.logger.info("Creating validation split for evaluation...")
        history_df, train_df, val_df = self.data_loader.create_validation_split()
        val_history = pl.concat([history_df, train_df], how='vertical')
        
        # Generate features for validation 
        self.logger.info("Generating features for validation...")
        val_features, val_target, _, val_request_ids = self.feature_factory.generate_batch(
            val_history, val_df, feature_names, target_name
        )
        
        # Predict on validation set
        self.logger.info("Making predictions on validation set...")
        val_preds = model.predict(val_features)
        val_preds = pl.Series(val_preds)
        
        # Calculate metrics
        self.logger.info("Calculating performance metrics...")
        metrics = self._calculate_metrics(val_target, val_preds)
        feature_importance = model.get_feature_importance()
        
        # Store results
        experiment_results = {
            'name': self.name,
            'feature_names': feature_names,
            'model_type': model_type,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'training_config': {
                'target_days': target_days,
                'step_days': step_days,
                'max_splits': max_splits
            },
            'runtime': time.time() - start_time,
        }
        self.results = experiment_results
        
        # Save results
        self._save_results()
        
        # Log top features
        self.logger.info("Top 10 most important features:")
        top_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        for feature, importance in top_features:
            self.logger.info(f"  {feature}: {importance:.6f}")
        
        self.logger.info(f"Experiment metrics: {metrics}")
        self.logger.info(f"Experiment completed in {time.time() - start_time:.2f} seconds")
        
        # Also return the trained model so it can be reused for prediction
        return experiment_results, model
    
    def evaluate_kaggle_performance(self, use_full_history=False):
        """
        Evaluates model performance on a simulated Kaggle test set.
        
        This method creates a clean split of data to mimic the Kaggle submission scenario:
        1. Uses the last X days of your data as a "test set" (simulating the Kaggle test set)
        2. Uses the previous Y days as training data (with validation)
        3. Everything before that becomes history data
        
        IMPORTANT: This should be done BEFORE any other training to avoid data leakage.
        
        Args:
            use_full_history: If True, uses full history sliding window approach;
                             otherwise uses standard training with validation
        
        Returns:
            Tuple[Dict, pl.DataFrame]: Evaluation metrics and predictions dataframe
        """
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        
        self.logger.info(f"Evaluating model performance on simulated Kaggle test set")
        self.logger.warning(f"IMPORTANT: This should be called BEFORE other training methods to avoid data leakage")
        start_time = time.time()
        
        # Get configuration
        test_days = self.config.get('kaggle_simulation.test_days', 30)
        train_days = self.config.get('kaggle_simulation.train_days', 30)
        validation_ratio = self.config.get('kaggle_simulation.validation_ratio', 0.2)
        
        self.logger.info(f"Creating Kaggle simulation split: {test_days} days for test, "
                        f"{train_days} days for training, {validation_ratio:.1%} validation ratio")
        
        # Load fresh data to avoid any contamination
        self.data_loader.load_data()
        history_df, train_df, val_df, test_df = self.data_loader.create_kaggle_simulation_split(
            test_days=test_days, 
            train_days=train_days,
            validation_ratio=validation_ratio
        )
        
        # Train model (full history or standard approach)
        if use_full_history:
            model = self._train_with_full_history_for_evaluation(history_df, train_df, val_df)
        else:
            model = self._train_with_validation_for_evaluation(history_df, train_df, val_df)
        
        # Generate features for test evaluation (using all available history)
        available_history = pl.concat([history_df, train_df, val_df], how='vertical')
        test_features, test_target, _, test_request_ids = self.cached_feature_generator.generate_batch(
            available_history, test_df, feature_names, target_name
        )
        
        # Make predictions
        self.logger.info(f"Making predictions on simulated Kaggle test set...")
        test_preds = model.predict(test_features)
        test_preds_series = pl.Series(test_preds)
        
        # Calculate metrics
        metrics = self._calculate_metrics(test_target, test_preds_series)
        
        # Try to calculate nDCG@10 if we have request_ids
        try:
            from .metrics import RankingMetrics
            ndcg = RankingMetrics.ndcg_at_k(test_target, test_preds_series, test_request_ids, k=10)
            metrics['ndcg@10'] = ndcg
            self.logger.info(f"nDCG@10 on simulated Kaggle test set: {ndcg:.4f}")
        except Exception as e:
            self.logger.warning(f"Could not calculate nDCG@10: {str(e)}")
        
        # Create predictions DataFrame for analysis
        preds_df = pl.DataFrame({
            'request_id': test_request_ids,
            'target': test_target,
            'predict': test_preds_series
        })
        
        self.logger.info(f"Simulated Kaggle test metrics: {metrics}")
        self.logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
        
        # Save results for analysis
        if self.config.get('output.save_evaluation_results', True):
            results_dir = self.config.get('output.results_dir', 'results')
            method = "full_history" if use_full_history else "standard"
            
            preds_path = os.path.join(results_dir, f"{self.name}_kaggle_sim_{method}_preds.csv")
            preds_df.write_csv(preds_path)
            self.logger.info(f"Saved predictions to {preds_path}")
            
            # Also save metrics
            metrics_path = os.path.join(results_dir, f"{self.name}_kaggle_sim_{method}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"Saved metrics to {metrics_path}")
        
        # Store the last trained model for potential reuse
        self.last_trained_model = model
        
        return metrics, preds_df, model
    
    def _train_with_validation_for_evaluation(self, history_df, train_df, val_df):
        """Helper method to train a model with validation for evaluation"""
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        
        self.logger.info("Using standard training approach with validation")
        
        # Generate features for training using cached feature generator
        train_features, train_target, cat_columns, _ = self.cached_feature_generator.generate_batch(
            history_df, train_df, feature_names, target_name
        )
        
        # Generate features for validation
        if len(val_df) > 0:
            val_history = pl.concat([history_df, train_df], how='vertical')
            val_features, val_target, _, _ = self.cached_feature_generator.generate_batch(
                val_history, val_df, feature_names, target_name
            )
            eval_set = (val_features, val_target)
            self.logger.info(f"Using {len(val_df)} validation records for early stopping")
        else:
            eval_set = None
            self.logger.info("No validation data available for early stopping")
        
        # Train model with validation
        model = self.model_factory.create_model()
        model.train(
            train_features, 
            train_target,
            cat_columns=cat_columns,
            eval_set=eval_set  # This will enable early stopping
        )
        
        return model
    
    def _train_with_full_history_for_evaluation(self, history_df, train_df, val_df):
        """Helper method to train a model with full history approach for evaluation"""
        self.logger.info("Using full history training approach")
        
        # Configure sliding window parameters for training
        target_days = self.config.get('training.target_days', 7)
        step_days = self.config.get('training.step_days', 7)
        max_splits = self.config.get('training.max_splits', 10)
        validation_days = self.config.get('training.validation_days')
        
        self.logger.info(f"Full history parameters: target_days={target_days}, "
                       f"step_days={step_days}, max_splits={max_splits}")
        
        # IMPORTANT: Only use available training data (history + train), NOT validation
        # This prevents data leakage when validation is part of the test simulation period
        training_data = pl.concat([history_df, train_df], how='vertical')
        
        self.logger.info(f"Training on {len(training_data)} records, "
                       f"from {training_data['timestamp'].min()} to {training_data['timestamp'].max()}")
        
        # Create temporary custom pipeline for this test only
        from .enhanced_training import EnhancedTrainingPipeline
        
        # Create a clean data loader with only the training data (no test data, no validation)
        temp_data_loader = DataLoader(self.config)
        temp_data_loader.train_df = training_data
        temp_data_loader.test_df = self.data_loader.test_df
        
        # Create the training pipeline with proper configuration
        temp_config = self.config.copy()
        
        # Update config to use validation_days parameter if configured
        if validation_days:
            # Use the same validation_days as in the main config
            self.logger.info(f"Using validation_days={validation_days} for sliding window validation")
        else:
            # If no explicit validation days set, use 1/4 of target_days as a default
            validation_days = max(1, target_days // 4)
            temp_config.set('training.validation_days', validation_days)
            self.logger.info(f"Setting default validation_days={validation_days} for sliding window validation")
            
        # Ensure we're using cached feature generator for training
        temp_config.set('feature_caching', True)
        
        # Create training pipeline with updated config
        training_pipeline = EnhancedTrainingPipeline(
            temp_config, 
            temp_data_loader,
            self.cached_feature_generator,
            self.model_factory
        )
        
        # Train model with full history approach, explicitly passing the training data
        # to prevent it from loading the full dataset again
        model = training_pipeline.train_with_full_history(provided_train_df=training_data)
        
        return model
    
    def create_kaggle_submission(self, model=None, best_params=None):
        """
        Create a submission file for Kaggle using a pre-trained model or by training a new one.
        
        Args:
            model: Pre-trained model to use (e.g. from evaluate_kaggle_performance) (optional)
            best_params: Best hyperparameters from tuning (optional)
            
        Returns:
            pl.DataFrame: Submission dataframe for Kaggle
        """
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        
        start_time = time.time()
        
        # Use provided model, last trained model, or train a new one
        if model is None:
            if hasattr(self, 'last_trained_model') and self.last_trained_model is not None:
                model = self.last_trained_model
                self.logger.info(f"Using last trained model for Kaggle submission")
            else:
                self.logger.info(f"Training new model for Kaggle submission")
                
                # Create final split - use all available data with validation
                train_split_ratio = 0.8
                val_split_ratio = 1 - train_split_ratio
                
                # Load data
                self.data_loader.load_data()
                train_df = self.data_loader.train_df
                
                # Split chronologically for training and validation
                train_df = train_df.sort('timestamp')
                split_idx = int(len(train_df) * train_split_ratio)
                
                train_data = train_df.slice(0, split_idx)
                val_data = train_df.slice(split_idx)
                
                self.logger.info(f"Using {len(train_data)} records for training, {len(val_data)} for validation")
                
                # Generate features
                train_features, train_target, cat_columns, _ = self.cached_feature_generator.generate_batch(
                    train_data, train_data, feature_names, target_name
                )
                
                # Generate validation features
                val_features, val_target, _, _ = self.cached_feature_generator.generate_batch(
                    train_data, val_data, feature_names, target_name
                )
                
                # Train model with validation for early stopping
                model = self.model_factory.create_model(best_params)
                model.train(
                    train_features, 
                    train_target,
                    cat_columns=cat_columns,
                    eval_set=(val_features, val_target)
                )
        
        # Generate features for Kaggle test data using all available training data
        # Use generate_features_only to avoid generating targets (which aren't available in test data)
        self.logger.info(f"Generating features for Kaggle test data (without targets)")
        all_train_data = self.data_loader.train_df
        test_features, cat_columns, _ = self.cached_feature_generator.generate_features_only(
            all_train_data, self.data_loader.test_df, feature_names
        )
        
        # Make predictions
        self.logger.info(f"Making predictions for Kaggle submission")
        test_preds = model.predict(test_features)
        
        # Create submission DataFrame
        # First, create a df with index, request_id, and prediction
        full_df = self.data_loader.test_df.select(['index', 'request_id']).with_columns(
            predict=pl.Series(test_preds)
        )
        
        # Sort by prediction score (descending) to get items in order of relevance
        sorted_df = full_df.sort('predict', descending=True)
        
        # Create final submission with only index and request_id
        submission_df = sorted_df.select(['index', 'request_id'])
        
        # Save model if configured
        if self.config.get('output.save_model', True):
            model_path = os.path.join(
                self.config.get('output.results_dir', 'results'),
                f"{self.name}_kaggle_model.pkl"
            )
            model.save(model_path)
            self.logger.info(f"Saved model to {model_path}")
        
        # Save predictions if configured
        if self.config.get('output.save_predictions', True):
            pred_path = os.path.join(
                self.config.get('output.results_dir', 'results'),
                f"{self.name}_kaggle_submission.csv"
            )
            submission_df.write_csv(pred_path)
            self.logger.info(f"Saved Kaggle submission to {pred_path}")
        
        self.logger.info(f"Kaggle submission created in {time.time() - start_time:.2f} seconds")
        
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