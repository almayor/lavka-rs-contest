import os
import json
import time
import pickle
import hashlib
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss

from .custom_logging import get_logger
from .data_loader import DataLoader
from .feature_factory import FeatureFactory
from .model_factory import ModelFactory, Model
from .hyperparameter_tuner import HyperparameterTuner
from .feature_selector import FeatureSelector
from .trainer import Trainer
from .time_splitter import SplitType  # Only need the enum, not the class
from .cached_feature_factory import CachedFeatureFactory


class ExperimentType(Enum):
    """Enumeration of different experiment types."""
    SINGLE_RUN = "single_run"  # Single model training and evaluation
    TUNING = "tuning"          # With hyperparameter tuning


class Experiment:
    """Unified experiment framework for recommender systems."""
    
    def __init__(self, name, config):
        """
        Initialize experiment with configuration.
        
        Args:
            name: Name of the experiment
            config: Configuration object
        """
        # Create a short hash of the config for unique identification
        config_str = json.dumps(config.to_dict(), sort_keys=True).encode('utf-8')
        config_hash = hashlib.md5(config_str).hexdigest()[:6]
        
        self.name = f"{name}_{config_hash}"
        self.config = config
        self.logger = get_logger(f"{self.__class__.__name__}({self.name})")
        
        # Initialize data loader
        self.data_loader = DataLoader(config)
        
        # Initialize model factory
        self.model_factory = ModelFactory(config)
        
        # Initialize feature factory with built-in caching 
        self.feature_factory = CachedFeatureFactory(config=config)
        
        # Initialize the unified trainer
        self.trainer = Trainer(config, self.data_loader, self.feature_factory, self.model_factory)
        
        # TimeSplitter is now used indirectly through Trainer
        
        # Initialize feature selector if enabled
        use_feature_selection = config.get('feature_selection.enabled', False)
        self.feature_selector = FeatureSelector(config) if use_feature_selection else None
        
        # Initialize hyperparameter tuner if enabled
        self.use_hyperparameter_tuning = config.get('experiment.use_hyperparameter_tuning', False)
        self.tuner = None  # Will be initialized in setup() if needed
        
        # Get experiment type from config
        experiment_type = config.get('experiment.type', 'single_run')
        if isinstance(experiment_type, str):
            if experiment_type.lower() == 'single_run':
                self.experiment_type = ExperimentType.SINGLE_RUN
            elif experiment_type.lower() == 'tuning':
                self.experiment_type = ExperimentType.TUNING
            else:
                self.logger.warning(f"Unknown experiment type '{experiment_type}', defaulting to single_run")
                self.experiment_type = ExperimentType.SINGLE_RUN
        else:
            self.experiment_type = ExperimentType.SINGLE_RUN
            
        # Get split type configuration
        split_type_str = config.get('training.split_type', 'standard')
        if split_type_str == 'fixed_window':
            self.split_type = SplitType.FIXED_WINDOW
        elif split_type_str == 'expanding_window':
            self.split_type = SplitType.EXPANDING_WINDOW
        else:  # Default to standard
            self.split_type = SplitType.STANDARD
            
        self.results = {}
        self.last_trained_model = None  # Store last trained model for reuse
        
        # Create all output directories
        self._create_output_directories()
        
        # Save initial configuration
        self._save_config()
    
    def setup(self):
        """
        Prepare the experiment by loading data and initializing components.
        This should be called before running the experiment.
        """
        # Load data
        self.data_loader.load_data()
        
        # Initialize hyperparameter tuner if needed
        if self.use_hyperparameter_tuning:
            self.tuner = HyperparameterTuner(
                self.config, 
                self.data_loader, 
                self.feature_factory, 
                self.model_factory
            )

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
    
    def run(self) -> Dict:
        """
        Run experiment based on configured experiment type.
        This is the main method to execute an experiment.
        
        Returns:
            Dict: Experiment results
        """
        self.logger.info(f"Starting experiment: {self.name} (type: {self.experiment_type.value}, split type: {self.split_type.value})")
        
        # Choose the appropriate experiment method based on type
        if self.experiment_type == ExperimentType.SINGLE_RUN:
            results, model = self._run_single_run()
            self.last_trained_model = model
            return results
        elif self.experiment_type == ExperimentType.TUNING:
            results = self._run_with_tuning()
            return results
        else:
            self.logger.error(f"Unknown experiment type: {self.experiment_type}")
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")
    
    def evaluate(self) -> Dict:
        """
        Evaluate the model on test data and optionally create Kaggle submission.
        This should be called after run().
        
        Returns:
            Dict: Evaluation results
        """
        # Check if Kaggle simulation is enabled
        perform_simulation = self.config.get('experiment.evaluation.perform_kaggle_simulation', True)
        create_submission = self.config.get('experiment.evaluation.create_submission', True)
        
        if not perform_simulation and not create_submission:
            self.logger.info("Evaluation skipped (disabled in config)")
            return {}
            
        # Perform Kaggle simulation if enabled
        evaluation_results = {}
        if perform_simulation:
            self.logger.info("Running Kaggle performance simulation")
            metrics, preds_df, model = self._run_kaggle_simulation()
            evaluation_results['kaggle_simulation'] = metrics
            self.last_trained_model = model
            
        # Create Kaggle submission if enabled 
        if create_submission:
            self.logger.info("Creating Kaggle submission")
            submission_df = self.create_kaggle_submission(self.last_trained_model)
            evaluation_results['submission_created'] = True
            
        return evaluation_results
    
    def _run_single_run(self) -> Tuple[Dict, Model]:
        """
        Run a single experiment using the unified Trainer.
        
        Returns:
            Tuple[Dict, Model]: Experiment results and the trained model
        """
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        model_type = self.config.get("model.type")
        
        self.logger.info(f"Running single run experiment")
        self.logger.info(f"Feature names: {feature_names}")
        self.logger.info(f"Model type: {model_type}")
        self.logger.info(f"Split type: {self.split_type.value}")
        
        start_time = time.time()
        
        # Train the model using our unified Trainer
        model = self.trainer.train(split_type=self.split_type)
        
        # Use the trainer to evaluate the model
        self.logger.info("Evaluating model using the trainer...")
        
        # Force reload data to make sure we have fresh data for evaluation
        try:
            self.data_loader.load_data()
            if self.data_loader.train_df is None or self.data_loader.train_df.is_empty():
                self.logger.error("Training data is empty or None. Cannot evaluate model.")
                metrics = {"error": "Training data is empty or None"}
                score = 0
            else:
                score, metrics = self.trainer.evaluate_model(model)
        except Exception as e:
            self.logger.error(f"Error loading data for evaluation: {str(e)}")
            metrics = {"error": f"Error loading data: {str(e)}"}
            score = 0
        
        if score == 0 and "error" in metrics:
            self.logger.warning(f"Model evaluation failed: {metrics.get('error')}")
            # Return empty metrics if evaluation failed
            metrics = {}
        
        # Get feature importance if the model supports it
        feature_importance = {}
        if model:
            try:
                feature_importance = model.get_feature_importance()
                if not feature_importance:
                    self.logger.warning("Model returned empty feature importance dictionary")
            except Exception as e:
                self.logger.error(f"Error getting feature importance: {str(e)}")
                # Continue with empty feature importance
        
        # Store results
        experiment_results = {
            'name': self.name,
            'experiment_type': 'single_run',
            'split_type': self.split_type.value,
            'feature_names': feature_names,
            'model_type': model_type,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'training_config': {
                'split_type': self.split_type.value,
                'history_days': self.config.get('training.history_days'),
                'target_days': self.config.get('training.target_days'),
                'step_days': self.config.get('training.step_days', 7),
                'max_splits': self.config.get('training.max_splits', 10)
            },
            'runtime': time.time() - start_time,
        }
        self.results = experiment_results
        
        # Save results
        self._save_results()
        
        # Log top features if we have feature importance
        if feature_importance:
            self.logger.info("Top 10 most important features:")
            try:
                # First filter out None values
                valid_importances = {k: v for k, v in feature_importance.items() if v is not None}
                top_features = sorted(
                    valid_importances.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
                for feature, importance in top_features:
                    # Handle different types of importance values
                    if importance is None:
                        self.logger.info(f"  {feature}: None")
                    else:
                        try:
                            self.logger.info(f"  {feature}: {float(importance):.6f}")
                        except (ValueError, TypeError):
                            # If it can't be formatted as float, print as-is
                            self.logger.info(f"  {feature}: {importance}")
            except Exception as e:
                self.logger.error(f"Error formatting feature importance: {str(e)}")
                # Just print the raw dictionary as fallback
                self.logger.info(f"  Feature importance: {feature_importance}")
        
        self.logger.info(f"Experiment metrics: {metrics}")
        self.logger.info(f"Experiment completed in {time.time() - start_time:.2f} seconds")
        
        # Return both the results and the trained model
        return experiment_results, model
    
    def _run_with_tuning(self) -> Dict:
        """
        Run experiment with hyperparameter tuning.
        
        Returns:
            Dict: Experiment results
        """
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        model_type = self.config.get("model.type")
        
        self.logger.info(f"Running experiment with hyperparameter tuning")
        self.logger.info(f"Feature names: {feature_names}")
        self.logger.info(f"Model type: {model_type}")
        self.logger.info(f"Split type: {self.split_type.value}")
        
        start_time = time.time()
        
        # Initialize tuner if not already done
        if self.tuner is None:
            self.tuner = HyperparameterTuner(
                self.config, 
                self.data_loader, 
                self.feature_factory, 
                self.model_factory
            )
        
        # Run hyperparameter tuning
        best_params = self.tuner.tune()
        
        # Train model with best parameters using our unified Trainer
        model = self.trainer.train(
            split_type=self.split_type,
            model_params=best_params
        )
        
        # Save trained model for potential reuse
        self.last_trained_model = model
        
        # Use the trainer to evaluate the model
        self.logger.info("Evaluating model using the trainer...")
        
        # Force reload data to make sure we have fresh data for evaluation
        try:
            self.data_loader.load_data()
            if self.data_loader.train_df is None or self.data_loader.train_df.is_empty():
                self.logger.error("Training data is empty or None. Cannot evaluate model.")
                metrics = {"error": "Training data is empty or None"}
                score = 0
            else:
                score, metrics = self.trainer.evaluate_model(model)
        except Exception as e:
            self.logger.error(f"Error loading data for evaluation: {str(e)}")
            metrics = {"error": f"Error loading data: {str(e)}"}
            score = 0
        
        if score == 0 and "error" in metrics:
            self.logger.warning(f"Model evaluation failed: {metrics.get('error')}")
            # Return empty metrics if evaluation failed
            metrics = {}
        
        # Get feature importance if the model supports it
        feature_importance = {}
        if model:
            try:
                feature_importance = model.get_feature_importance()
                if not feature_importance:
                    self.logger.warning("Model returned empty feature importance dictionary")
            except Exception as e:
                self.logger.error(f"Error getting feature importance: {str(e)}")
                # Continue with empty feature importance
        
        # Store results
        experiment_results = {
            'name': self.name,
            'experiment_type': 'tuning',
            'split_type': self.split_type.value,
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
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Experiment completed in {time.time() - start_time:.2f} seconds")
        
        return experiment_results
    
    def _run_kaggle_simulation(self) -> Tuple[Dict, pl.DataFrame, Model]:
        """
        Evaluates model performance on a simulated Kaggle test set.
        
        This method creates a clean split of data to mimic the Kaggle submission scenario:
        1. Uses the last X days of data as a "test set" (simulating the Kaggle test set)
        2. Uses the previous Y days as training data (with validation)
        3. Everything before that becomes history data
        
        Returns:
            Tuple[Dict, pl.DataFrame, Model]: Evaluation metrics, predictions dataframe, and trained model
        """
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        
        self.logger.info(f"Evaluating model performance on simulated Kaggle test set")
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
        
        # Train a model based on the configured split type
        model = self._train_for_kaggle_simulation(history_df, train_df, val_df)
        
        # Generate features for test evaluation (using all available history)
        available_history = pl.concat([history_df, train_df, val_df], how='vertical')
        test_features, test_target, _, test_request_ids = self.feature_factory.generate_batch(
            available_history, test_df, feature_names, target_name
        )
        
        # Make predictions
        self.logger.info(f"Making predictions on simulated Kaggle test set...")
        test_preds = model.predict(test_features)
        test_preds_series = pl.Series(test_preds)
        
        # Calculate metrics
        metrics = self._calculate_metrics(test_target, test_preds_series, test_request_ids)
        
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
            metrics_dir = self.config.get('output.metrics_dir', 'results/metrics')
            
            # Save predictions to results directory
            preds_path = os.path.join(results_dir, f"{self.name}_kaggle_sim_preds.csv")
            preds_df.write_csv(preds_path)
            self.logger.info(f"Saved predictions to {preds_path}")
            
            # Save metrics to metrics directory
            metrics_path = os.path.join(metrics_dir, f"{self.name}_kaggle_sim_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"Saved metrics to {metrics_path}")
        
        return metrics, preds_df, model
    
    def _train_for_kaggle_simulation(self, history_df, train_df, val_df) -> Model:
        """
        Train a model for Kaggle simulation with the configured split type.
        
        Args:
            history_df: Historical data for feature generation
            train_df: Training data
            val_df: Validation data
            
        Returns:
            Model: Trained model
        """
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        
        # IMPORTANT: Only use available training data (history + train), NOT validation
        # This prevents data leakage when validation is part of the test simulation period
        training_data = pl.concat([history_df, train_df], how='vertical')
        
        self.logger.info(f"Training on {len(training_data)} records, "
                       f"from {training_data['timestamp'].min()} to {training_data['timestamp'].max()}")
        
        # Create a temporary Trainer for this simulation
        # This avoids modifying the original trainer's configuration
        temp_config = self.config.copy()
        temp_config.set('feature_caching', True)
        
        # Create a temporary data loader with only the training data
        temp_data_loader = DataLoader(temp_config)
        temp_data_loader.train_df = training_data
        temp_data_loader.test_df = self.data_loader.test_df
        
        # Create a new trainer with the temporary components
        temp_trainer = Trainer(
            temp_config, 
            temp_data_loader,
            self.feature_factory,
            self.model_factory
        )
        
        # Apply hyperparameter tuning if enabled
        model_params = None
        if self.use_hyperparameter_tuning and self.tuner is not None:
            self.logger.info("Applying hyperparameter tuning for Kaggle simulation")
            model_params = self.tuner.tune()
            self.logger.info(f"Using best parameters: {model_params}")
        
        # Train model with the unified trainer
        self.logger.info(f"Training model with {self.split_type.value} split type for Kaggle simulation")
        model = temp_trainer.train(
            split_type=self.split_type,
            model_params=model_params,
            provided_train_df=training_data
        )
        
        return model
    
    def create_kaggle_submission(self, model=None) -> pl.DataFrame:
        """
        Create a submission file for Kaggle using a pre-trained model or by training a new one.
        
        Args:
            model: Pre-trained model to use (optional)
            
        Returns:
            pl.DataFrame: Submission dataframe for Kaggle
        """
        feature_names = self.config.get("features")
        target_name = self.config.get('target')
        
        start_time = time.time()
        
        # Use provided model, last trained model, or train a new one
        if model is None:
            if self.last_trained_model is not None:
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
                train_features, train_target, cat_columns, _ = self.feature_factory.generate_batch(
                    train_data, train_data, feature_names, target_name
                )
                
                # Generate validation features
                val_features, val_target, _, _ = self.feature_factory.generate_batch(
                    train_data, val_data, feature_names, target_name
                )
                
                # Train model with validation for early stopping
                model = self.model_factory.create_model()
                model.train(
                    train_features, 
                    train_target,
                    cat_columns=cat_columns,
                    eval_set=(val_features, val_target)
                )
        
        # Generate features for Kaggle test data using all available training data
        self.logger.info(f"Generating features for Kaggle test data")
        all_train_data = self.data_loader.train_df
        test_features, cat_columns, _ = self.feature_factory.generate_features_only(
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
                self.config.get('output.model_cache_dir', 'results/model_cache'),
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
        
        # Add a to_csv method to the Polars DataFrame to make it compatible with pandas-style code
        # This allows users to call df.to_csv() instead of df.write_csv()
        submission_df_with_to_csv = submission_df.clone()
        submission_df_with_to_csv.to_csv = lambda path_or_buf, index=None, **kwargs: submission_df.write_csv(path_or_buf)
        
        return submission_df_with_to_csv
    
    def _calculate_metrics(self, true_labels, predictions, request_ids=None):
        """
        Calculate evaluation metrics.
        
        Args:
            true_labels: Ground truth labels
            predictions: Model predictions
            request_ids: Request IDs for ranking metrics (optional)
            
        Returns:
            Dict: Dictionary of metrics
        """
        metrics = {}
        
        # AUC
        try:
            metrics['auc'] = roc_auc_score(true_labels, predictions)
        except:
            metrics['auc'] = None
            self.logger.warning("Could not calculate AUC")
        
        # Log loss
        try:
            metrics['logloss'] = log_loss(true_labels, predictions)
        except:
            metrics['logloss'] = None
            self.logger.warning("Could not calculate Log Loss")
            
        # Calculate nDCG@10 if request_ids are provided
        if request_ids is not None:
            try:
                from .metrics import RankingMetrics
                metrics['ndcg@10'] = RankingMetrics.ndcg_at_k(true_labels, predictions, request_ids, k=10)
            except Exception as e:
                metrics['ndcg@10'] = None
                self.logger.warning(f"Could not calculate nDCG@10: {str(e)}")
        
        return metrics
    
    def _create_output_directories(self):
        """Create all necessary output directories."""
        # Get directories from config or use defaults
        results_dir = self.config.get('output.results_dir', 'results')
        model_cache_dir = self.config.get('output.model_cache_dir', 'results/model_cache')
        feature_cache_dir = self.config.get('output.feature_cache_dir', 'results/feature_cache')
        metrics_dir = self.config.get('output.metrics_dir', 'results/metrics')
        visualizations_dir = self.config.get('output.visualizations_dir', 'results/visualizations')
        
        # Additional directories for better organization
        feature_selection_cache_dir = self.config.get('output.feature_selection_cache_dir', 
                                                   os.path.join(results_dir, 'feature_selection'))
        backups_dir = os.path.join(results_dir, 'backups')
        experiment_dir = os.path.join(results_dir, f'{self.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Create all directories
        directories = [
            results_dir, 
            model_cache_dir, 
            feature_cache_dir, 
            metrics_dir, 
            visualizations_dir,
            feature_selection_cache_dir,
            backups_dir,
            experiment_dir,
            os.path.join(experiment_dir, 'feature_selection')
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                self.logger.debug(f"Created directory: {directory}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {directory}: {str(e)}")
        
        # Store experiment directory path for later use
        self.experiment_dir = experiment_dir
        self.logger.info(f"Created experiment directory: {experiment_dir}")
                
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
        
        metrics_dir = self.config.get('output.metrics_dir', 'results/metrics')
        results_path = os.path.join(
            metrics_dir,
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
            elif isinstance(obj, datetime):
                return obj.isoformat()
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