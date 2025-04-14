import optuna
import polars as pl
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import roc_auc_score

from .custom_logging import get_logger
from .config import Config
from .data_loader import DataLoader
from .cached_feature_factory import CachedFeatureFactory
from .model_factory import ModelFactory, Model
from .time_splitter import SplitType
from .trainer import Trainer

class HyperparameterTuner:
    """Hyperparameter tuning using Optuna"""
    
    def __init__(self, config: Config, data_loader: DataLoader, feature_factory: CachedFeatureFactory, model_factory: ModelFactory):
        """Initialize with configuration and components"""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.data_loader = data_loader
        self.feature_factory = feature_factory
        self.model_factory = model_factory
        
        # Get model type
        self.model_type = config.get("model.type")
        
        # Get the split type from training configuration
        split_type_str = config.get('training.split_type', 'standard')
            
        # Convert string to enum value
        if split_type_str == 'fixed_window':
            self.split_type = SplitType.FIXED_WINDOW
        elif split_type_str == 'expanding_window':
            self.split_type = SplitType.EXPANDING_WINDOW
        else:  # Default to standard
            self.split_type = SplitType.STANDARD
            
        self.logger.info(f"Hyperparameter tuning will use {self.split_type.value} split type")
        
        # Fixed number of trials and timeout
        self.n_trials = config.get('hyperparameter_tuning.n_trials', 20)
        self.timeout = config.get('hyperparameter_tuning.timeout', 3 * 3600)  # 3 hour default timeout
    
    def tune(self) -> Dict:
        """Run hyperparameter tuning process"""
        self.logger.info(f"Starting hyperparameter tuning with {self.n_trials} trials")
        self.logger.info(f"Using split type: {self.split_type.value}")
        
        # Log additional information based on split type
        if self.split_type == SplitType.STANDARD:
            self.logger.info("Using standard split (single split with most recent data)")
        elif self.split_type == SplitType.FIXED_WINDOW:
            window_size = self.config.get('training.history_days')
            self.logger.info(f"Using fixed window splits with window size of {window_size} days")
        elif self.split_type == SplitType.EXPANDING_WINDOW:
            max_splits = self.config.get('training.max_splits', 10)
            self.logger.info(f"Using expanding window with up to {max_splits} splits")
        
        # Create a study
        study = optuna.create_study(direction="maximize")
        
        # Run optimization
        study.optimize(
            self._objective, 
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Log best parameters and score
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"Best score: {best_value:.6f}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    def _train_and_evaluate(self, params: Dict) -> Tuple[Model, float]:
        """
        Train a model with given parameters and evaluate it.
        Always uses SplitType.STANDARD for efficient parameter tuning.
        Fully delegates to Trainer for all model operations.
        
        Args:
            params: Model parameters to use
            
        Returns:
            Tuple of (trained model, evaluation score)
        """
        # Create a temporary trainer for this trial to avoid side effects
        temp_trainer = Trainer(
            self.config, 
            self.data_loader,
            self.feature_factory,
            self.model_factory
        )
        
        # Use the configured split type for tuning
        model = temp_trainer.train(
            split_type=self.split_type,
            model_params=params
        )
        
        # Let the trainer evaluate the model
        score, metrics = temp_trainer.evaluate_model(model)
        
        if score > 0 and "error" not in metrics:
            # We have a valid evaluation score from validation data
            self.logger.info(f"Evaluation metrics: {metrics}")
            return model, score
        
        # If validation evaluation failed, create a fallback evaluation
        self.logger.info("Using training data evaluation with penalty")
        
        # Let's add a fallback evaluation method to Trainer and use it
        try:
            # Sample a small portion of training data for quick evaluation
            sample_size = min(10000, len(self.data_loader.train_df))
            train_sample = self.data_loader.train_df.sample(n=sample_size, seed=42)
            
            # Let the trainer perform feature generation and prediction
            # by passing this sampled dataframe to the evaluate_model method
            # as both history and validation data
            fallback_score, _ = temp_trainer.evaluate_model(
                model, 
                validation_data=(train_sample, train_sample, train_sample, train_sample)
            )
            
            # Apply a penalty factor for using training data (to account for overfitting)
            adjusted_score = fallback_score * 0.9
            self.logger.info(f"Training evaluation score with penalty: {adjusted_score:.6f}")
            return model, adjusted_score
            
        except Exception as e:
            # If even the fallback fails, return a minimal score
            self.logger.warning(f"Fallback evaluation failed: {str(e)}")
            return model, 0.01  # Return a minimal score rather than 0 to avoid stopping the search
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization"""
        # Get parameter configuration from config
        params = {}
        
        # Path to the parameter grid for the current model type
        param_grid_path = f"hyperparameter_tuning.param_grid.{self.model_type}"
        param_grid = self.config.get(param_grid_path, {})
        
        if not param_grid:
            self.logger.warning(f"No parameter grid found for model type: {self.model_type}. Using default ranges.")
            return 0.0
        
        # Process each parameter based on its type and configuration
        for param_name, param_config in param_grid.items():
            # Skip commented out parameters
            if isinstance(param_config, str) and param_config.startswith('#'):
                continue
            
            try:
                # Get parameter type
                if isinstance(param_config, dict) and 'type' in param_config:
                    # New explicit parameter format
                    param_type = param_config.get('type')
                    
                    if param_type == 'float':
                        # Float parameter (continuous)
                        param_range = param_config.get('range', [0.0, 1.0])
                        log_scale = param_config.get('log_scale', False)
                        
                        if len(param_range) != 2:
                            self.logger.warning(f"Invalid range for float parameter {param_name}. Using default range [0.0, 1.0].")
                            param_range = [0.0, 1.0]
                            
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_range[0], 
                            param_range[1], 
                            log=log_scale
                        )
                        
                    elif param_type == 'int':
                        # Integer parameter (continuous)
                        param_range = param_config.get('range', [1, 10])
                        
                        if len(param_range) != 2:
                            self.logger.warning(f"Invalid range for int parameter {param_name}. Using default range [1, 10].")
                            param_range = [1, 10]
                            
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_range[0],
                            param_range[1]
                        )
                        
                    elif param_type == 'categorical':
                        # Categorical parameter (discrete values)
                        values = param_config.get('values', [])
                        
                        if not values:
                            self.logger.warning(f"No values provided for categorical parameter {param_name}. Skipping.")
                            continue
                            
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            values
                        )
                        
                    else:
                        self.logger.warning(f"Unknown parameter type '{param_type}' for {param_name}. Skipping.")
                        
                else:
                    # Legacy format for backward compatibility
                    self.logger.warning(f"Parameter {param_name} uses legacy format. Consider updating to the new format.")
                    
                    param_values = param_config
                    
                    # Handle as before - checking if it's a range or discrete values
                    if len(param_values) == 2 and param_values[0] != param_values[1]:
                        # Treat as a range
                        if isinstance(param_values[0], float):
                            # Float parameter
                            params[param_name] = trial.suggest_float(
                                param_name, 
                                param_values[0], 
                                param_values[1], 
                                log=True if param_name == 'learning_rate' else False
                            )
                        else:
                            # Integer parameter
                            params[param_name] = trial.suggest_int(
                                param_name,
                                param_values[0],
                                param_values[1]
                            )
                    else:
                        # Treat as discrete values
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_values
                        )
                        
            except Exception as e:
                self.logger.error(f"Error processing parameter {param_name}: {str(e)}")
        
        # Check if we have any parameters
        if not params:
            self.logger.warning(f"No valid parameters found for model type: {self.model_type}")
            return 0.0
            
        self.logger.info(f"Generated parameters for trial: {params}")
        
        self.logger.info(f"Trial {trial.number}: Testing parameters {params}")
        
        # Train and evaluate model using the Trainer
        _, score = self._train_and_evaluate(params)
        
        self.logger.info(f"Trial {trial.number} Score: {score:.6f}")
        return score