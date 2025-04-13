import optuna
import polars as pl
import numpy as np
from typing import Dict, Any
from sklearn.metrics import roc_auc_score

from .custom_logging import get_logger
from .config import Config
from .data_loader import DataLoader
from .feature_factory import FeatureFactory
from .model_factory import ModelFactory

class HyperparameterTuner:
    """Hyperparameter tuning using Optuna"""
    
    def __init__(self, config: Config, data_loader: DataLoader, feature_factory: FeatureFactory, model_factory: ModelFactory):
        """Initialize with configuration and components"""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.data_loader = data_loader
        self.feature_factory = feature_factory
        self.model_factory = model_factory
        
        # Get model type
        self.model_type = config.get("model.type")
        
        # Fixed number of trials and timeout
        self.n_trials = 20
        self.timeout = 3 * 3600  # 3 hour timeout
    
    def tune(self):
        """Run hyperparameter tuning process"""
        self.logger.info(f"Starting hyperparameter tuning with {self.n_trials} trials")
        
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
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization"""
        # Suggest parameters
        if self.model_type == "catboost":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "iterations": trial.suggest_int("iterations", 100, 1000)
            }
        elif self.model_type == "lightgbm":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000)
            }
        else:
            self.logger.warning(f"Unknown model type: {self.model_type}. Using default parameters.")
            return 0.0
        
        self.logger.info(f"Trial {trial.number}: Testing parameters {params}")
        
        # Create a single validation split - simpler than k-fold
        history_df, train_df, val_df = self.data_loader.create_validation_split()
        
        # Get feature names and target name
        feature_sets = self.config.get("features")
        target_name = self.config.get('target')
        
        # Generate features
        train_features, train_target, cat_columns, _ = self.feature_factory.generate_batch(
            history_df, train_df, feature_sets, target_name
        )

        val_history = pl.concat([history_df, train_df], how='vertical')
        val_features, val_target, _, val_request_ids = self.feature_factory.generate_batch(
            val_history, val_df, feature_sets, target_name
        )
        
        # Create and train model with suggested parameters
        model = self.model_factory.create_model(params)
        model.train(
            train_features, 
            train_target,
            cat_columns=cat_columns,
            eval_set=(val_features, val_target)
        )
        
        # Predict on validation set
        val_preds = model.predict(val_features)
        val_preds = pl.Series(val_preds)
        
        # Calculate AUC score (simpler than NDCG)
        score = roc_auc_score(val_target, val_preds)
        self.logger.info(f"Trial {trial.number} AUC: {score:.6f}")
        
        return score
