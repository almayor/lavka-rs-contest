import os
import json
import hashlib
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss

from .utils.config import Config
from .utils.custom_logging import get_logger
from .data_loader import DataLoader
from .models import ModelFactory, Model
from .feature_factory import CachedFeatureFactory
from .utils.metrics import RankingMetrics


def compute_hash(config_obj: Any, length: int = 6) -> str:
    """
    Generate a deterministic short MD5 hash from a configuration object.
    """
    data = config_obj.to_dict() if hasattr(config_obj, 'to_dict') else config_obj
    raw = json.dumps(data, sort_keys=True).encode('utf-8')
    return hashlib.md5(raw).hexdigest()[:length]


class Experiment:
    """
    Coordinates data loading, feature engineering, model training, evaluation,
    and optional Kaggle-style simulation and submission workflows.
    """

    def __init__(self, base_name: str, config: Config):
        self.base_name = base_name
        self.config = config
        self.name = f"{base_name}_{compute_hash(config)}"
        self.logger = get_logger(f"Experiment({self.name})")
        self.logger.info(f"Initialized experiment: {self.name}")

        # Components
        self.data_loader = DataLoader(config)
        self.feature_factory = CachedFeatureFactory(config)
        self.model_factory = ModelFactory(config)

        # Outputs
        self.results: Dict[str, Any] = {}
        self.last_model: Optional[Model] = None

        self.results_dir = config.get('output.results_dir', 'results')
        self._ensure_directories()
        self._persist_config()

    def setup(self) -> None:
        """
        Load data and setup components. Must be called before other methods.
        """
        self.logger.info("Setting up experiment environment...")
        self.data_loader.setup()
        #Initialize more components if needed
        self.logger.info("Setup complete.")

    def run(self) -> Dict[str, Any]:
        """
        Execute training (and tuning if configured), then collect and save run results.
        """
        self.logger.info("Starting experiment run...")
        train_history, train_target, val_history, val_target = self.data_loader.train_split()
        self.last_model = self._train_model((train_history, train_target), (val_history, val_target))
        
        if self.config.get('data.holdout.enabled', False):
            self.logger.info("Evaluating on holdout.")
            holdout_history, holdout_target = self.data_loader.holdout_split()
            metrics = self._evaluate_model(self.last_model, (holdout_history, holdout_target))
        else:
            self.logger.info("Evaluating on validation.")
            metrics = self._evaluate_model(self.last_model, (val_history, val_target))

        self.results = self._consolidate_metrics(self.last_model, metrics)
        self._save_results(self.results)
        self.logger.info("Run finished and results saved.")
        return self.results
    
    def create_submission(self):
        """
        Generate a Kaggle-style submission file from the trained model.
        """
        self.logger.info("Computing predictions...")
        history, target = self.data_loader.final_split()
        self.last_model = self._train_model((history, target))
        sub_history, sub_target = self.data_loader.submission_split()
        scores = self._make_predictions(self.last_model, (sub_history, sub_target))
        submission = (
            sub_target.select(['index', 'request_id'])
                   .with_columns(pl.Series('score', scores))
                   .sort('score', descending=True)
                   .select(['index', 'request_id'])
        )
        self._save_submission_files(self.last_model, submission)
        return submission

    # --- Internal Helpers ---

    def _train_model(
        self,
        train_splits: tuple[pl.DataFrame, pl.DataFrame],
        val_splits: Optional[tuple[pl.DataFrame, pl.DataFrame]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Model:
        """
        Train a model using the specified split and parameters.
        Returns the trained model.
        """

        X_train, y_train, cat_cols, train_group_ids = \
            self.feature_factory.generate_batch(*train_splits)
        if val_splits:
            X_val, y_val, _, val_group_ids = \
                self.feature_factory.generate_batch(*val_splits)
            #Reorder columns for consistency
            X_val = X_val.select(X_train.columns)
        else:
            X_val, y_val, _, val_group_ids = None, None, None, None
        
        model = self.model_factory.create_model(model_params)
        start = time.time()
        model.train(
            train_features=X_train,
            train_labels=y_train,
            train_group_ids=train_group_ids,
            val_features=X_val,
            val_labels=y_val,
            val_group_ids=val_group_ids,
            cat_columns=cat_cols
        )
        self.logger.info(f"Training completed in {time.time() - start:.2f}s")
        return model

    def _evaluate_model(
        self,
        model: Model,
        eval_splits: tuple[pl.DataFrame, pl.DataFrame],
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on a validation or test set.
        Returns (primary_score, metrics_dict).
        """
        X_eval, y_eval, _, eval_group_ids = \
            self.feature_factory.generate_batch(*eval_splits)
        preds = model.predict(X_eval, group_ids=eval_group_ids)
        
        preds_arr = np.asarray(preds)
        true_arr = np.asarray(y_eval)

        # Converting `true_arr` to a binary metric
        if not np.isin(true_arr, [0, 1]).all():
            self.logger.warning("Binarizing target for metric calculation")
            true_arr = np.where(true_arr == true_arr.max(), 1, 0)
    
        return self.calculate_metrics(true_arr, preds_arr, eval_group_ids)

    def _consolidate_metrics(
        self,
        model: Model,
        metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Extract feature importance, compile run metadata, and add it to existing metrics.
        """
        importance = model.get_feature_importance() or {}

        result = {
            'experiment_name': self.name,
            'base_name': self.base_name,
            'metrics': metrics,
            'feature_importance': {k: float(v) for k, v in importance.items()},
            'timestamp': datetime.now().isoformat(),
        }
        return result

    def _make_predictions(
            self,
            model: Model,
            splits: tuple[pl.DataFrame, pl.DataFrame],
        ) -> pl.Series:
        """
        Predict
        """
        submission_features, cat_cols, group_ids = \
            self.feature_factory.generate_features_only(*splits)
        return model.predict(
            submission_features,
            cat_columns=cat_cols,
            group_ids=group_ids
        )

    def _ensure_directories(self) -> None:
        """
        Create output directories defined in config.
        """
        dirs = [
            self.results_dir,
            self.config.get('output.model_cache_dir', 'results/model_cache'),
            self.config.get('output.feature_cache_dir', 'results/feature_cache'),
            self.config.get('output.submissions_dir', 'results/submissions'),
            self.config.get('output.simulations_dir', 'results/simulations')
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def _persist_config(self) -> None:
        path = os.path.join(self.results_dir, f"{self.name}_config.json")
        self.config.save(path)
        self.logger.info(f"Config saved: {path}")

    def _save_results(self, results: Dict[str, Any]) -> None:
        path = os.path.join(self.results_dir, f"{self.name}_results.json")
        with open(path, 'w') as fp:
            json.dump(results, fp, indent=2)
        self.logger.info(f"Run results saved: {path}")

    def _save_submission_files(
        self,
        model: Model,
        submission_df: pl.DataFrame
    ) -> None:
        sub_dir = self.config.get('output.submissions_dir', os.path.join(self.results_dir, 'submissions'))
        os.makedirs(sub_dir, exist_ok=True)
        submission_path = os.path.join(sub_dir, f"{self.name}_submission.csv")
        submission_df.write_csv(submission_path)
        self.logger.info(f"Submission saved: {submission_path}")
        model_path = os.path.join(sub_dir, f"{self.name}_model.pkl")
        model.save(model_path)
        self.logger.info(f"Submission model saved: {model_path}")

    @staticmethod
    def calculate_metrics(
        true: np.ndarray,
        preds: np.ndarray,
        group_ids: Optional[np.ndarray] = None,
        k: int = 10
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics: AUC, log loss, and optional nDCG@k.
        """
        results: Dict[str, float] = {}
        # AUC
        results['auc'] = roc_auc_score(true, preds)
        # Log loss (clip preds to avoid edge cases)
        eps = 1e-15
        clipped = np.clip(preds, eps, 1 - eps)
        results['logloss'] = log_loss(true, clipped)
        # nDCG@k if group_ids provided
        if group_ids is not None:
            results[f'ndcg@{k}'] = RankingMetrics.ndcg_at_k(true, preds, group_ids, k)
        return results
