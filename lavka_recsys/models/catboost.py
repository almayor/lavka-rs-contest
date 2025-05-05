import numpy as np
import pandas as pd
import polars as pl
import pickle

from typing import Iterable

from .base import Model

class CatBoostModel(Model):
    """General CatBoost model implementation"""

    def _to_pandas(self, df):
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        return df

    def _prepare_pool(self,
        features: pd.DataFrame | pl.DataFrame,
        labels: Iterable[str] = None,
        *,
        group_ids = None,
        cat_columns = None
        ):
        """Train CatBoost model.
        Args:
            features (pd.DataFrame or pl.DataFrame): Features
            labels (pd.Series or pl.Series): Labels (optional).
            group_ids (pd.Series or pl.Series): Group IDs (optional).
            cat_columns (list): Categorical column names.
        """
        from catboost import Pool

        features = self._to_pandas(features)
        labels_list = None
        if labels is not None:
            labels_list = labels.to_list() if isinstance(labels, pl.Series) else list(labels)

        cat_columns = cat_columns or []
        for col in cat_columns:
            self.logger.info(f"Converting '{col}' to string for CatBoost")
            features[col] = features[col].astype(str)
        
        if group_ids is not None:
            group_ids = [str(id) for id in group_ids]

        return Pool(
            data=features,
            label=labels_list,
            cat_features=cat_columns,
            group_id=group_ids
        )
    
    def _sort_by_group_id(self, features, group_ids, labels=None):
        """Sort features and labels by group IDs
        Returns
            - features: sorted by group_ids
            - labels: sorted by group_ids
            - group_ids: sorted
            - original_index
        """
        if isinstance(features, pl.DataFrame):
            features = features.to_pandas()
        if isinstance(labels, pl.Series):
            labels = labels.to_list()
        
        original_index = features.index.copy()

        # Sort by group id if available to ensure CatBoost's requirement that queryIds should be grouped
        self.logger.info("Sorting data by group_id for grouped ranking")
        sort_order = np.argsort(group_ids)
        features = features.iloc[sort_order]
        group_ids = group_ids[sort_order]
        if labels is not None:
            labels = pd.Series(labels, index=features.index).loc[features.index].reset_index(drop=True)

        # Convert group_ids to string for CatBoost
        group_ids = pd.Series(group_ids, index=features.index).astype(str)
        
        return features, labels, group_ids, original_index

    def _extract_group_ids(self, features):
        """Extract group IDs from features and labels, if necessary, then, sort the data"""
        if isinstance(features, pl.DataFrame):
            features = features.to_pandas()

        if 'group_id' in features.columns:
            group_id_col = 'group_id'
        elif 'request_id' in features.columns:
            group_id_col = 'request_id'
        else:
            self.logger.error('Missing either `group_id` or `request_id` in features dataframe')
            raise ValueError('Missing either `group_id` or `request_id` in features dataframe')
        
        group_ids = features[group_id_col].astype(str).values    
        features = features.drop(columns=[group_id_col])
        return features, group_ids


class CatBoostClassifierModel(CatBoostModel):
    """CatBoostClassifier"""
    
    def __init__(self, **params):
        super().__init__('catboost_classifier', params)
        from catboost import CatBoostClassifier
        self.params = params
        self.model = CatBoostClassifier(**self.params)
    
    def train(self,
        train_features: pd.DataFrame | pl.DataFrame,
        train_labels: Iterable[str],
        *,
        train_group_ids=None,
        val_features=None,
        val_labels=None,
        val_group_ids=None,
        cat_columns=None,
        **kwargs
    ):
        """Train CatBoost model.
        Args:
            train_features (pd.DataFrame or pl.DataFrame): Training features.
            train_labels (pd.Series or pl.Series): Training labels.
            train_group_ids (pd.Series or pl.Series): Request IDs for training data.
            cat_columns (list): Categorical column names.
        """
        if train_group_ids is not None:
            train_features, train_labels, train_group_ids, _ = self._sort_by_group_id(
                train_features, train_group_ids, labels=train_labels
            )
        if val_group_ids is not None:
            val_features, val_labels, val_group_ids, _ = self._sort_by_group_id(
                val_features, val_group_ids, labels=val_labels
            )
        train_pool = self._prepare_pool(train_features, labels=train_labels, group_ids=train_group_ids, cat_columns=cat_columns)
        val_pool = None
        if val_features is not None and val_labels is not None:
            val_pool = self._prepare_pool(val_features, val_labels, group_ids=val_group_ids, cat_columns=cat_columns)
        
        self.logger.info(
            f"Training CatBoostClassifier model with columns: {train_features.columns.tolist()} "
            f"(cat_columns: {cat_columns})"
        )
        self.model.fit(train_pool, eval_set=val_pool, verbose=False, plot=True)
        return self
    
    def predict(self, features: pd.DataFrame | pl.DataFrame, **kwargs):
        """Make probability predictions with CatBoost"""
        # Make a copy of the dataframe to avoid modifying the original
        feature_names = self.model.feature_names_

        # Ensure we have the exact same columns the model was trained on
        missing = set(self.model.feature_names_) - set(features.columns)
        if missing:
            raise ValueError(f"Missing columns in prediction data: {missing}")
        
        cat_idxs = self.model.get_cat_feature_indices()
        cat_cols = [self.model.feature_names_[i] for i in cat_idxs] if cat_idxs else None
        pool = self._prepare_pool(features, cat_columns=cat_cols)
        return self.model.predict_proba(pool)[:, 1]
    
    def get_feature_importance(self):
        names = self.model.feature_names_
        imps = self.model.feature_importances_
        imps = np.nan_to_num(imps).tolist()
        return dict(zip(names, imps))
    
    def save(self, filename):
        self.model.save_model(filename)
    
    @classmethod
    def load(cls, filename):
        from catboost import CatBoostClassifier
        model = cls()
        model.model = CatBoostClassifier()
        model.model.load_model(filename)
        return model


class CatBoostRankerModel(CatBoostModel):
    """CatBoost model implementation for ranking tasks"""
    
    def __init__(self, **params):
        super().__init__('catboost_ranker', params)
        from catboost import CatBoostRanker
        self.params = params
        self.model = CatBoostRanker(**self.params)

        # Initialize empty feature importances that will be filled during training
        self._feature_importances = {}
    
    def train(self,
        train_features: pd.DataFrame | pl.DataFrame,
        train_labels: Iterable[str],
        *,
        train_group_ids = None,
        val_features = None,
        val_labels = None,
        val_group_ids = None,
        cat_columns = None,
        **kwargs
    ):
        """Train CatBoost ranker model.
        Args:
            train_features (pd.DataFrame or pl.DataFrame): Training features.
            train_labels (pd.Series or pl.Series): Training labels.
            train_group_ids (pd.Series or pl.Series): Request IDs for training data.
            val_features (pd.DataFrame or pl.DataFrame): Evaluation features.
            val_labels (pd.Series or pl.Series): Evaluation labels.
            val_group_ids (pd.Series or pl.Series): Request IDs for evaluation data.
            cat_columns (list): Categorical column names.
        """
        # Extract groups if not provided
        if train_group_ids is None:
            train_features, train_group_ids = self._extract_group_ids(train_features)
        if val_group_ids is None:
            val_features, val_group_ids = self._extract_group_ids(val_features)
        
        # Sort features and group_ids by group_id
        train_features, train_labels, train_group_ids, _ = self._sort_by_group_id(
            train_features, train_group_ids, labels=train_labels
        )
        val_features, val_labels, val_group_ids, _ = self._sort_by_group_id(
            val_features, val_group_ids, labels=val_labels
        )
        
        train_pool = self._prepare_pool(
            train_features,
            labels=train_labels,
            group_ids=train_group_ids,
            cat_columns=cat_columns,
        )
        val_pool = None if val_features is None else self._prepare_pool(
            val_features,
            labels=val_labels,
            group_ids=val_group_ids,
            cat_columns=cat_columns,
        )
        self.logger.info(
            f"Training CatBoostRanker model with columns: {train_features.columns.tolist()} "
            f"(cat_columns: {cat_columns})"
        )
        self.model.fit(train_pool, eval_set=val_pool, plot=True, verbose=False)

        # Calculate and save feature importance during training, since we have the train_pool available
        self._compute_feature_importances(train_pool)
        return self
    
    def predict(self, features, group_ids=None, **kwargs):
        """Make relevance predictions with CatBoost ranker
        
        Args:
            features (pd.DataFrame or pl.DataFrame): Features for prediction
            group_ids (pd.Series or pl.Series): Group IDs for ranking
            **kwargs: Additional arguments
            
        Returns:
            Prediction scores
        """   
        features = self._to_pandas(features)
    
        if group_ids is None:
           features, group_ids = self._extract_group_ids(features)
        
        # Sort features and group_ids by group_id
        features, _, group_ids, original_index = self._sort_by_group_id(
            features, group_ids, labels=None
        )
 
        # Ensure we have the exact same columns the model was trained on
        missing = set(self.model.feature_names_) - set(features.columns)
        if missing:
            raise ValueError(f"Missing columns in prediction data: {missing}")
        
        cat_idxs = self.model.get_cat_feature_indices()
        cat_cols = [self.model.feature_names_[i] for i in cat_idxs] if cat_idxs else None
        pool = self._prepare_pool(features, group_ids=group_ids, cat_columns=cat_cols)

        preds = self.model.predict(pool)
        series = pd.Series(preds, index=features.index)
        return series.reindex(original_index).values
    
    def get_feature_importance(self):
        return self._feature_importances
    
    def _compute_feature_importances(self, train_pool):
        self._feature_importances = {}
        for typo in ['PredictionValuesChange', 'LossFunctionChange']:
            try:
                imps = self.model.get_feature_importance(train_pool, type=typo)
                self._feature_importances = dict(zip(self.model.feature_names_, imps))
                break
            except Exception as e:
                self.logger.warning(f'Failed to calculate feature importances using {typo}: {str(e)}')
                continue
    
    def save(self, filename):
        """Save model to file"""
        # For saving via pickle (used by model cache), the _feature_importances attribute 
        # will be automatically included in the pickled data
        
        # For direct save to file, save both model and metadata
        self.model.save_model(filename)
        if self._feature_importances:
            with open(f"{filename}.meta", 'wb') as f:
                pickle.dump({'feature_importances': self._feature_importances}, f)
    
    @classmethod
    def load(cls, filename):
        from catboost import CatBoostRanker
        import os

        obj = cls()
        obj.model = CatBoostRanker()
        obj.model.load_model(filename)
        meta = f"{filename}.meta"
        if os.path.exists(meta):
            with open(meta, 'rb') as f:
                data = pickle.load(f)
                obj._feature_importances = data.get('feature_importances', {})
        return obj