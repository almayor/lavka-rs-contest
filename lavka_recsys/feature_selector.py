import numpy as np
import polars as pl
from typing import List, Tuple, Dict, Any
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

from .custom_logging import get_logger
from .config import Config
from .feature_factory import FeatureFactory

class FeatureSelector:
    """Feature selection utilities for recommender system"""
    
    def __init__(self, config: Config):
        """Initialize feature selector with configuration"""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        
    def select_features(
            self, 
            train_features: pl.DataFrame, 
            train_target: pl.Series,
            method: str = None,
            threshold: float = None,
            n_features: int = None
        ) -> List[str]:
        """
        Select features based on the specified method.
        
        Args:
            train_features: Training features DataFrame
            train_target: Target values Series
            method (optional, use config if None): Feature selection method ('variance', 'importance', 'correlation').
                1. 'variance' - Select features with variance above the threshold.
                2. 'importance' - Select features based on feature importance from a Random Forest model.
                3. 'correlation' - Select features based on correlation with target and remove highly intercorrelated features.
            threshold (optional, use config if None): Threshold for feature selection
            n_features (optional, use config if None): Number of features to select (if None, use threshold)  
        Returns:
            List of selected feature names
        """
        method = self.config.get('feature_selection.method', method)
        threshold = self.config.get('feature_selection.threshold', threshold)
        n_features = self.config.get('feature_selection.n_features', n_features)

        if method not in ['variance', 'importance', 'correlation']:
            self.logger.error(f"Unknown feature selection method: {method}.")
            raise ValueError(f"Unknown feature selection method: {method}")
        if n_features is not None and n_features <= 0:
            self.logger.error("Number of features to select must be greater than 0.")
            raise ValueError("Number of features to select must be greater than 0.")
        if threshold is not None and threshold < 0:
            self.logger.error("Threshold must be non-negative.")
            raise ValueError("Threshold must be non-negative.")
        
        self.logger.info(f"Selecting features using method: {method}, threshold: {threshold}, n_features: {n_features}")
        
        # Convert to numpy for sklearn
        if isinstance(train_features, pl.DataFrame):
            feature_names = train_features.columns
            train_features_np = train_features.to_numpy()
        else:
            feature_names = train_features.columns
            train_features_np = train_features.values
            
        if isinstance(train_target, pl.Series):
            train_target_np = train_target.to_numpy()
        else:
            train_target_np = train_target.values
            
        # Apply feature selection method
        if method == 'variance':
            selected_indices = self._select_by_variance(train_features_np, threshold)
        elif method == 'importance':
            selected_indices = self._select_by_importance(
                train_features_np, train_target_np, threshold, n_features
            )
        elif method == 'correlation':
            selected_indices = self._select_by_correlation(
                train_features_np, train_target_np, threshold, n_features
            )
        else:
            self.logger.error(f"Unknown feature selection method: {method}.")
            raise ValueError(f"Unknown feature selection method: {method}")
            
        # Get selected feature names
        selected_features = [feature_names[i] for i in selected_indices]
        
        self.logger.info(f"Selected {len(selected_features)} features out of {len(feature_names)}")
        self.logger.info(f"Selected features: {selected_features}")
        
        return selected_features
        
    def _select_by_variance(self, X: np.ndarray, threshold: float) -> List[int]:
        """Select features based on variance threshold"""
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        # Get indices of selected features
        return np.where(selector.get_support())[0].tolist()
        
    def _select_by_importance(
            self, 
            X: np.ndarray, 
            y: np.ndarray, 
            threshold: float = 0.05,
            n_features: int = None
        ) -> List[int]:
        """Select features based on feature importance"""
        # Train a random forest to get feature importance
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(X, y)
        
        if n_features is not None:
            # Select top N features
            selector = SelectFromModel(
                rf, 
                max_features=n_features,
                prefit=True
            )
        else:
            # Select features based on threshold
            selector = SelectFromModel(
                rf, 
                threshold=threshold,
                prefit=True
            )
            
        # Get indices of selected features
        return np.where(selector.get_support())[0].tolist()
        
    def _select_by_correlation(
            self, 
            X: np.ndarray, 
            y: np.ndarray, 
            threshold: float = 0.05,
            n_features: int = None
        ) -> List[int]:
        """
        Select features based on correlation with target 
        and remove highly intercorrelated features
        """
        # Convert to DataFrame for correlation calculation
        X_df = pl.DataFrame(X)
        y_df = pl.Series(y)
        
        # Calculate correlation with target
        corr_with_target = []
        for i in range(X_df.shape[1]):
            corr = abs(X_df.iloc[:, i].corr(y_df, method='spearman'))
            corr_with_target.append((i, corr))
            
        # Sort by correlation
        corr_with_target.sort(key=lambda x: x[1], reverse=True)
        
        if n_features is not None:
            # Select top N features
            selected = [x[0] for x in corr_with_target[:n_features]]
        else:
            # Select features with correlation above threshold
            selected = [x[0] for x in corr_with_target if x[1] >= threshold]
            
        # Remove highly intercorrelated features
        final_selected = self._remove_correlated(X_df, selected)
        
        return final_selected
        
    def _remove_correlated(
            self, 
            X_df: pl.DataFrame, 
            selected_indices: List[int], 
            threshold: float = 0.9
        ) -> List[int]:
        """Remove highly correlated features from the selected set"""
        # Create correlation matrix for selected features
        selected_df = X_df.iloc[:, selected_indices]
        corr_matrix = selected_df.corr().abs()
        
        # Initialize set of indices to drop
        to_drop = set()
        
        # Iterate through correlation matrix to find highly correlated pairs
        for i in range(len(corr_matrix.columns)):
            if i in to_drop:
                continue
                
            # Find features that are highly correlated with this one
            correlated_indices = np.where(corr_matrix.iloc[i, :] > threshold)[0]
            correlated_indices = [idx for idx in correlated_indices if idx != i and idx not in to_drop]
            
            # Add correlated features to drop set
            to_drop.update(correlated_indices)
            
        # Convert original selected indices to mapping
        idx_map = {j: selected_indices[j] for j in range(len(selected_indices))}
        
        # Get final selected indices
        final_selected = [idx_map[j] for j in range(len(selected_indices)) if j not in to_drop]
        
        return final_selected
