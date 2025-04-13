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

        self.trained = False
        self.cat_columns = None # Placeholder for categorical columns
        self.selected_features = None # Placeholder for selected features

    def __call__(self, features: pl.DataFrame) -> pl.DataFrame:
        """Call method to select features"""
        if not self.trained:
            self.logger.error("FeatureSelector not trained. Call train() first.")
            raise RuntimeError("FeatureSelector not trained. Call train() first.")

        self.logger.debug("Applying feature selection")
        if self.cat_columns is not None:
            return features.select(self.cat_columns + self.selected_features)
        else:
            return features.select(self.selected_features)

    def train(
            self, 
            train_features: pl.DataFrame, 
            train_target: pl.Series,
            cat_columns: List[str] = None,
            method: str = None,
            threshold: float = None,
            n_features: int = None
        ) -> List[str]:
        """
        Select features based on the specified method.
        
        Args:
            train_features: Training features DataFrame
            train_target: Target values Series
            cat_columns: List of categorical columns (optional)
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
                
        
        feature_names = train_features.columns
        if cat_columns is not None:
            self.cat_columns = cat_columns
            train_features = train_features.drop(cat_columns)
            feature_names = [col for col in feature_names if col not in cat_columns]
        
        # Convert to numpy for sklearn
        train_features_np = train_features.to_numpy()
        train_target_np = train_target.to_numpy()

        # Apply feature selection method
        self.logger.info(f"Selecting features using method: {method}, threshold: {threshold}, n_features: {n_features}")
        if method == 'variance':
            selected_indices = self._select_by_variance(train_features_np, feature_names, threshold)
        elif method == 'importance':
            selected_indices = self._select_by_importance(
                train_features_np, train_target_np, feature_names, threshold, n_features
            )
        elif method == 'correlation':
            selected_indices = self._select_by_correlation(
                train_features_np, train_target_np, feature_names, threshold, n_features
            )
        else:
            self.logger.error(f"Unknown feature selection method: {method}.")
            raise ValueError(f"Unknown feature selection method: {method}")
            
        # Get selected feature names
        selected_features = [feature_names[i] for i in selected_indices]
        
        self.logger.info(f"Selected {len(selected_features)} features out of {len(feature_names)}")
        self.logger.info(f"Removed {len(feature_names) - len(selected_features)} features")
        self.logger.info(f"Removed features: {set(feature_names) - set(selected_features)}")
        self.logger.info(f"Selected features: {selected_features}")
        
        self.trained = True
        self.selected_features = selected_features
        self.cat_columns = cat_columns

        return selected_features
        
    def _select_by_variance(
            self,
            X: np.ndarray,
            feature_names: List[str],
            threshold: float
        ) -> List[int]:
        """Select features based on variance threshold"""
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        # Get indices of selected features
        return np.where(selector.get_support())[0].tolist()
        
    def _select_by_importance(
            self, 
            X: np.ndarray, 
            y: np.ndarray,
            feature_names: List[str], 
            threshold: float = 0.05,
            n_features: int = None
        ) -> List[int]:
        """Select features based on feature importance"""
        # Train a random forest to get feature importance
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            verbose=True
        )
        rf.fit(X, y)
        self.logger.info(f"Feature importances: {rf.feature_importances_}")
        
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
            feature_names: List[str],
            threshold: float = 0.05,
            n_features: int = None
        ) -> List[int]:
        """
        Select features based on correlation with target 
        and remove highly intercorrelated features
        """
        # Convert to DataFrame for further processing
        X_df = pl.DataFrame(X, schema=feature_names)
        y_series = pl.Series("target", y)
        
        # Calculate correlation with target
        corr_with_target = []
        for i, feature in enumerate(feature_names):
            # Compute correlation between each feature and the target using numpy
            feature_values = X_df.select(pl.col(feature)).to_numpy().flatten()
            target_values = y_series.to_numpy()
            # Use NumPy's corrcoef function (Pearson correlation)
            # For Spearman, we would need scipy, but this will work for now
            corr = abs(np.corrcoef(feature_values, target_values)[0, 1])
            # Handle NaN values that might occur
            if np.isnan(corr):
                corr = 0.0
            corr_with_target.append((i, corr))
            
        # Sort by correlation
        corr_with_target.sort(key=lambda x: x[1], reverse=True)
        
        if n_features is not None:
            # Select top N features
            selected = [x[0] for x in corr_with_target[:n_features]]
        else:
            # Select features with correlation above threshold
            selected = [x[0] for x in corr_with_target if x[1] >= threshold]

        self.logger.info(f"Selected {len(selected)} features based on correlation with target")
        
        # Handle case where no features meet the threshold
        if not selected:
            self.logger.warning(f"No features met the correlation threshold of {threshold}. Using top feature instead.")
            # Use the top correlated feature as fallback
            selected = [corr_with_target[0][0]] if corr_with_target else []
            
        if selected:
            self.logger.info(f"Selected features: {[feature_names[idx] for idx in selected]}")
            # Remove highly intercorrelated features
            final_selected = self._remove_correlated(X_df, selected, feature_names, threshold=0.9)
            return final_selected
        else:
            self.logger.warning("No features available for selection.")
            return []
        
    def _remove_correlated(
            self, 
            X_df: pl.DataFrame, 
            selected_indices: List[int],
            feature_names: List[str],
            threshold: float = 0.9
        ) -> List[int]:
        """Remove highly correlated features from the selected set"""
        # Handle case where there's only one or zero selected features
        if len(selected_indices) <= 1:
            return selected_indices
            
        # Create correlation matrix for selected features
        selected_feature_names = [feature_names[idx] for idx in selected_indices]
        
        try:
            selected_df = X_df.select(selected_feature_names)
            
            # Calculate correlation matrix using numpy
            selected_np = selected_df.to_numpy()
            
            # Check if array is empty
            if selected_np.size == 0:
                self.logger.warning("Empty array encountered during correlation calculation.")
                return selected_indices
                
            corr_matrix = np.corrcoef(selected_np.T)
            
            # Handle single feature case (corrcoef returns scalar for 1 feature)
            if not isinstance(corr_matrix, np.ndarray) or corr_matrix.ndim < 2:
                return selected_indices
                
            # Initialize set of indices to drop
            to_drop = set()
            
            # Iterate through correlation matrix to find highly correlated pairs
            for i in range(len(selected_indices)):
                if i in to_drop:
                    continue
                    
                # Find features that are highly correlated with this one
                for j in range(i + 1, len(selected_indices)):
                    if j in to_drop:
                        continue
                        
                    if abs(corr_matrix[i, j]) > threshold:
                        # Keep the one with higher index in selected_indices 
                        # (which corresponds to higher correlation with target)
                        to_drop.add(j)
                        
                        correlated_feature = feature_names[selected_indices[j]]
                        correlating_feature = feature_names[selected_indices[i]]
                        self.logger.info(f"Feature {correlating_feature} is highly correlated with: {correlated_feature}")
                
            # Get final selected indices
            final_selected = [selected_indices[j] for j in range(len(selected_indices)) if j not in to_drop]
            self.logger.info(f"Removed {len(to_drop)} highly correlated features")
            
            return final_selected
            
        except Exception as e:
            self.logger.error(f"Error in remove_correlated: {str(e)}")
            # In case of error, return the original selection
            return selected_indices
