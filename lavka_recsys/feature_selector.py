import numpy as np
import polars as pl
import os
import hashlib
import json
import pickle
from typing import List, Tuple, Dict, Any, Optional
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

        self.method = self.config.get('feature_selection.method')
        self.threshold = self.config.get('feature_selection.threshold')
        self.n_features = self.config.get('feature_selection.n_features')

        if self.method not in ['variance', 'importance', 'correlation']:
            self.logger.error(f"Unknown feature selection method: {self.method}.")
            raise ValueError(f"Unknown feature selection method: {self.method}")
        if self.n_features is not None and self.n_features <= 0:
            self.logger.error("Number of features to select must be greater than 0.")
            raise ValueError("Number of features to select must be greater than 0.")
        if self.threshold is not None and self.threshold < 0:
            self.logger.error("Threshold must be non-negative.")
            raise ValueError("Threshold must be non-negative.")
        if self.threshold is None and self.n_features is None:
            self.logger.error("Either threshold or n_features must be specified.")
            raise ValueError("Either threshold or n_features must be specified.")

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
            use_cache: bool = True
        ) -> List[str]:
        """
        Select features based on the specified method.
        
        Args:
            train_features: Training features DataFrame
            train_target: Target values Series
            cat_columns: List of categorical columns (optional)
            use_cache (bool): Whether to use cache. Default is True.
        Returns:
            List of selected feature names
        """
        
        column_names = train_features.columns
        
        # Process feature selection (cache checking is now handled by Experiment)
        if cat_columns is not None:
            self.cat_columns = cat_columns
            train_features = train_features.drop(cat_columns)
            column_names = [col for col in column_names if col not in cat_columns]
        
        # Convert to numpy for sklearn
        train_features_np = train_features.to_numpy()
        train_target_np = train_target.to_numpy()

        # Apply feature selection method
        self.logger.info(f"Selecting features using method: {self.method}, threshold: {self.threshold}, n_features: {self.n_features}")
        if self.method == 'variance':
            selected_indices = self._select_by_variance(train_features_np, column_names)
        elif self.method == 'importance':
            selected_indices = self._select_by_importance(
                train_features_np, train_target_np, column_names
            )
        elif self.method == 'correlation':
            selected_indices = self._select_by_correlation(
                train_features_np, train_target_np, column_names
            )
        else:
            self.logger.error(f"Unknown feature selection method: {self.method}.")
            raise ValueError(f"Unknown feature selection method: {self.method}")
            
        # Get selected feature names
        selected_features = [column_names[i] for i in selected_indices]
        
        self.logger.info(f"Selected {len(selected_features)} features out of {len(column_names)}")
        self.logger.info(f"Removed {len(column_names) - len(selected_features)} features")
        self.logger.info(f"Removed features: {set(column_names) - set(selected_features)}")
        self.logger.info(f"Selected features: {selected_features}")
        
        self.trained = True
        self.selected_features = selected_features
        self.cat_columns = cat_columns
        
        # Save to cache if enabled
        if use_cache:
            self.save_to_cache(column_names=column_names, cat_columns=cat_columns)

        return selected_features
        
    def _select_by_variance(
            self,
            X: np.ndarray,
            column_names: List[str],
        ) -> List[int]:
        """Select features based on variance threshold"""
        selector = VarianceThreshold(threshold=self.threshold)
        selector.fit(X)
        
        # Get indices of selected features
        return np.where(selector.get_support())[0].tolist()
        
    def _select_by_importance(
            self, 
            X: np.ndarray, 
            y: np.ndarray,
            column_names: List[str], 
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
        
        if self.n_features is not None:
            # Select top N features
            selector = SelectFromModel(
                rf, 
                max_features=self.n_features,
                prefit=True
            )
        else:
            # Select features based on threshold
            selector = SelectFromModel(
                rf, 
                threshold=self.threshold,
                prefit=True
            )
            
        # Get indices of selected features
        return np.where(selector.get_support())[0].tolist()
        
    def _select_by_correlation(
            self, 
            X: np.ndarray, 
            y: np.ndarray,
            column_names: List[str],
        ) -> List[int]:
        """
        Select features based on correlation with target 
        and remove highly intercorrelated features
        """
        # Convert to DataFrame for further processing
        X_df = pl.DataFrame(X, schema=column_names)
        y_series = pl.Series("target", y)
        
        # Calculate correlation with target
        corr_with_target = []
        for i, feature in enumerate(column_names):
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
        
        if self.n_features is not None:
            # Select top N features
            selected = [x[0] for x in corr_with_target[:self.n_features]]
        else:
            # Select features with correlation above threshold
            selected = [x[0] for x in corr_with_target if x[1] >= self.threshold]

        self.logger.info(f"Selected {len(selected)} features based on correlation with target")
        
        # Handle case where no features meet the threshold
        if not selected:
            self.logger.warning(f"No features met the correlation threshold of {self.threshold}. Using top feature instead.")
            # Use the top correlated feature as fallback
            selected = [corr_with_target[0][0]] if corr_with_target else []
            
        if selected:
            self.logger.info(f"Selected features: {[column_names[idx] for idx in selected]}")
            # Remove highly intercorrelated features
            final_selected = self._remove_correlated(X_df, selected, column_names, threshold=0.9)
            return final_selected
        else:
            self.logger.warning("No features available for selection.")
            return []
        
    def _remove_correlated(
            self, 
            X_df: pl.DataFrame, 
            selected_indices: List[int],
            column_names: List[str],
            threshold: float = 0.9
        ) -> List[int]:
        """Remove highly correlated features from the selected set"""
        # Handle case where there's only one or zero selected features
        if len(selected_indices) <= 1:
            return selected_indices
            
        # Create correlation matrix for selected features
        selected_feature_names = [column_names[idx] for idx in selected_indices]
        
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
                        
                        correlated_feature = column_names[selected_indices[j]]
                        correlating_feature = column_names[selected_indices[i]]
                        self.logger.info(f"Feature {correlating_feature} is highly correlated with: {correlated_feature}")
                
            # Get final selected indices
            final_selected = [selected_indices[j] for j in range(len(selected_indices)) if j not in to_drop]
            self.logger.info(f"Removed {len(to_drop)} highly correlated features")
            
            return final_selected
            
        except Exception as e:
            self.logger.error(f"Error in remove_correlated: {str(e)}")
            # In case of error, return the original selection
            return selected_indices
    
    def generate_cache_key(
        self,
        column_names: List[str] = None,
        cat_columns: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a cache key based on requested feature names from config and parameters
        
        Args:
            column_names: List of column names in the generated features (not used for key generation)
            cat_columns: List of categorical column names (not used for key generation)
        """
        # Get requested features from config - this is what determines the cache key
        requested_features = self.config.get("features", [])
        sorted_requested_features = sorted(requested_features)
        
        # Create a dictionary with only the parameters that affect feature selection
        cache_params = {
            "requested_features": sorted_requested_features,  # Use requested feature names from config
            "method": self.method,
            "threshold": self.threshold,
            "n_features": self.n_features
        }
        
        # Convert to JSON string
        params_str = json.dumps(cache_params, sort_keys=True)
        
        # Create hash
        hash_obj = hashlib.md5(params_str.encode())
        hash_str = hash_obj.hexdigest()
        
        # Print detailed debug info
        self.logger.info(f"Generated cache key: {hash_str}")
        self.logger.info(f"Based on requested features: {sorted_requested_features}")
        self.logger.info(f"Feature selection method: {self.method}, threshold: {self.threshold}, n_features: {self.n_features}")
        
        return hash_str

    def get_cache_path(self, cache_key: str) -> str:
        """Get the full path for a cache file"""
        results_dir = self.config.get("output.results_dir", "results")
        
        # Make sure the directory exists
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            
        cache_path = os.path.join(results_dir, f"feature_selection_{cache_key}.pkl")
        self.logger.debug(f"Cache path: {cache_path}")
        return cache_path

    def save_to_cache(
        self,
        column_names: List[str],
        cat_columns: Optional[List[str]]
    ):
        """Save the feature selection results to cache"""
        # Get the actual features requested from config (not columns generated)
        requested_features = self.config.get("features", [])
        
        cache_data = {
            "selected_features": self.selected_features,
            "cat_columns": cat_columns,
            "trained": True,
            # Save all parameters to verify cache
            "method": self.method,
            "threshold": self.threshold,
            "n_features": self.n_features,
            "requested_features": requested_features
        }
        
        # Generate cache key based on config only
        cache_key = self.generate_cache_key()
        cache_path = self.get_cache_path(cache_key)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        self.logger.info(f"Saved feature selection results to cache: {cache_path}")
        self.logger.info(f"Contains {len(self.selected_features)} selected features")

    def load_from_cache(self) -> bool:
        """
        Load feature selection results directly from cache based on config parameters.
        This doesn't require any training data.
        
        Returns:
            bool: True if successfully loaded from cache, False otherwise
        """
        cache_key = self.generate_cache_key()
        cache_path = self.get_cache_path(cache_key)
        
        self.logger.info(f"Checking for cache at: {cache_path}")
        
        if not os.path.exists(cache_path):
            self.logger.info(f"No cache file found at: {cache_path}")
            return False
        
        try:
            self.logger.info(f"Found cache file, attempting to load: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Get requested features from config
            requested_features = self.config.get("features", [])
            
            # Log the comparison details
            self.logger.info(f"Cache vs Current settings:")
            self.logger.info(f"  Method: {cache_data['method']} vs {self.method}")
            self.logger.info(f"  Threshold: {cache_data['threshold']} vs {self.threshold}")
            self.logger.info(f"  N features: {cache_data['n_features']} vs {self.n_features}")
            
            # Check if feature sets match
            cache_features = set(cache_data.get("requested_features", []))
            current_features = set(requested_features)
            features_match = cache_features == current_features
            
            if not features_match:
                self.logger.info(f"Feature sets don't match:")
                self.logger.info(f"  Cache features: {sorted(cache_features)}")
                self.logger.info(f"  Current features: {sorted(current_features)}")
                self.logger.info(f"  Missing: {sorted(current_features - cache_features)}")
                self.logger.info(f"  Extra: {sorted(cache_features - current_features)}")
            
            # Verify parameters match to prevent collisions
            if (cache_data["method"] == self.method and
                cache_data["threshold"] == self.threshold and
                cache_data["n_features"] == self.n_features and
                features_match):

                self.logger.info(f"All parameters match! Loading feature selection from cache")
                self.selected_features = cache_data["selected_features"]
                self.cat_columns = cache_data["cat_columns"]
                self.trained = cache_data["trained"]
                self.logger.info(f"Using cached feature selection result with {len(self.selected_features)} features")
                return True
            else:
                self.logger.warning(f"Cache parameters mismatch, not using cache")
                return False
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {str(e)}")
            import traceback
            self.logger.warning(traceback.format_exc())
            return False
