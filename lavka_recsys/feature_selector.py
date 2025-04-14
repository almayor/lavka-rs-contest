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
            "requested_features": sorted_requested_features,
            "method": self.method,
            "threshold": self.threshold,
            "n_features": self.n_features
        }
        
        # Convert to JSON string and create hash
        try:
            params_str = json.dumps(cache_params, sort_keys=True)
            hash_obj = hashlib.md5(params_str.encode())
            hash_str = hash_obj.hexdigest()
            
            self.logger.info(f"Generated cache key: {hash_str}")
            self.logger.info(f"Feature selection method: {self.method}, n_features: {self.n_features}")
            
            return hash_str
        except Exception as e:
            self.logger.warning(f"Error generating cache key: {str(e)}")
            # Fallback to simpler key generation
            fallback_key = hashlib.md5(str(sorted_requested_features).encode()).hexdigest()
            self.logger.info(f"Using fallback cache key: {fallback_key}")
            return fallback_key

    def get_cache_path(self, cache_key: str) -> str:
        """Get the full path for a cache file"""
        # Use dedicated feature selection cache directory if specified, otherwise use results dir
        feature_selection_cache_dir = self.config.get("output.feature_selection_cache_dir")
        results_dir = self.config.get("output.results_dir", "results")
        
        # If feature_selection_cache_dir is not specified, create it under results_dir
        if not feature_selection_cache_dir:
            feature_selection_cache_dir = os.path.join(results_dir, "feature_selection")
            
        # Make the paths absolute if they're not already
        if not os.path.isabs(feature_selection_cache_dir):
            feature_selection_cache_dir = os.path.abspath(feature_selection_cache_dir)
        
        # Make sure the directory exists
        if not os.path.exists(feature_selection_cache_dir):
            os.makedirs(feature_selection_cache_dir, exist_ok=True)
            self.logger.info(f"Created cache directory: {feature_selection_cache_dir}")
            
        # Create cache path
        cache_path = os.path.join(feature_selection_cache_dir, f"feature_selection_{cache_key}.pkl")
        self.logger.info(f"Cache path: {cache_path}")
        
        return cache_path

    def save_to_cache(
        self,
        column_names: List[str],
        cat_columns: Optional[List[str]]
    ):
        """Save the feature selection results to cache"""
        # Get the actual features requested from config (not columns generated)
        requested_features = self.config.get("features", [])
        
        # Add timestamp for tracking
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        cache_data = {
            "selected_features": self.selected_features,
            "cat_columns": cat_columns,
            "trained": True,
            # Save all parameters to verify cache
            "method": self.method,
            "threshold": self.threshold,
            "n_features": self.n_features,
            "requested_features": requested_features,
            "created_at": timestamp,
            "selected_feature_count": len(self.selected_features) if self.selected_features else 0,
            "original_column_count": len(column_names) if column_names else 0
        }
        
        # Generate cache key based on config only
        cache_key = self.generate_cache_key()
        cache_path = self.get_cache_path(cache_key)
        
        try:
            # Create a temporary file first to avoid corruption
            temp_path = f"{cache_path}.tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Safely replace the original file
            if os.path.exists(cache_path):
                os.remove(cache_path)
            
            os.rename(temp_path, cache_path)
            self.logger.info(f"Saved feature selection results to: {cache_path}")
            
            # Log cache information
            self.logger.info(f"Contains {len(self.selected_features)} selected features")
            
            # Create metadata file
            feature_selection_cache_dir = self.config.get("output.feature_selection_cache_dir")
            if not feature_selection_cache_dir:
                feature_selection_cache_dir = os.path.join(self.config.get("output.results_dir", "results"), "feature_selection")
                
            metadata_dir = os.path.join(feature_selection_cache_dir, "metadata")
            if not os.path.exists(metadata_dir):
                os.makedirs(metadata_dir, exist_ok=True)
                
            metadata_path = os.path.join(metadata_dir, f"feature_selection_{cache_key}_metadata.json")
            metadata = {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v 
                       for k, v in cache_data.items() 
                       if k != "selected_features" and k != "cat_columns"}
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving feature selection to cache: {str(e)}")

    def verify_cache_data(self, cache_data: Dict, requested_features: List[str]) -> Tuple[bool, str, Dict]:
        """
        Verify if the cache data is valid and compatible with current parameters.
        
        Args:
            cache_data: The loaded cache data dictionary
            requested_features: The features requested in the current run
            
        Returns:
            Tuple[bool, str, Dict]: (is_valid, reason, match_details)
        """
        # Check if required fields exist
        if "selected_features" not in cache_data or not cache_data["selected_features"]:
            return False, "Missing selected features", {}
            
        # Compare parameters
        current_features = set(requested_features)
        cache_features = set(cache_data.get("requested_features", []))
        
        # Check exact match
        features_match = cache_features == current_features
        method_match = cache_data.get("method") == self.method
        n_features_match = cache_data.get("n_features") == self.n_features
        
        # Full parameters match
        if features_match and method_match and n_features_match:
            return True, "Parameters match", {
                "selected_feature_count": len(cache_data.get("selected_features", []))
            }
            
        # Check subset match if enabled
        subset_match = current_features.issubset(cache_features)
        allow_subset = self.config.get("feature_selection.allow_subset_match", True)
        
        if subset_match and method_match and n_features_match and allow_subset:
            return True, "Features subset match", {
                "selected_feature_count": len(cache_data.get("selected_features", []))
            }
            
        return False, "Parameters don't match", {}
    
    def load_from_cache(self) -> bool:
        """
        Load feature selection results directly from cache based on config parameters.
        
        Returns:
            bool: True if successfully loaded from cache, False otherwise
        """
        cache_key = self.generate_cache_key()
        cache_path = self.get_cache_path(cache_key)
        
        self.logger.info(f"Attempting to load feature selection cache: {cache_path}")
        
        # Check if the cache file exists
        if not os.path.exists(cache_path):
            self.logger.info(f"Cache file not found: {cache_path}")
            return False
            
        try:
            # Load the cache file
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Get requested features from config
            requested_features = self.config.get("features", [])
            
            # Verify the cache data
            is_valid, reason, match_details = self.verify_cache_data(cache_data, requested_features)
            
            self.logger.info(f"Cache verification result: {reason}")
            
            if is_valid:
                # Load the cache
                self.selected_features = cache_data["selected_features"]
                self.cat_columns = cache_data.get("cat_columns")
                self.trained = True
                
                feature_count = len(self.selected_features) if self.selected_features else 0
                self.logger.info(f"Loaded feature selection with {feature_count} features")
                
                return True
            else:
                self.logger.info(f"Cache verification failed: {reason}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Error loading cache: {str(e)}")
            return False
        
    def cleanup_old_cache_files(self, max_age_days: int = 30, dry_run: bool = True) -> Dict:
        """
        Clean up old cache files to save disk space
        
        Args:
            max_age_days: Maximum age of cache files in days
            dry_run: If True, just report files that would be deleted but don't delete them
            
        Returns:
            Dict: Summary of cleanup operation
        """
        from datetime import datetime, timedelta
        import glob
        
        # Get cache directory
        feature_selection_cache_dir = self.config.get("output.feature_selection_cache_dir")
        if not feature_selection_cache_dir:
            feature_selection_cache_dir = os.path.join(
                self.config.get("output.results_dir", "results"), 
                "feature_selection"
            )
        
        if not os.path.isabs(feature_selection_cache_dir):
            feature_selection_cache_dir = os.path.abspath(feature_selection_cache_dir)
            
        # Find all cache files
        cache_files = glob.glob(os.path.join(feature_selection_cache_dir, "feature_selection_*.pkl"))
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Filter files by age
        old_files = []
        for file_path in cache_files:
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if mtime < cutoff_date:
                    old_files.append((file_path, mtime, os.path.getsize(file_path)))
            except Exception as e:
                self.logger.warning(f"Error processing {file_path}: {str(e)}")
        
        # Sort by modification time (oldest first)
        old_files.sort(key=lambda x: x[1])
        
        total_size = sum(size for _, _, size in old_files)
        total_size_mb = total_size / (1024 * 1024)
        
        self.logger.info(f"Found {len(cache_files)} cache files, {len(old_files)} older than {max_age_days} days")
        self.logger.info(f"Total space to be freed: {total_size_mb:.2f} MB")
        
        # Delete files if not dry run
        deleted_count = 0
        if not dry_run and old_files:
            for file_path, _, _ in old_files:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    self.logger.error(f"Error deleting {file_path}: {str(e)}")
        
        # Return summary
        return {
            "total_files": len(cache_files),
            "old_files": len(old_files),
            "deleted_files": deleted_count,
            "total_size_mb": total_size_mb,
            "dry_run": dry_run
        }
