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
from .experiment import Experiment
from .model_factory import Model

class FeatureSelector(Experiment):
    """
    Feature selection as a specialized experiment.
    
    This class treats feature selection as a lightweight experiment
    that learns which features are most important for a specific model type.
    """
    
    def __init__(self, name, config, select_for_model_type=None):
        """
        Initialize feature selector with lightweight configuration.
        
        Args:
            name: Name of the experiment
            config: Base configuration
            select_for_model_type: Model type to select features for (defaults to config's model type)
        """
        # Create a lightweight configuration for feature selection
        selection_config = config.copy()
        
        # Reduce iterations and data size for faster training
        if 'model.config.catboost.iterations' in selection_config.to_dict():
            selection_config.set('model.config.catboost.iterations', 50)
        if 'model.config.catboost_ranker.iterations' in selection_config.to_dict():
            selection_config.set('model.config.catboost_ranker.iterations', 50)
        if 'training.max_splits' in selection_config.to_dict():
            selection_config.set('training.max_splits', 2)
        if 'data.sample_fraction' not in selection_config.to_dict():
            selection_config.set('data.sample_fraction', 0.3)  # Use 30% of data by default
            
        # Initialize as Experiment with modified config
        super().__init__(f"{name}_feature_selection", selection_config)
        
        # Store additional feature selection parameters
        self.select_for_model_type = select_for_model_type or config.get('model.type')
        self.original_config = config  # Keep reference to original config
        
        # Feature selection specific parameters
        self.method = config.get('feature_selection.method', 'importance')
        self.threshold = config.get('feature_selection.threshold')
        self.n_features = config.get('feature_selection.n_features', 10)
        
        # Placeholders for results
        self.selected_features = None
        self.cat_columns = None
        self.trained = False
        
        self.logger.info(f"Initialized FeatureSelector for model type: {self.select_for_model_type}")
        self.logger.info(f"Selection method: {self.method}, n_features: {self.n_features}")
        
    def run(self) -> Dict:
        """
        Run feature selection as a lightweight experiment.
        
        Returns:
            Dict: Results with selected features and metadata
        """
        self.logger.info(f"Running feature selection for model type: {self.select_for_model_type}")
        
        # Override model type to match what we're selecting for
        original_model_type = self.config.get('model.type')
        if self.select_for_model_type != original_model_type:
            self.config.set('model.type', self.select_for_model_type)
            self.logger.info(f"Temporarily changed model type from {original_model_type} to {self.select_for_model_type}")
        
        # Try to load from cache first
        cache_key = self._generate_cache_key()
        cache_path = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_path) and self.config.get('feature_selection.use_cache', True):
            self.logger.info(f"Attempting to load feature selection from cache: {cache_path}")
            if self._load_from_cache(cache_path):
                return {
                    'selected_features': self.selected_features,
                    'cat_columns': self.cat_columns,
                    'from_cache': True
                }
        
        # Run a lightweight experiment to get feature importance
        self.logger.info("No valid cache found or cache disabled. Running feature selection experiment...")
        
        # Use parent class to run the experiment and get a trained model
        results, model = self._run_single_experiment()
        
        # Extract feature importance from model
        feature_importance = results.get('feature_importance', {})
        
        if not feature_importance:
            self.logger.warning("No feature importance available from model. Using all features.")
            feature_names = self.config.get('features', [])
            self.selected_features = feature_names
            self.trained = True
            return {
                'selected_features': self.selected_features,
                'cat_columns': None,
                'from_cache': False
            }
        
        # Select top features based on importance
        self.logger.info(f"Selecting top {self.n_features} features based on importance")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Get categorical columns if available
        if hasattr(self.data_loader, 'categorical_columns'):
            self.cat_columns = self.data_loader.categorical_columns
        
        # Determine number of features to select
        n_to_select = min(self.n_features, len(sorted_features))
        self.selected_features = [f[0] for f in sorted_features[:n_to_select]]
        
        self.logger.info(f"Selected {len(self.selected_features)} features:")
        for i, feature in enumerate(self.selected_features[:10]):  # Log top 10
            importance = feature_importance.get(feature, 0)
            self.logger.info(f"  {i+1}. {feature}: {importance:.6f}")
        
        if len(self.selected_features) > 10:
            self.logger.info(f"  ... and {len(self.selected_features)-10} more")
        
        # Save to cache
        self._save_to_cache(cache_key, sorted_features)
        
        # Mark as trained
        self.trained = True
        
        # Restore original model type if changed
        if self.select_for_model_type != original_model_type:
            self.config.set('model.type', original_model_type)
        
        return {
            'selected_features': self.selected_features,
            'cat_columns': self.cat_columns,
            'from_cache': False,
            'feature_importance': {f: i for f, i in sorted_features},
            'model_type': self.select_for_model_type
        }
        
    def apply_selection(self, features_df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply feature selection to a DataFrame, keeping only selected features.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            pl.DataFrame: DataFrame with only selected features
        """
        if not self.trained:
            self.logger.error("Feature selector not trained. Call run() first.")
            raise RuntimeError("Feature selector not trained")
            
        if not self.selected_features:
            self.logger.warning("No features were selected! Returning original DataFrame.")
            return features_df
            
        self.logger.debug(f"Applying selection to keep {len(self.selected_features)} features")
        
        # Determine which columns to select
        columns_to_select = []
        
        # Add categorical columns first if they exist
        if self.cat_columns:
            columns_to_select.extend([col for col in self.cat_columns if col in features_df.columns])
            
        # Add selected feature columns
        columns_to_select.extend([col for col in self.selected_features if col in features_df.columns])
        
        # Check if we have all columns
        missing_columns = set(self.selected_features) - set(features_df.columns)
        if missing_columns:
            self.logger.warning(f"Missing {len(missing_columns)} selected features in DataFrame")
            self.logger.debug(f"Missing features: {missing_columns}")
            
        # Apply selection
        if not columns_to_select:
            self.logger.error("No valid columns to select!")
            return features_df
            
        return features_df.select(columns_to_select)
    
    def _generate_cache_key(self) -> str:
        """Generate a unique key for caching based on selection parameters"""
        # Get requested features from config
        requested_features = self.config.get("features", [])
        sorted_requested_features = sorted(requested_features)
        
        # Create a dictionary with parameters that affect feature selection
        cache_params = {
            "requested_features": sorted_requested_features,
            "method": self.method,
            "threshold": self.threshold,
            "n_features": self.n_features,
            "model_type": self.select_for_model_type
        }
        
        # Convert to JSON string and create hash
        try:
            params_str = json.dumps(cache_params, sort_keys=True)
            hash_obj = hashlib.md5(params_str.encode())
            hash_str = hash_obj.hexdigest()
            
            self.logger.debug(f"Generated cache key: {hash_str}")
            return hash_str
        except Exception as e:
            self.logger.warning(f"Error generating cache key: {str(e)}")
            # Fallback to simpler key generation
            return hashlib.md5(str(sorted_requested_features).encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the full path for a cache file"""
        # Use dedicated feature selection cache directory if specified
        feature_selection_cache_dir = self.config.get("output.feature_selection_cache_dir")
        results_dir = self.config.get("output.results_dir", "results")
        
        # If not specified, create it under results_dir
        if not feature_selection_cache_dir:
            feature_selection_cache_dir = os.path.join(results_dir, "feature_selection_cache")
            
        # Make the paths absolute if they're not already
        if not os.path.isabs(feature_selection_cache_dir):
            feature_selection_cache_dir = os.path.abspath(feature_selection_cache_dir)
        
        # Make sure the directory exists
        if not os.path.exists(feature_selection_cache_dir):
            os.makedirs(feature_selection_cache_dir, exist_ok=True)
            
        # Create cache path
        cache_path = os.path.join(feature_selection_cache_dir, f"feature_selection_{cache_key}.pkl")
        return cache_path
    
    def _save_to_cache(self, cache_key: str, sorted_features: List[Tuple[str, float]]):
        """Save feature selection results to cache"""
        # Prepare data to save
        cache_data = {
            "selected_features": self.selected_features,
            "cat_columns": self.cat_columns,
            "sorted_features": sorted_features,
            "method": self.method,
            "threshold": self.threshold,
            "n_features": self.n_features,
            "model_type": self.select_for_model_type,
            "timestamp": self._get_timestamp(),
            "trained": True
        }
        
        # Get cache path
        cache_path = self._get_cache_path(cache_key)
        
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
            
            # Save diagnostic information as JSON for easy inspection
            self._save_diagnostic_info(cache_key, cache_data)
            
        except Exception as e:
            self.logger.error(f"Error saving feature selection to cache: {str(e)}")
    
    def _load_from_cache(self, cache_path: str) -> bool:
        """Load feature selection results from cache"""
        try:
            # Load the cache file
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Verify the cache data
            if not self._verify_cache_data(cache_data):
                self.logger.info("Cache verification failed")
                return False
                
            # Load the cache
            self.selected_features = cache_data.get("selected_features")
            self.cat_columns = cache_data.get("cat_columns")
            self.trained = True
            
            self.logger.info(f"Loaded {len(self.selected_features)} features from cache")
            return True
                
        except Exception as e:
            self.logger.warning(f"Error loading cache: {str(e)}")
            return False
    
    def _verify_cache_data(self, cache_data: Dict) -> bool:
        """Verify cache data compatibility with current parameters"""
        # Check for required fields
        if "selected_features" not in cache_data or not cache_data["selected_features"]:
            return False
            
        # Check method and model type
        if cache_data.get("method") != self.method:
            return False
            
        if cache_data.get("model_type") != self.select_for_model_type:
            return False
            
        # Check feature count
        if cache_data.get("n_features") != self.n_features:
            return False
            
        return True
    
    def _save_diagnostic_info(self, cache_key: str, cache_data: Dict):
        """Save diagnostic information for feature selection cache"""
        # Use the cache directory
        feature_selection_cache_dir = self.config.get("output.feature_selection_cache_dir")
        results_dir = self.config.get("output.results_dir", "results")
        
        if not feature_selection_cache_dir:
            feature_selection_cache_dir = os.path.join(results_dir, "feature_selection_cache")
            
        # Create metadata directory if it doesn't exist
        metadata_dir = os.path.join(feature_selection_cache_dir, "metadata")
        if not os.path.exists(metadata_dir):
            os.makedirs(metadata_dir, exist_ok=True)
            
        # Create diagnostic data - exclude large fields
        diagnostic_data = {
            "method": cache_data.get("method"),
            "model_type": cache_data.get("model_type"),
            "n_features": cache_data.get("n_features"),
            "threshold": cache_data.get("threshold"),
            "timestamp": cache_data.get("timestamp"),
            "selected_feature_count": len(cache_data.get("selected_features", [])),
            "top_features": cache_data.get("selected_features", [])[:10],  # Only include top 10
            "has_categorical_columns": cache_data.get("cat_columns") is not None
        }
        
        # Add top 5 features with importance if available
        sorted_features = cache_data.get("sorted_features", [])
        if sorted_features:
            top_importance = {f: i for f, i in sorted_features[:5]}
            diagnostic_data["top_importance"] = top_importance
        
        # Save diagnostic data
        diagnostic_path = os.path.join(metadata_dir, f"feature_selection_{cache_key}_diagnostic.json")
        try:
            with open(diagnostic_path, 'w') as f:
                json.dump(diagnostic_data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Error saving diagnostic info: {str(e)}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()