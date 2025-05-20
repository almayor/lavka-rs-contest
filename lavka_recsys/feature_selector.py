import json
import polars as pl
import pandas as pd
import numpy as np
import hashlib # For creating cache key
import os      # For checking if cache file exists

from typing import Iterable, List # Added List for type hinting

# Assuming these are correctly defined elsewhere in your project
from .utils.config import Config
from .utils.custom_logging import get_logger

class FeatureSelector:

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger('FeatureSelector')

    def __call__(self, features_df: pl.DataFrame, cat_columns: Iterable) -> tuple[pl.DataFrame, List[str]]: # Added return type hint
        if not self.config.get('feature_selector.enabled', True):
            self.logger.info("Skipping feature selection as it's disabled in config.")
            return features_df, list(cat_columns) # Ensure cat_columns is a list

        # Ensure cat_columns is a list for consistent handling
        cat_columns_list = list(cat_columns)

        try:
            with open(self.config['feature_selector.importances_path'], 'r') as f:
                importances_dict = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Importances file not found at: {self.config['feature_selector.importances_path']}. Skipping selection.")
            return features_df, cat_columns_list
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON from importances file: {self.config['feature_selector.importances_path']}. Skipping selection.")
            return features_df, cat_columns_list
        
        # Prepare numerical features DataFrame
        # Ensure cat_columns_list contains actual column names present in features_df
        numerical_feature_df_cols = [col for col in features_df.columns if col not in cat_columns_list]
        if not numerical_feature_df_cols:
            self.logger.info("No numerical features found after excluding categorical columns. Skipping selection.")
            return features_df, cat_columns_list
            
        X = features_df.select(numerical_feature_df_cols).to_pandas()
        
        correlation_threshold = self.config.get('feature_selector.correlation_threshold', 0.9)

        # --- Caching Logic ---
        # Create a signature for the current computation
        # Based on numerical feature names, threshold, and content of importances_dict
        sorted_numerical_feature_names = sorted(X.columns.tolist())
        
        hasher = hashlib.sha256()
        hasher.update("||".join(sorted_numerical_feature_names).encode('utf-8'))
        hasher.update(str(correlation_threshold).encode('utf-8'))
        # Hash the content of importances_dict for robustness
        importances_content_hash = hashlib.sha256(json.dumps(importances_dict, sort_keys=True).encode('utf-8')).hexdigest()
        hasher.update(importances_content_hash.encode('utf-8'))
        input_signature_key = hasher.hexdigest()

        cache_file_path = self.config.get('feature_selector.cache_file', 'feature_selector_cache.json')
        selected_numerical_features_list = None
        cache_data = {}

        if self.config.get('feature_selector.cache_enabled', True): # Allow disabling cache via config
            if os.path.exists(cache_file_path):
                try:
                    with open(cache_file_path, 'r') as f:
                        cache_data = json.load(f)
                    if input_signature_key in cache_data:
                        selected_numerical_features_list = cache_data[input_signature_key]
                        self.logger.info(f"Loaded selected features from cache (key: {input_signature_key[:10]}...).")
                except Exception as e:
                    self.logger.warning(f"Could not load or parse cache file {cache_file_path}: {e}. Recomputing.")
                    cache_data = {} # Reset cache data if file is corrupt
            else:
                self.logger.info(f"Cache file {cache_file_path} not found. Will compute features.")
        else:
            self.logger.info("Feature selection caching is disabled in config.")


        if selected_numerical_features_list is None:
            self.logger.info("No valid cache found or caching disabled. Running feature selection logic...")
            
            self.logger.info('Calculating correlation matrix...')
            corr_matrix = X.corr().abs()

            features_to_drop = set()
            processed_features_in_groups = set()

            self.logger.info(f"Identifying correlated groups (threshold > {correlation_threshold}):")

            for feature1 in X.columns:
                if feature1 in processed_features_in_groups or feature1 in features_to_drop:
                    continue

                current_correlated_group = {feature1} 

                for feature2 in X.columns:
                    if feature1 == feature2 or feature2 in processed_features_in_groups or feature2 in features_to_drop:
                        continue

                    if corr_matrix.loc[feature1, feature2] > correlation_threshold:
                        current_correlated_group.add(feature2)

                if len(current_correlated_group) > 1:
                    self.logger.info(f"  Found correlated group: {current_correlated_group}")
                    best_feature_in_group = None
                    max_importance_in_group = -float('inf') 

                    for f_in_group in current_correlated_group:
                        importance = importances_dict.get(f_in_group, 0) 
                        if importance > max_importance_in_group:
                            max_importance_in_group = importance
                            best_feature_in_group = f_in_group
                        # Optional: Add sophisticated tie-breaking here if needed

                    if best_feature_in_group: 
                        self.logger.info(f"    Best feature in group (highest importance): '{best_feature_in_group}' (Importance: {max_importance_in_group:.4f})")
                        for f_in_group in current_correlated_group:
                            processed_features_in_groups.add(f_in_group) 
                            if f_in_group != best_feature_in_group:
                                self.logger.info(f"      -> Marking '{f_in_group}' for removal (Importance: {importances_dict.get(f_in_group, 0):.4f})")
                                features_to_drop.add(f_in_group)
                    else: 
                        self.logger.warning(f"    Warning: Could not determine best feature for group {current_correlated_group}")
                        for f_in_group in current_correlated_group:
                            processed_features_in_groups.add(f_in_group)
            
            selected_numerical_features_list = [f for f in X.columns if f not in features_to_drop]

            # Save to cache if caching is enabled
            if self.config.get('feature_selector.cache_enabled', True):
                cache_data[input_signature_key] = selected_numerical_features_list
                try:
                    with open(cache_file_path, 'w') as f:
                        json.dump(cache_data, f, indent=4)
                    self.logger.info(f"Saved selected features to cache (key: {input_signature_key[:10]}...). File: {cache_file_path}")
                except Exception as e:
                    self.logger.error(f"Could not save cache file {cache_file_path}: {e}")
        
        # Combine selected numerical features with original categorical columns
        # Ensure no duplicates if cat_columns could somehow overlap with numerical ones (unlikely but safe)
        final_selected_columns_set = set(selected_numerical_features_list)
        for col in cat_columns_list: # Add categorical columns, preserving their original order relative to each other
            final_selected_columns_set.add(col)
        
        # Maintain original order as much as possible for kept columns
        original_order_all_cols = features_df.columns
        final_selected_columns = [col for col in original_order_all_cols if col in final_selected_columns_set]


        self.logger.info(f"Original feature count: {len(features_df.columns)}")
        self.logger.info(f"Selected numerical feature count: {len(selected_numerical_features_list)}")
        self.logger.info(f"Categorical feature count: {len(cat_columns_list)}")
        self.logger.info(f"Total final selected feature count: {len(final_selected_columns)}")
        # self.logger.info(f"Final selected features: {final_selected_columns}") # Can be very verbose

        return features_df.select(final_selected_columns), cat_columns_list