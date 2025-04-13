from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

from .config import Config
from .custom_logging import get_logger


class FeatureFactory:
    """Feature generation with selective feature creation"""
    
    # Class-level registry of feature generators
    _feature_registry = {}
    _possible_targets = set()
    
    @classmethod
    def register(cls, feature_name: str,
                 depends_on: str | List[str] | None = None,
                 categorical_cols: List[str] | None = None):
        """
        Decorator to register a method as a feature generator and to specify its dependencies. The method
        must accept two arguments: history_df and target_df, and return a target_df with new columns.
        Args:
            feature_name (str): Name of the feature to be generated.
            categorical_cols (List[str] | None): List of categorical columns this feature produces.
            depends_on (str | List[str] | None): List of features that this feature depends on.
        """
        depends_on = depends_on or []
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            cls._feature_registry[feature_name] = {
                'func': wrapper,
                'depends_on': depends_on,
                'categorical_cols': categorical_cols or [],
            }
            return wrapper
        
        return decorator

    @classmethod
    def register_target(cls, feature_name: str,
                 depends_on: str | List[str] | None = None):
        """
        Decorator to register a method as a target generator and to specify its dependencies. The method
        must accept two arguments: history_df and target_df, and return a pl.Series of the target. Rows where
        the target is null will be filtered out.
        Args:
            feature_name (str): Name of the target to be generated.
            depends_on (str | List[str] | None): List of features that this target depends on.
        """
        depends_on = depends_on or []
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return result, []
        
            cls._feature_registry[feature_name] = {
                'func': wrapper,
                'depends_on': depends_on,
                'categorical_cols': [],
            }
            cls._possible_targets.add(feature_name)

            return wrapper
        
        return decorator

    def __init__(self, config: Config):
        """Initialize feature factory"""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    def generate_batch(
            self, history_df, target_df, requested_features=None, requested_target=None
        ) -> Tuple[pl.DataFrame, pl.Series, List[str], pl.Series]:
        """
        Generate features and target for a batch of requests.
        Args:
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            requested_features (str | List[str] | None): Features to generate (if None, config is used).
            requested_target (str | None): Target to generate (if None, config is used).
        Returns:
            Tuple[pl.DataFrame, pl.Series, List[str], pl.Series]: Tuple containing:
                - Generated features (pl.DataFrame)
                - Target (pl.Series)
                - Categorical column names in the generated features (List[str])
                - Request IDs per row (pl.Series)
        """
        request_ids = target_df['request_id']
        features, cat_columns = self.generate_features(history_df, target_df, requested_features)
        target, _ = self.generate_target(history_df, target_df, requested_target)
        mask = ~target.is_null()
        return (
            features.filter(mask),
            target.filter(mask),
            cat_columns,
            request_ids.filter(mask),
        )

    def generate_features(
            self, history_df: pl.DataFrame, target_df: pl.DataFrame, requested_features: List[str] | None = None
        ) -> pl.DataFrame:
        """
        Generate only the requested features and their dependencies
        Args:
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            requested_features (List[str] | None): Features to generate (if None, config is used).
        Returns:
            Tuple[pl.DataFrame, List[str]]: Tuple containing:
                - Generated features (pl.DataFrame)
                - Categorical column names in the generated features (List[str])
        """
        if requested_features is None:
            requested_features = self.config.get("features")
        if len(requested_features) != len(set(requested_features)):
            self.logger.error("Duplicate feature names in requested_features")
            raise ValueError("Duplicate feature names in requested_features")
        self.logger.info(f"Generating features: {', '.join(requested_features)}")
                
        # Generate each requested feature (and dependencies)
        all_columns, all_cat_columns = set(), set()
        for feature_name in requested_features:
            target_df, columns = self._generate_feature(
                feature_name, history_df, target_df
            )
            all_columns.update(columns)
            cat_columns = self.__class__._feature_registry[feature_name]['categorical_cols']
            all_columns.update(cat_columns) # preserve categorical features even if they already existed
            all_cat_columns.update(cat_columns)
        
        all_cat_columns = list(all_cat_columns) if len(all_cat_columns) else None
        self.logger.info("Joined features")
        self.logger.info(f"All column names: {all_columns}")
        self.logger.info(f"All categorical column names: {all_cat_columns}")
        return target_df.select(all_columns), all_cat_columns
    
    def generate_target(self, history_df, target_df, requested_target: str | None = None) -> pl.Series:
        """
        Generate the target variable.
        Args:
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            requested_target (str | None): Target to generate (if None, config is used).
        Returns:
            pl.Series: Generated target (pl.Series)
        """
        if requested_target is None:
            requested_target = self.config.get("target")
        if requested_target not in self.__class__._possible_targets:
            raise ValueError(f"Unknown target {requested_target}")

        feature, _ = self._generate_feature(requested_target, history_df, target_df)
        return feature
    
    def _generate_feature(
            self, feature_name: str, history_df: pl.DataFrame, target_df: pl.DataFrame,
            generated_features=None
        ) -> Tuple[pl.DataFrame, List[str]]:
        """
        Generate a single feature, handling dependencies
        Args:
            feature_name (str): Name of the feature to be generated.
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            generated_features (set | None): Set of already generated features (for caching).
        Returns:
            Tuple[pl.DataFrame, List[str], List[str]]: Tuple containing:
                - Generated features (pl.DataFrame)
                - Newly added column names
        """
        # Return from cache if already generated
        if generated_features and feature_name in generated_features:
            return target_df
        
        # Check if feature exists
        if feature_name not in self.__class__._feature_registry:
            msg = f"Feature '{feature_name}' is not registered"
            self.logger.error(msg)
            raise ValueError(msg)
        
        # Get feature info
        feature_info = self.__class__._feature_registry[feature_name]
        generator_func = feature_info['func']
        dependencies = feature_info['depends_on']
        
        # Generate dependencies first
        generated_features = generated_features or set()
        for dep in dependencies:
            self._generate_feature(dep, history_df, target_df, generated_features)
        
        # Generate this feature
        self.logger.debug(f"Generating feature: {feature_name}")
        old_columns = target_df.columns
        features = generator_func(history_df, target_df)
    
        if isinstance(features, pl.DataFrame):
            new_columns = list(set(features.columns).difference(old_columns))
        else:
            new_columns = [feature_name]
        self.logger.debug(f"New column names: {new_columns}")
        generated_features.add(feature_name)
        
        # Cache and return
        return features, new_columns


# ========== BASIC FEATURES = ==========

@FeatureFactory.register('count_purchase_user_product')
def generate_count_purchase_user_product(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Count purchases by user-product pairs"""
    return history_df.filter(
        pl.col('action_type') == "AT_Purchase"
    ).group_by(
        'user_id', 'product_id'
    ).agg(
        pl.len().alias('count_purchase_u_p')
    ).join(
        target_df,
        on=['user_id', 'product_id'],
        how='right'
    ).fill_null(0)

@FeatureFactory.register('count_purchase_user_store')
def generate_count_purchase_user_store(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Count purchases by user-store pairs"""
    return history_df.filter(
        pl.col('action_type') == "AT_Purchase"
    ).group_by(
        'user_id', 'store_id'
    ).agg(
        pl.len().alias('count_purchase_u_s')
    ).join(
        target_df,
        on=['user_id', 'store_id'],
        how='right'
    ).fill_null(0)

@FeatureFactory.register('ctr_product')
def generate_ctr_product(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Calculate CTR (Click-Through Rate) for products"""
    actions = history_df.group_by(
        'action_type', 'product_id'
    ).agg(
        pl.len()
    )
    
    clicks = actions.filter(pl.col('action_type') == "AT_Click")
    views = actions.filter(pl.col('action_type') == "AT_View")
    
    feature = clicks.join(
        views, on='product_id'
    ).with_columns(
        ctr_product=pl.col('len') / pl.col('len_right')
    ).select(
        'product_id', 'ctr_product'
    )
    return target_df.join(
        feature,
        on=['product_id'],
        how='left'
    )

@FeatureFactory.register('recency_user_product')
def generate_recency_user_product(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Generate recency features for user-product pairs"""
    latest_time = history_df['timestamp'].max()
    
    feature = history_df.group_by(['user_id', 'product_id']).agg(
        pl.max('timestamp').alias('last_interaction_u_p')
    ).with_columns(
        days_since_interaction_u_p=(latest_time - pl.col('last_interaction_u_p')) / (24 * 60 * 60)
    )
    return target_df.join(
        feature,
        on=['user_id', 'product_id'],
        how='left'
    )

@FeatureFactory.register('recency_user_store')
def generate_recency_user_store(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Generate recency features for user-store pairs"""
    latest_time = history_df['timestamp'].max()
    
    feature = history_df.group_by(['user_id', 'store_id']).agg(
        pl.max('timestamp').alias('last_interaction_u_s')
    ).with_columns(
        days_since_interaction_u_s=(latest_time - pl.col('last_interaction_u_s')) / (24 * 60 * 60)
    )
    return target_df.join(
        feature,
        on=['user_id', 'store_id'],
        how='left'
    )

@FeatureFactory.register('user_stats')
def generate_user_stats(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Generate user-level statistics"""
    feature = history_df.group_by('user_id').agg([
        pl.len().alias('user_total_interactions'),
        pl.col('action_type').eq('AT_Purchase').sum().alias('user_total_purchases'),
        pl.col('action_type').eq('AT_View').sum().alias('user_total_views'),
        pl.n_unique('product_id').alias('user_unique_products')
    ])
    return target_df.join(
        feature,
        on=['user_id'],
        how='left'
    )

@FeatureFactory.register('product_stats')
def generate_product_stats(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Generate product-level statistics"""
    features = history_df.group_by('product_id').agg([
        pl.len().alias('product_total_interactions'),
        pl.col('action_type').eq('AT_Purchase').sum().alias('product_total_purchases'),
        pl.col('action_type').eq('AT_View').sum().alias('product_total_views'),
        pl.n_unique('user_id').alias('product_unique_users')
    ])
    return target_df.join(
        features,
        on=['product_id'],
        how='left'
    )
@FeatureFactory.register('store_stats')
def generate_store_stats(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Generate store-level statistics"""
    feature = history_df.group_by('store_id').agg([
        pl.len().alias('store_total_interactions'),
        pl.col('action_type').eq('AT_Purchase').sum().alias('store_total_purchases'),
        pl.col('action_type').eq('AT_View').sum().alias('store_total_views'),
        pl.n_unique('product_id').alias('store_unique_products')
    ])
    return target_df.join(
        feature,
        on=['store_id'],
        how='left'
    )
@FeatureFactory.register('city_stats', categorical_cols=['city_name'])
def generate_city_stats(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Generate city-level statistics"""
    feature = history_df.group_by('city_name').agg([
        pl.len().alias('city_total_interactions'),
        pl.col('action_type').eq('AT_Purchase').sum().alias('city_total_purchases'),
        pl.col('action_type').eq('AT_View').sum().alias('city_total_views'),
        pl.n_unique('store_id').alias('city_unique_stores')
    ])
    # add ratio of purchases to views
    feature = feature.with_columns(
        city_purchase_view_ratio=pl.col('city_total_purchases') / pl.col('city_total_views')
    )
    # add ratio of purchases to interactions
    feature = feature.with_columns(
        city_purchase_interaction_ratio=pl.col('city_total_purchases') / pl.col('city_total_interactions')
    )
    return target_df.join(
        feature,
        on=['city_name'],
        how='left'
    )
    

# ========== TIME FEATURES = ==========

@FeatureFactory.register('time_features')
def generate_time_features(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Generate time-related features (hour of day, day of week, etc.)"""
    return target_df.with_columns([
        pl.col('timestamp').dt.hour().alias('hour_of_day'),
        pl.col('timestamp').dt.weekday().alias('day_of_week'),
        pl.col('timestamp').dt.month().alias('month'),
        pl.col('timestamp')
            .dt.weekday()
                    .is_in([6, 7])  # 5=Saturday, 6=Sunday
                    .alias('is_weekend')
    ])

@FeatureFactory.register('product_temporal_patterns')
def generate_product_temporal_patterns(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Generate features related to typical purchase times and days for products"""
    # Filter to only purchase events
    purchases = history_df.filter(pl.col('action_type') == "AT_Purchase")
    
    # Extract temporal features
    purchases = purchases.with_columns(
        pl.col('timestamp').dt.hour().alias('hour_of_day'),
        pl.col('timestamp').dt.weekday().alias('day_of_week')
    )
    
    # Calculate hourly patterns for each product
    hour_stats = purchases.group_by('product_id').agg(
        pl.mean('hour_of_day').alias('avg_purchase_hour'),
        pl.std('hour_of_day').alias('std_purchase_hour')
    )
    
    # Calculate day of week patterns for each product
    day_stats = purchases.group_by(['product_id', 'day_of_week']).agg(
        pl.count().alias('purchase_count')
    )
    
    # Find the most common purchase day for each product
    most_common_day = day_stats.sort(['product_id', 'purchase_count'], descending=[False, True]) \
        .group_by('product_id') \
        .agg(pl.first('day_of_week').alias('most_common_purchase_day'))
    
    # Join hour and day stats
    temporal_stats = hour_stats.join(most_common_day, on='product_id')
    
    # Join with target data
    result = target_df.join(
        temporal_stats,
        on=['product_id'],
        how='left'
    )
    
    # Add current temporal information
    result = result.with_columns(
        pl.col('timestamp').dt.hour().alias('current_hour'),
        pl.col('timestamp').dt.weekday().alias('current_day')
    )
    
    # Calculate differences (using abs() on the column)
    result = result.with_columns(
        (pl.col('current_hour') - pl.col('avg_purchase_hour')).abs().alias('hour_diff'),
        (pl.col('current_day') - pl.col('most_common_purchase_day')).abs().alias('day_diff')
    )
    
    # Apply circular distance formula 
    result = result.with_columns(
        pl.min_horizontal(pl.col('hour_diff'), 24 - pl.col('hour_diff')).alias('hour_distance'),
        pl.min_horizontal(pl.col('day_diff'), 7 - pl.col('day_diff')).alias('day_distance')
    )
    
    # Convert to relevance scores (0 to 1)
    result = result.with_columns(
        (1 - pl.col('hour_distance') / 12).alias('hour_relevance'),
        (1 - pl.col('day_distance') / 3.5).alias('day_of_week_relevance')
    )
    
    # Keep only the relevant columns
    final_cols = target_df.columns + ['avg_purchase_hour', 'std_purchase_hour', 
                                     'most_common_purchase_day', 'hour_relevance', 
                                     'day_of_week_relevance']
    return result.select(final_cols)

# ========== TIME WINDOW FEATURES ===========

@FeatureFactory.register('time_window_user_product')
def generate_time_window_features(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Generate time-window based features for user-product pairs"""
    latest_time = history_df['timestamp'].max()
    
    # 1-day, 7-day, and 30-day windows (in seconds)
    day_seconds = 24 * 60 * 60
    windows = {
        'day': day_seconds,
        'week': 7 * day_seconds,
        'month': 30 * day_seconds
    }
    
    result = target_df
    
    for name, seconds in windows.items():
        cutoff_time = latest_time - seconds
        
        window_df = history_df.filter(pl.col('timestamp') >= cutoff_time)
        
        # Count interactions in this window
        counts = window_df.group_by(['user_id', 'product_id']).agg([
            pl.len().alias(f'interactions_{name}_u_p'),
            pl.col('action_type').eq('AT_Purchase').sum().alias(f'purchases_{name}_u_p'),
            pl.col('action_type').eq('AT_View').sum().alias(f'views_{name}_u_p')
        ])
        
        result = result.join(counts, on=['user_id', 'product_id'], how='left').fill_null(0)
    
    return result

# ========== SESSION FEATURES = ==========

@FeatureFactory.register('session_features')
def generate_session_features(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Generate session-based features.
    A session is defined as a sequence of actions by the same user within a time window.
    """
    # Define session window (30 minutes in seconds)
    session_window = 30 * 60
    
    # Sort history by user and timestamp
    sorted_history = history_df.sort(['user_id', 'timestamp'])
    
    # Create session ID column
    history_with_sessions = sorted_history.with_columns([
        # Create a new column that's 1 when this row starts a new session
        # (either first action by user or >30min gap from previous action)
        pl.when(
            (pl.col('timestamp') - pl.col('timestamp').shift(1) > session_window) | 
            (pl.col('user_id') != pl.col('user_id').shift(1))
        ).then(1).otherwise(0).alias('new_session')
    ])
    
    # Create a cumulative sum to get session IDs
    history_with_sessions = history_with_sessions.with_columns([
        pl.col('new_session').cum_sum().over('user_id').alias('session_id')
    ])
    
    # Calculate session-level features
    session_features = history_with_sessions.group_by(['user_id', 'session_id']).agg([
        pl.len().alias('session_length'),
        pl.col('action_type').eq('AT_Purchase').sum().alias('session_purchases'),
        pl.n_unique('product_id').alias('session_unique_products'),
        pl.n_unique('store_id').alias('session_unique_stores'),
        pl.col('timestamp').min().alias('session_start'),
        pl.col('timestamp').max().alias('session_end')
    ])
    
    # Calculate session duration
    session_features = session_features.with_columns([
        (pl.col('session_end') - pl.col('session_start')).alias('session_duration_seconds')
    ])
    
    # Get the most recent session for each user
    latest_sessions = session_features.group_by('user_id').agg([
        pl.col('session_id').max().alias('latest_session_id')
    ])
    
    latest_session_features = latest_sessions.join(
        session_features, 
        left_on=['user_id', 'latest_session_id'],
        right_on=['user_id', 'session_id']
    ).select([
        'user_id', 
        'session_length', 
        'session_purchases', 
        'session_unique_products',
        'session_unique_stores',
        'session_duration_seconds'
    ])
    
    # Join with target data
    return target_df.join(latest_session_features, on='user_id', how='left')

# ========== FREQUENCY FEATURES = ==========

@FeatureFactory.register('frequency_features')
def generate_frequency_features(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Generate frequency-based features"""
    # Get timestamps of all interactions for each user-product pair
    interactions = history_df.filter(
        ~pl.col('action_type').is_in(['AT_View'])  # Exclude views for meaningful frequency
    ).group_by(['user_id', 'product_id']).agg([
        pl.col('timestamp').sort().alias('interaction_times')
    ])
    
    # Calculate intervals between interactions
    def calculate_intervals(times):
        if len(times) <= 1:
            return None
        intervals = [times[i] - times[i-1] for i in range(1, len(times))]
        return intervals
    
    # Add return_dtype to the first map_elements
    interactions = interactions.with_columns([
        pl.col('interaction_times')
          .map_elements(calculate_intervals, return_dtype=pl.List(pl.Float64))
          .alias('intervals')
    ])
    
    # Calculate mean interval (frequency)
    def mean_interval(intervals):
        if intervals is None or len(intervals) == 0:
            return None
        return sum(intervals) / len(intervals)
    
    # Add return_dtype to the second map_elements
    interactions = interactions.with_columns([
        pl.col('intervals')
          .map_elements(mean_interval, return_dtype=pl.Float64)
          .alias('mean_interval_seconds')
    ])
    
    # Convert to days for readability
    interactions = interactions.with_columns([
        (pl.col('mean_interval_seconds') / (24 * 60 * 60)).alias('mean_interval_days')
    ]).select(['user_id', 'product_id', 'mean_interval_days'])
    
    # Join with target data
    result = target_df.join(interactions, on=['user_id', 'product_id'], how='left')
    
    # Fix 3: Fill NULL values in the numeric feature with a meaningful default
    # Using -1 as it indicates "no previous interval data available"
    result = result.with_columns([
        pl.col('mean_interval_days').fill_null(-1).alias('mean_interval_days')
    ])
    
    return result

# ========== PRODUCT POPULARITY TRENDING = ==========

@FeatureFactory.register('product_popularity_trend')
def generate_product_popularity_trend(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Generate product popularity trend features"""
    # Get the earliest and latest timestamp
    min_time = history_df['timestamp'].min()
    max_time = history_df['timestamp'].max()
    
    # Split history into two time periods
    mid_time = min_time + (max_time - min_time) / 2
    
    early_period = history_df.filter(pl.col('timestamp') < mid_time)
    late_period = history_df.filter(pl.col('timestamp') >= mid_time)
    
    # Calculate product popularity in each period
    def get_period_popularity(df):
        return df.group_by('product_id').agg([
            pl.len().alias('interactions'),
            pl.col('action_type').eq('AT_Purchase').sum().alias('purchases')
        ])
    
    early_popularity = get_period_popularity(early_period)
    late_popularity = get_period_popularity(late_period)
    
    # Join and calculate trend
    popularity_trend = early_popularity.join(
        late_popularity, 
        on='product_id', 
        how='outer',
        suffix='_late'
    ).fill_null(0)
    
    # Calculate popularity trend
    popularity_trend = popularity_trend.with_columns([
        ((pl.col('interactions_late') - pl.col('interactions')) / 
         pl.max_horizontal(pl.lit(1), pl.col('interactions'))
        ).alias('interaction_trend'),
        
        ((pl.col('purchases_late') - pl.col('purchases')) / 
         pl.max_horizontal(pl.lit(1), pl.col('purchases'))
        ).alias('purchase_trend')
    ])
    
    # Join with target data
    return target_df.join(
        popularity_trend.select(['product_id', 'interaction_trend', 'purchase_trend']), 
        on='product_id', 
        how='left'
    )

# ========== CROSS FEATURES = ==========

@FeatureFactory.register('cross_features', depends_on=['user_stats', 'product_stats', 'store_stats', 'city_stats'])
def generate_cross_features(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Generate cross-features (interactions between existing features)"""
    # First ensure the base features exist
    features_needed = ['user_total_purchases', 'product_total_purchases', 
                      'store_total_purchases', 'city_total_purchases']
    
    result = target_df
    
    for feature in features_needed:
        if feature not in result.columns:
            # Generate the missing feature
            if 'user_' in feature:
                result = generate_user_stats(history_df, result)
            elif 'product_' in feature:
                result = generate_product_stats(history_df, result)
            elif 'store_' in feature:
                result = generate_store_stats(history_df, result)
            elif 'city_' in feature:
                result = generate_city_stats(history_df, result)
    
    # Create cross-features
    result = result.with_columns([
        # User-product purchase interaction
        (pl.col('user_total_purchases') * pl.col('product_total_purchases')).alias('user_product_purchase_cross'),
        
        # User-store purchase interaction
        (pl.col('user_total_purchases') * pl.col('store_total_purchases')).alias('user_store_purchase_cross'),
        
        # User-product-store interaction
        (pl.col('user_total_purchases') * pl.col('product_total_purchases') * 
         pl.col('store_total_purchases')).alias('user_product_store_cross')
    ])
    
    return result

# ========== BEHAVIOURAL SEGMENTS ==========

@FeatureFactory.register('user_segments', categorical_cols=['user_segment'])
def generate_user_segments(
    history_df: pl.DataFrame, target_df: pl.DataFrame
) -> pl.DataFrame:
    """Segment users based on their behavior"""
    # Calculate user metrics
    user_metrics = history_df.group_by('user_id').agg([
        pl.len().alias('total_interactions'),
        pl.col('action_type').eq('AT_Purchase').sum().alias('total_purchases'),
        pl.col('action_type').eq('AT_View').sum().alias('total_views'),
        (pl.col('timestamp').max() - pl.col('timestamp').min()).alias('activity_span')
    ])
    
    # Calculate purchase rate
    user_metrics = user_metrics.with_columns([
        (pl.col('total_purchases') / pl.max_horizontal(pl.lit(1), pl.col('total_views'))).alias('purchase_rate'),
        (pl.col('activity_span') / (24 * 60 * 60)).alias('activity_span_days')
    ])
    
    # Define segments
    def assign_segment(row):
        if row['total_interactions'] < 5:
            return 'new_user'
        elif row['purchase_rate'] > 0.2:
            return 'high_converter'
        elif row['total_interactions'] > 20:
            return 'power_user'
        elif row['activity_span_days'] > 30:
            return 'loyal_user'
        else:
            return 'average_user'
    
    user_segments = user_metrics.with_columns([
        pl.struct(['total_interactions', 'purchase_rate', 'activity_span_days'])
          .map_elements(assign_segment, return_dtype=pl.Utf8)
          .alias('user_segment')
    ]).select(['user_id', 'user_segment'])
    
    # Join with target data
    result = target_df.join(user_segments, on='user_id', how='left')
    
    result = result.with_columns([
        pl.col('user_segment').fill_null('unknown_user').alias('user_segment')
    ])
    
    return result

# ========== TARGETS = ==========

@FeatureFactory.register_target('CartUpdate_vs_View')
def target(history_df: pl.DataFrame, target_df: pl.DataFrame) -> pl.Series:
    """Assign 0 for 'AT_View' and 1 for 'AT_CartUpdate'."""
    mapping = {
        'AT_View': 0,
        'AT_CartUpdate': 1,
    }
    return target_df.with_columns(
        target=pl.col("action_type").map_elements(
            lambda x: mapping.get(x, None),
            return_dtype=pl.Int64
        )
    )['target']

@FeatureFactory.register_target('CartUpdate_Purchase_vs_View')
def target(history_df: pl.DataFrame, target_df: pl.DataFrame) -> pl.Series:
    """Assign 0 for 'AT_View' and 1 for 'AT_CartUpdate' and 'AT_Purchase'."""
    mapping = {
        'AT_View': 0,
        'AT_CartUpdate': 1,
        'AT_Purchase': 1,
    }
    return target_df.with_columns(
        target=pl.col("action_type").map_elements(
            lambda x: mapping.get(x, None),
            return_dtype=pl.Int64
        )
    )['target']

@FeatureFactory.register_target('CartUpdate_Purchase_vs_View')
def target(history_df: pl.DataFrame, target_df: pl.DataFrame) -> pl.Series:
    """Assign 0 for 'AT_View' and 1 for 'AT_CartUpdate' and 'AT_Purchase'."""
    mapping = {
        'AT_View': 0,
        'AT_CartUpdate': 1,
        'AT_Purchase': 1,
    }
    return target_df.with_columns(
        target=pl.col("action_type").map_elements(
            lambda x: mapping.get(x, None),
            return_dtype=pl.Int64
        )
    )['target']

@FeatureFactory.register_target('CartUpdate_conversion_aware')
def target(history_df: pl.DataFrame, target_df: pl.DataFrame) -> pl.Series:
    """Assign 0 for 'AT_View' and 1 for 'AT_CartUpdate' and 'AT_Purchase'."""
    conversion_action_types = ["AT_CartUpdate", "AT_Purchase", "AT_Click"]
    target_df = target_df.with_columns(
        conv=pl.col("action_type")
            .is_in(conversion_action_types)
            .any()  # any returns True if at least one conversion event is present
            .over(["request_id", "product_id"])
    )
    target_series = target_df.with_columns(
        target=pl.when(pl.col("action_type").is_in(conversion_action_types))
                    .then(1)
                    .otherwise(
                        pl.when((pl.col("action_type") == "AT_View") & (~pl.col("conv")))
                          .then(0)
                          .otherwise(None)
                    )
    )["target"]
    return target_series
