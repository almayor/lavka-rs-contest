from ..feature_factory import FeatureFactory
from ..utils.config import Config

import polars as pl
import numpy as np
import holidays

from tqdm.auto import tqdm, trange
from datetime import timedelta

IS_VIEW      = (pl.col("action_type") == "AT_View").cast(pl.Int8)
IS_CLICK     = (pl.col("action_type") == "AT_Click").cast(pl.Int8)
IS_PURCHASE  = pl.col("action_type").is_in(["AT_CartUpdate", "AT_Purchase"]).cast(pl.Int8)
WINDOWS      = {
    "1w":  timedelta(weeks=1),
    "1mo": timedelta(days=30),
    "3mo": timedelta(days=90),
    "6mo": timedelta(days=180),
    "1y":  timedelta(days=365),
}
PAIRS        = {
    'u_p': ['user_id', 'product_id'],
    'u_c': ['user_id', 'product_category'],
    'u_c_source': ['user_id', 'product_category', 'source_type'],
    'u_s': ['user_id', 'store_id'],
    'u_p_source': ['user_id', 'product_id', 'source_type']
}

def register_common_fgens():
    # ========== BASIC FEATURES = ==========
    @FeatureFactory.register('random_noise', num_cols=['random_noise'])
    def generate_random_noise(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Just a meaningless feature that's need for testing and baselines"""
        noise = np.random.rand(target_df.height)  #Values between 0 and 1
        # Add the noise as a new column
        return target_df.with_columns(
            pl.Series(name="random_noise", values=noise)
        )

    @FeatureFactory.register('source_type', cat_cols=['source_type'])
    def generate_source_type(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Source type is already present, we just needed to register it as a categorical feature."""
        return target_df

    @FeatureFactory.register(
        'time_windows',
        num_cols=[f'{name}_{pair_label}_{dur_label}' 
                    for name in ['views', 'clicks', 'purchases', 'interactions', 'ctr', 'purchases_view_ratio']
                    for pair_label in PAIRS.keys()
                    for dur_label in WINDOWS.keys()]
    )
    def generate_window_features(
            history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config,
        ) -> pl.DataFrame:
        t0 = history_df["timestamp"].max()
        out = target_df.clone()

        for pair_label, pair_cols in PAIRS.items():
            for win_label, delta in WINDOWS.items():
                recent = history_df.filter(
                    (pl.col("timestamp") > t0 - delta) &
                    (pl.col("timestamp") <= t0)
                )

                feats = (
                    recent
                    .group_by(pair_cols)
                    .agg([
                        IS_VIEW.sum().alias(
                            f"views_{pair_label}_{win_label}"
                        ),
                        IS_CLICK.sum().alias(
                            f"clicks_{pair_label}_{win_label}"
                        ),
                        IS_PURCHASE.sum().alias(
                            f"purchases_{pair_label}_{win_label}"
                        ),
                        pl.len().alias(
                            f"interactions_{pair_label}_{win_label}"
                        ),
                    ])
                    .with_columns([
                        (pl.col(f"clicks_{pair_label}_{win_label}")
                            .cast(pl.Float64) /
                            (pl.col(f"views_{pair_label}_{win_label}")
                            .cast(pl.Float64) + 1))
                            .alias(f"ctr_{pair_label}_{win_label}"),

                        (pl.col(f"purchases_{pair_label}_{win_label}")
                            .cast(pl.Float64) /
                            (pl.col(f"views_{pair_label}_{win_label}")
                            .cast(pl.Float64) + 1))
                            .alias(f"purchases_view_ratio_{pair_label}_{win_label}"),
                    ])
                )
                out = out.join(feats, on=pair_cols, how="left")

        return out

    @FeatureFactory.register(
        'count_purchase_user_product',
        num_cols=['count_purchase_u_p']
    )
    def generate_count_purchase_user_product(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Count purchases by user-product pairs"""
        return history_df.filter(
            pl.col('action_type') == "AT_CartUpdate"
        ).group_by(
            'user_id', 'product_id'
        ).agg(
            pl.len().alias('count_purchase_u_p')
        ).join(
            target_df,
            on=['user_id', 'product_id'],
            how='right'
        ).fill_null(0)

    @FeatureFactory.register(
        'count_purchase_user_category',
        num_cols=['count_purchase_u_c']
    )
    def generate_count_purchase_user_category(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Count purchases by user-product pairs"""
        return history_df.filter(
            pl.col('action_type') == "AT_CartUpdate"
        ).group_by(
            'user_id', 'product_category'
        ).agg(
            pl.len().alias('count_purchase_u_c')
        ).join(
            target_df,
            on=['user_id', 'product_category'],
            how='right'
        ).fill_null(0)

    @FeatureFactory.register(
        'count_purchase_user_store',
        num_cols=['count_purchase_u_s']
    )
    def generate_count_purchase_user_store(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Count purchases by user-store pairs"""
        return history_df.filter(
            pl.col('action_type') == "AT_CartUpdate"
        ).group_by(
            'user_id', 'store_id'
        ).agg(
            pl.len().alias('count_purchase_u_s')
        ).join(
            target_df,
            on=['user_id', 'store_id'],
            how='right'
        ).fill_null(0)

    @FeatureFactory.register('recency_user_product', num_cols=['days_since_interaction_u_p'])
    def generate_recency_user_product(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate recency features for user-product pairs"""
        latest_time = history_df['timestamp'].max()
        
        feature = history_df.group_by(['user_id', 'product_id']).agg(
            pl.max('timestamp').alias('last_interaction_u_p')
        ).with_columns(
            days_since_interaction_u_p=(latest_time - pl.col('last_interaction_u_p')).dt.total_days()
        )
        return target_df.join(
            feature,
            on=['user_id', 'product_id'],
            how='left'
        ).drop('last_interaction_u_p')
    
    @FeatureFactory.register('recency_user_category', num_cols=['days_since_interaction_u_c'])
    def generate_recency_user_category(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate recency features for user-product pairs"""
        latest_time = history_df['timestamp'].max()
        
        feature = history_df.group_by(['user_id', 'product_category']).agg(
            pl.max('timestamp').alias('last_interaction_u_c')
        ).with_columns(
            days_since_interaction_u_c=(latest_time - pl.col('last_interaction_u_c')).dt.total_days()
        )
        return target_df.join(
            feature,
            on=['user_id', 'product_category'],
            how='left'
        ).drop('last_interaction_u_c')

    @FeatureFactory.register('recency_user_store', num_cols=['days_since_interaction_u_s'])
    def generate_recency_user_store(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate recency features for user-store pairs"""
        latest_time = history_df['timestamp'].max()
        
        feature = history_df.group_by(['user_id', 'store_id']).agg(
            pl.max('timestamp').alias('last_interaction_u_s')
        ).with_columns(
            days_since_interaction_u_s=(latest_time - pl.col('last_interaction_u_s')).dt.total_days()
        )
        return target_df.join(
            feature,
            on=['user_id', 'store_id'],
            how='left'
        ).drop('last_interaction_u_s')

    @FeatureFactory.register(
        'user_stats',
        num_cols=['user_total_interactions', 'user_total_purchases', 'user_total_views', 'user_unique_products']
    )
    def generate_user_stats(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate user-level statistics"""
        feature = history_df.group_by('user_id').agg([
            pl.len().alias('user_total_interactions'),
            pl.col('action_type').eq('AT_CartUpdate').sum().alias('user_total_purchases'),
            pl.col('action_type').eq('AT_View').sum().alias('user_total_views'),
            pl.n_unique('product_id').alias('user_unique_products')
        ])
        return target_df.join(
            feature,
            on=['user_id'],
            how='left'
        )

    @FeatureFactory.register(
        'product_stats',
        num_cols=['product_total_interactions', 'product_total_purchases', 'product_total_views', 'product_unique_users']
    )
    def generate_product_stats(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate product-level statistics"""
        features = history_df.group_by('product_id').agg([
            pl.len().alias('product_total_interactions'),
            pl.col('action_type').is_in(['AT_CartUpdate', 'AT_Purchase']).sum().alias('product_total_purchases'),
            pl.col('action_type').eq('AT_View').sum().alias('product_total_views'),
            pl.n_unique('user_id').alias('product_unique_users')
        ])
        return target_df.join(
            features,
            on=['product_id'],
            how='left'
        )
    @FeatureFactory.register(
        'store_stats',
        num_cols=['store_total_interactions', 'store_total_purchases', 'store_total_views', 'store_unique_products']
    )
    def generate_store_stats(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate store-level statistics"""
        feature = history_df.group_by('store_id').agg([
            pl.len().alias('store_total_interactions'),
            pl.col('action_type').eq('AT_CartUpdate').sum().alias('store_total_purchases'),
            pl.col('action_type').eq('AT_View').sum().alias('store_total_views'),
            pl.n_unique('product_id').alias('store_unique_products')
        ])
        return target_df.join(
            feature,
            on=['store_id'],
            how='left'
        )
    
    @FeatureFactory.register(
        'city_stats',
        num_cols=['city_total_interactions', 'city_total_purchases', 'city_total_views', 'city_unique_stores'],
        cat_cols=['city_name']
    )
    def generate_city_stats(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate city-level statistics"""
        feature = history_df.group_by('city_name').agg([
            pl.len().alias('city_total_interactions'),
            pl.col('action_type').eq('AT_CartUpdate').sum().alias('city_total_purchases'),
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


    @FeatureFactory.register(
        'time_features_cycl_v2',
        num_cols=['hour_of_day_sin_v2', 'hour_of_day_cos_v2', 
                'day_of_week_sin_v2', 'day_of_week_cos_v2', 
                'month_sin_v2', 'month_cos_v2'], # Added cos columns
        cat_cols=['is_weekend_v2']
    )
    def generate_time_features(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate time-related features (hour of day, day of week, etc.)"""
        return target_df.with_columns([
            pl.col('timestamp').dt.hour().mul(2*np.pi/24).sin().alias('hour_of_day_sin_v2'),
            pl.col('timestamp').dt.hour().mul(2*np.pi/24).cos().alias('hour_of_day_cos_v2'),
            pl.col('timestamp').dt.weekday().mul(2*np.pi/7).sin().alias('day_of_week_sin_v2'),
            pl.col('timestamp').dt.weekday().mul(2*np.pi/7).cos().alias('day_of_week_cos_v2'),
            pl.col('timestamp').dt.month().mul(2*np.pi/12).sin().alias('month_sin_v2'),
            pl.col('timestamp').dt.month().mul(2*np.pi/12).cos().alias('month_cos_v2'),
            pl.col('timestamp')
                .dt.weekday()
                .cast(pl.Int32)
                .is_in([6, 7])
                .alias('is_weekend_v2')
        ])

    @FeatureFactory.register(
        'russian_holiday',
        cat_cols=['is_russian_holiday']
    )
    def generate_russian_holidays(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """
        Adds a categorical feature indicating if the date is a Russian public holiday
        using the 'holidays' library and Polars' efficient 'is_in'.
        """
        # Extract date column once
        target_with_date = target_df.with_columns(
            pl.col("timestamp").dt.date().alias("date_only")
        )

        min_year = target_with_date.select(pl.min("date_only").dt.year()).item()
        max_year = target_with_date.select(pl.max("date_only").dt.year()).item()
        
        if min_year is None or max_year is None: # Handle empty target_df case
            return target_df.with_columns(
                pl.lit(0).cast(pl.Int8).alias('is_russian_holiday')
            )

        years_to_check = list(range(min_year, max_year + 1))
        # The holidays object itself can be used for 'in' checks with date objects.
        # Polars' is_in will check if the date_only values are present as keys in ru_holidays.
        ru_holidays = holidays.RU(years=years_to_check) 

        return target_with_date.with_columns(
            pl.col('date_only').is_in(list(ru_holidays.keys())).cast(pl.Int8).alias('is_russian_holiday')
        ).drop('date_only') # Drop the temporary date_only column


    @FeatureFactory.register(
        'product_temporal_patterns',
        num_cols=['avg_purchase_hour', 'std_purchase_hour', 
                  'most_common_purchase_day', 'hour_relevance', 
                  'day_of_week_relevance']
    )
    def generate_product_temporal_patterns(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate features related to typical purchase times and days for products"""
        # Filter to only purchase events
        purchases = history_df.filter(pl.col('action_type') == "AT_CartUpdate")
        
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

       
    @FeatureFactory.register(
        'category_temporal_patterns',
        num_cols=['cat_avg_purchase_hour', 'cat_std_purchase_hour', 
                  'cat_most_common_purchase_day',
                  'cat_hour_relevance', 
                  'cat_day_of_week_relevance']
    )
    def generate_category_temporal_patterns(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate features related to typical purchase times and days for categories"""
        # Filter to only purchase events
        purchases = (
            history_df
            .filter(IS_PURCHASE.eq(1))
            .with_columns(
                pl.col("timestamp").dt.hour().alias("hour_of_day"),
                pl.col("timestamp").dt.weekday().alias("day_of_week"),
            )
        )
        
        hour_stats = purchases.group_by("product_category").agg([
            pl.mean("hour_of_day").alias("cat_avg_purchase_hour"),
            pl.std ("hour_of_day").alias("cat_std_purchase_hour"),
        ])
        common_day = (
            purchases.group_by("product_category")
                .agg(
                    pl.col("day_of_week")
                        .mode()
                        .first()
                        .cast(pl.Int32)
                        .alias("cat_most_common_purchase_day")
                )
        )
        temporal_stats = hour_stats.join(common_day, on="product_category", how="inner")
        result = (
        target_df.join(temporal_stats, on="product_category", how="left")
                 .with_columns([
                     pl.col("timestamp").dt.hour().alias("cur_hour"),
                     pl.col("timestamp").dt.weekday().alias("cur_dow"),
                 ])
                 # circular distance hour:24, dow:7
                 .with_columns([
                     (pl.col("cur_hour") - pl.col("cat_avg_purchase_hour")).abs().alias("h_diff"),
                     (pl.col("cur_dow")  - pl.col("cat_most_common_purchase_day")).abs().alias("d_diff"),
                 ])
                 .with_columns([
                     pl.min_horizontal("h_diff", (24 - pl.col("h_diff"))).alias("h_dist"),
                     pl.min_horizontal("d_diff", (7  - pl.col("d_diff"))).alias("d_dist"),
                 ])
                 .with_columns([
                     (1 - pl.col("h_dist") / 12).alias("cat_hour_relevance"),
                     (1 - pl.col("d_dist") / 3.5).alias("cat_day_of_week_relevance"),
                 ])
                 # drop helpers
                 .drop(["cur_hour", "cur_dow", "h_diff", "d_diff", "h_dist", "d_dist"])
        )
        return result
    
    @FeatureFactory.register(
        'store_temporal_patterns',
        num_cols=['store_avg_purchase_hour', 'store_std_purchase_hour', 
                  'store_most_common_purchase_day',
                  'store_hour_relevance', 
                  'store_day_of_week_relevance']
    )
    def generate_store_temporal_patterns(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate features related to typical purchase times and days for stores"""
        # Filter to only purchase events
        purchases = (
            history_df
            .filter(IS_PURCHASE.eq(1))
            .with_columns(
                pl.col("timestamp").dt.hour().alias("hour_of_day"),
                pl.col("timestamp").dt.weekday().alias("day_of_week"),
            )
        )
        
        hour_stats = purchases.group_by("store_id").agg([
            pl.mean("hour_of_day").alias("store_avg_purchase_hour"),
            pl.std ("hour_of_day").alias("store_std_purchase_hour"),
        ])
        common_day = (
            purchases.group_by("store_id")
                .agg(
                    pl.col("day_of_week")
                        .mode()
                        .first()
                        .cast(pl.Int32)
                        .alias("store_most_common_purchase_day")
                )
        )
        temporal_stats = hour_stats.join(common_day, on="store_id", how="inner")
        result = (
        target_df.join(temporal_stats, on="store_id", how="left")
                 .with_columns([
                     pl.col("timestamp").dt.hour().alias("cur_hour"),
                     pl.col("timestamp").dt.weekday().alias("cur_dow"),
                 ])
                 # circular distance hour:24, dow:7
                 .with_columns([
                     (pl.col("cur_hour") - pl.col("store_avg_purchase_hour")).abs().alias("h_diff"),
                     (pl.col("cur_dow")  - pl.col("store_most_common_purchase_day")).abs().alias("d_diff"),
                 ])
                 .with_columns([
                     pl.min_horizontal("h_diff", (24 - pl.col("h_diff"))).alias("h_dist"),
                     pl.min_horizontal("d_diff", (7  - pl.col("d_diff"))).alias("d_dist"),
                 ])
                 .with_columns([
                     (1 - pl.col("h_dist") / 12).alias("store_hour_relevance"),
                     (1 - pl.col("d_dist") / 3.5).alias("store_day_of_week_relevance"),
                 ])
                 # drop helpers
                 .drop(["cur_hour", "cur_dow", "h_diff", "d_diff", "h_dist", "d_dist"])
        )
        return result
    
    # ========== FREQUENCY FEATURES = ==========

    @FeatureFactory.register('frequency_features', num_cols=['mean_interval_days'])
    def generate_frequency_features(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate frequency-based features"""
        #TODO interactions may span both history and target data!

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
            
            # When working with datetime objects, we need to calculate time differences
            # that result in timedelta objects and then convert to seconds
            intervals = []
            for i in range(1, len(times)):
                # Get difference between consecutive timestamps in seconds
                diff = (times[i] - times[i-1]).total_seconds()
                intervals.append(diff)
            
            return intervals
        
        # Add return_dtype to the first map_elements - intervals will be a list of seconds (float)
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
        
        # Calculate mean interval in seconds
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

        # ========== SESSION FEATURES = ==========

    @FeatureFactory.register(
        'session_features_v2',
        num_cols=[
            'session_length_v2',
            'session_unique_products_v2',
            'session_unique_stores_v2',
            'session_duration_seconds_v2'
        ]
    )
    def generate_session_features(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """
        Generate session-based features.
        A session is defined as a sequence of actions by the same user within a time window.
        The session window duration is configurable via the config.
        """
        session_window_minutes = config.get('feature_config.session_window_minutes', 30)
        window_str = f"{session_window_minutes}m"
        
        combo = (
            pl.concat([history_df, target_df], how="diagonal")
            .sort(["user_id", "timestamp", "request_id"])
            .rolling(
                "timestamp",
                period=window_str,
                closed="left",
                group_by="user_id"
            )
            .agg([
                pl.len().alias("session_length_v2"),
                pl.col("product_id").n_unique().alias("session_unique_products_v2"),
                pl.col("store_id").n_unique().alias("session_unique_stores_v2"),
                pl.col("product_category").n_unique().alias("session_unique_categories_v2"),
                (pl.col("timestamp").max() - pl.col("timestamp").min()).dt.total_seconds().alias("session_duration_seconds_v2")
            ])
            .unique(('user_id', 'timestamp'), keep='last')
        )
        return (
            target_df
            .join(
               combo,
               on=["user_id", "timestamp"],
               how="left" 
            )
        )
        

    # ========== PRODUCT POPULARITY TRENDING = ==========

    @FeatureFactory.register(
        'product_popularity_trend',
        num_cols=['interaction_trend', 'purchase_trend']
    )
    def generate_product_popularity_trend(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
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

        
    @FeatureFactory.register(
        'category_popularity_trend',
        num_cols=['cat_interaction_trend', 'cat_purchase_trend']
    )
    def generate_category_popularity_trend(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate category popularity trend features"""
        # Get the earliest and latest timestamp
        max_time = history_df['timestamp'].max()
        min_time = max_time - timedelta(days=60)
        
        # Split history into two time periods
        mid_time = min_time + (max_time - min_time) / 2
        
        early_period = history_df.filter((pl.col('timestamp') < mid_time) & (pl.col('timestamp') >= min_time))
        late_period = history_df.filter(pl.col('timestamp') >= mid_time)
        
        # Calculate popularity in each period
        def get_period_popularity(df):
            return df.group_by('product_category').agg([
                pl.len().alias('interactions'),
                IS_PURCHASE
                    .sum()
                    .alias('purchases')
            ])
        
        early_popularity = get_period_popularity(early_period)
        late_popularity = get_period_popularity(late_period)
        
        popularity_trend = early_popularity.join(
            late_popularity, 
            on='product_category', 
            how='outer',
            suffix='_late'
        ).fill_null(0)
        
        popularity_trend = popularity_trend.with_columns([
            ((pl.col('interactions_late') - pl.col('interactions')) / 
            pl.max_horizontal(pl.lit(1), pl.col('interactions'))
            ).alias('cat_interaction_trend'),
            
            ((pl.col('purchases_late') - pl.col('purchases')) / 
            pl.max_horizontal(pl.lit(1), pl.col('purchases'))
            ).alias('cat_purchase_trend')
        ])
        
        return target_df.join(
            popularity_trend.select(['product_category', 'cat_interaction_trend', 'cat_purchase_trend']), 
            on='product_category', 
            how='left'
        )
    
    @FeatureFactory.register(
        'store_popularity_trend',
        num_cols=['store_interaction_trend', 'store_purchase_trend']
    )
    def generate_store_popularity_trend(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate store popularity trend features"""
        # Get the earliest and latest timestamp
        max_time = history_df['timestamp'].max()
        min_time = max_time - timedelta(days=60)
        
        # Split history into two time periods
        mid_time = min_time + (max_time - min_time) / 2
        
        early_period = history_df.filter((pl.col('timestamp') < mid_time) & (pl.col('timestamp') >= min_time))
        late_period = history_df.filter(pl.col('timestamp') >= mid_time)
        
        # Calculate product popularity in each period
        def get_period_popularity(df):
            return df.group_by('store_id').agg([
                pl.len().alias('interactions'),
                IS_PURCHASE
                    .sum()
                    .alias('purchases')
            ])
        
        early_popularity = get_period_popularity(early_period)
        late_popularity = get_period_popularity(late_period)
        
        popularity_trend = early_popularity.join(
            late_popularity, 
            on='store_id', 
            how='outer',
            suffix='_late'
        ).fill_null(0)
        
        popularity_trend = popularity_trend.with_columns([
            ((pl.col('interactions_late') - pl.col('interactions')) / 
                pl.max_horizontal(pl.lit(1), pl.col('interactions'))
                ).alias('store_interaction_trend'),
            
            ((pl.col('purchases_late') - pl.col('purchases')) / 
                pl.max_horizontal(pl.lit(1), pl.col('purchases'))
                ).alias('store_purchase_trend')
        ])
        
        return target_df.join(
            popularity_trend.select(['store_id', 'store_interaction_trend', 'store_purchase_trend']), 
            on='store_id', 
            how='left'
        )

    # ========== CROSS FEATURES = ==========

    @FeatureFactory.register(
        'cross_features',
        num_cols=[
            'user_product_purchase_cross',
            'user_store_purchase_cross',
            'user_product_store_cross'
        ],
        depends_on=['user_stats', 'product_stats', 'store_stats', 'city_stats']
    )
    def generate_cross_features(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
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

    @FeatureFactory.register('user_segments', cat_cols=['user_segment'])
    def generate_user_segments(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
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
            pl.col('activity_span').dt.total_days().alias('activity_span_days')
        ])
        
        # Get segment thresholds from config
        config = Config()
        new_user_threshold = config.get('feature_config.user_segments.new_user_threshold', 5)
        high_converter_threshold = config.get('feature_config.user_segments.high_converter_threshold', 0.2)
        power_user_threshold = config.get('feature_config.user_segments.power_user_threshold', 20)
        loyal_user_days = config.get('feature_config.user_segments.loyal_user_days', 30)
        
        # Define segments using config values
        def assign_segment(row):
            if row['total_interactions'] < new_user_threshold:
                return 'new_user'
            elif row['purchase_rate'] > high_converter_threshold:
                return 'high_converter'
            elif row['total_interactions'] > power_user_threshold:
                return 'power_user'
            elif row['activity_span_days'] > loyal_user_days:
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