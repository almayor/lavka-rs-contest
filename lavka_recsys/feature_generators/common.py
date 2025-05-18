from ..feature_factory import FeatureFactory
from ..utils.config import Config

import polars as pl
import numpy as np
import holidays

IS_VIEW      = (pl.col("action_type") == "AT_View").cast(pl.Int8)
IS_CLICK     = (pl.col("action_type") == "AT_Click").cast(pl.Int8)
IS_PURCHASE  = pl.col("action_type").is_in(["AT_CartUpdate", "AT_Purchase"]).cast(pl.Int8)
ONE          = pl.lit(1, dtype=pl.Int8)
WINDOWS      = {"1w": "1w",
                "1mo": "1mo",
                "3mo": "3mo",
                "1y": "1y"}
PAIRS        = {'u_p': ['user_id', 'product_id'],
                'u_c': ['user_id', 'product_category'],
                'u_c_source': ['user_id', 'product_category', 'source_type'],
                'u_s': ['user_id', 'store_id'],
                'u_p_source': ['user_id', 'product_id', 'source_type']}


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
                    for name in ['views', 'clicks', 'purchases', 'interactions', 'ctr', 'purchase_view_ratio']
                    for pair_label in PAIRS.keys()
                    for dur_label in WINDOWS.keys()]
    )
    def generate_window_features(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate time-window based features"""

        combo = pl.concat([history_df, target_df], how="diagonal") \
                  .sort(["user_id", "product_id", "source_type", "timestamp"])
        

        exprs = []
        for pair_label, pair_cols in PAIRS.items():
            for win_label, win_len in WINDOWS.items():
                views      = IS_VIEW    .rolling_sum_by("timestamp", window_size=win_len, closed="both").over(pair_cols)
                clicks     = IS_CLICK   .rolling_sum_by("timestamp", window_size=win_len, closed="both").over(pair_cols)
                purchases  = IS_PURCHASE.rolling_sum_by("timestamp", window_size=win_len, closed="both").over(pair_cols)
                inters     = ONE        .rolling_sum_by("timestamp", window_size=win_len, closed="both").over(pair_cols)

                exprs.extend([
                    views     .alias(f"views_{pair_label}_{win_label}"),
                    clicks    .alias(f"clicks_{pair_label}_{win_label}"),
                    purchases .alias(f"purchases_{pair_label}_{win_label}"),
                    inters    .alias(f"interactions_{pair_label}_{win_label}"),
                    (clicks   / (views.cast(pl.Float64) + 1)).alias(f"ctr_{pair_label}_{win_label}"),
                    (purchases/ (views.cast(pl.Float64) + 1)).alias(f"purchase_view_ratio_{pair_label}_{win_label}"),
                ])

        combo = combo.with_columns(exprs)
        return combo.tail(target_df.height)

    @FeatureFactory.register(
        'recency',
        num_cols=[f'{name}_{pair_label}'
                  for pair_label in PAIRS.keys()
                  for name in ['days_since_interaction', 'mean_days_since_interaction']]
    )
    def generate_recency(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate recency features"""
        combo = (
            pl.concat([history_df, target_df], how="diagonal")
            .sort(["timestamp"])
        )
        recency_exprs = []
        for label, cols in PAIRS.items():
            prev_ts = pl.col("timestamp").shift(1).over(cols)
            recency = (pl.col("timestamp") - prev_ts).dt.total_days()
            recency_exprs.append(recency.alias(f"days_since_interaction_{label}"))
        
        combo = combo.with_columns(recency_exprs)

        for label, cols in PAIRS.items():
            mean_gap = (
                history_df.sort(cols + ["timestamp"])
                        .with_columns(
                            (pl.col("timestamp") - pl.col("timestamp").shift(1).over(cols))
                            .dt.total_days()
                            .alias("gap")
                        )
                        .group_by(cols)
                        .agg(pl.col("gap").drop_nulls().mean().alias(f"mean_days_since_interaction_{label}"))
            )

            combo = combo.join(mean_gap, on=cols, how="left")

        return combo.tail(target_df.height)
    

    @FeatureFactory.register(
        'recency_purchase',
        num_cols=[f'{name}_{pair_label}'
                  for pair_label in PAIRS.keys()
                  for name in ['days_since_purchase', 'mean_days_since_purchase']]
    )
    def generate_recency_purchase(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate recency-since-purchase features"""
        combo = (
            pl.concat([history_df, target_df], how="diagonal")
            .sort(["timestamp"])             
        )

        recency_exprs = []
        for lbl, cols in PAIRS.items():
            last_purch_ts = (
                pl.when(IS_PURCHASE).then(pl.col("timestamp")).otherwise(None)
                .forward_fill()                    # keep “latest seen purchase”
                .over(cols)
                .shift(1)                          # … but exclude *this* row
                .over(cols)
            )

            recency_exprs.append(
                (pl.col("timestamp") - last_purch_ts)
                    .dt.total_days()
                    .alias(f"days_since_purchase_{lbl}")
            )

        combo = combo.with_columns(recency_exprs)

        for lbl, cols in PAIRS.items():
            mean_gap = (
                history_df
                .filter(IS_PURCHASE)
                .sort(cols + ["timestamp"])
                .with_columns(
                    (pl.col("timestamp") - pl.col("timestamp").shift(1).over(cols))
                        .dt.total_days()
                        .alias("gap")
                )
                .group_by(cols)
                .agg(
                    pl.col("gap").drop_nulls().mean().alias(f"mean_days_since_purchase_{lbl}")
                )
            )

            combo = combo.join(mean_gap, on=cols, how="left")

        return combo.tail(target_df.height)


    @FeatureFactory.register(
        'user_stats',
        num_cols=['user_total_purchases', 'user_total_views', 'user_unique_products']
    )
    def generate_user_stats(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate user-level statistics"""
        feature = history_df.group_by('user_id').agg([
            pl.len().alias('user_total_interactions'),
            IS_PURCHASE.sum().alias('user_total_purchases'),
            IS_VIEW.sum().alias('user_total_views'),
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
            IS_PURCHASE.sum().alias('product_total_purchases'),
            IS_VIEW.sum().alias('product_total_views'),
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
            IS_PURCHASE.sum().alias('store_total_purchases'),
            IS_VIEW.sum().alias('store_total_views'),
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
            IS_PURCHASE.sum().alias('city_total_purchases'),
            IS_VIEW.eq('AT_View').sum().alias('city_total_views'),
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
        'time_features_cycl',
        num_cols=['hour_of_day_sin', 'hour_of_day_cos', 
                'day_of_week_sin', 'day_of_week_cos', 
                'month_sin', 'month_cos'], # Added cos columns
        cat_cols=['is_weekend']
    )
    def generate_time_features(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate time-related features (hour of day, day of week, etc.)"""
        return target_df.with_columns([
            pl.col('timestamp').dt.hour().mul(2*np.pi/24).sin().alias('hour_of_day_sin'),
            pl.col('timestamp').dt.hour().mul(2*np.pi/24).cos().alias('hour_of_day_cos'),
            pl.col('timestamp').dt.weekday().mul(2*np.pi/7).sin().alias('day_of_week_sin'),
            pl.col('timestamp').dt.weekday().mul(2*np.pi/7).cos().alias('day_of_week_cos'),
            pl.col('timestamp').dt.month().mul(2*np.pi/12).sin().alias('month_sin'),
            pl.col('timestamp').dt.month().mul(2*np.pi/12).cos().alias('month_cos'),
            pl.col('timestamp')
                .dt.weekday()
                .cast(pl.Int32)
                .is_in([6, 7])
                .alias('is_weekend')
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
        
        if min_year is None or max_year is None:
            return target_df.with_columns(
                pl.lit(0).cast(pl.Int8).alias('is_russian_holiday')
            )

        years_to_check = list(range(min_year, max_year + 1))
        ru_holidays = holidays.RU(years=years_to_check) 

        return target_with_date.with_columns(
            pl.col('date_only').is_in(list(ru_holidays.keys())).cast(pl.Int8).alias('is_russian_holiday')
        ).drop('date_only') # Drop the temporary date_only column


    @FeatureFactory.register(
        'product_temporal_patterns',
        num_cols=['avg_purchase_hour', 'std_purchase_hour', 
                  'most_common_purchase_day',
                  'hour_relevance', 
                  'day_of_week_relevance']
    )
    def generate_product_temporal_patterns(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate features related to typical purchase times and days for products"""
        # Filter to only purchase events
        purchases = (
            history_df
            .filter(IS_PURCHASE)
            .with_columns(
                pl.col("timestamp").dt.hour().alias("hour_of_day"),
                pl.col("timestamp").dt.weekday().alias("day_of_week"),
            )
        )
        
        hour_stats = purchases.group_by("product_id").agg([
            pl.mean("hour_of_day").alias("avg_purchase_hour"),
            pl.std ("hour_of_day").alias("std_purchase_hour"),
        ])
        common_day = (
            purchases.group_by("product_id")
                .agg(
                    pl.col("day_of_week")
                    .mode()
                    .alias("most_common_purchase_day")
                )
        )
        temporal_stats = hour_stats.join(common_day, on="product_id", how="inner")
        result = (
        target_df.join(temporal_stats, on="product_id", how="left")
                 .with_columns([
                     pl.col("timestamp").dt.hour().alias("cur_hour"),
                     pl.col("timestamp").dt.weekday().alias("cur_dow"),
                 ])
                 # circular distance hour:24, dow:7
                 .with_columns([
                     (pl.col("cur_hour") - pl.col("avg_purchase_hour")).abs().alias("h_diff"),
                     (pl.col("cur_dow")  - pl.col("most_common_purchase_day")).abs().alias("d_diff"),
                 ])
                 .with_columns([
                     pl.min_horizontal("h_diff", (24 - pl.col("h_diff"))).alias("h_dist"),
                     pl.min_horizontal("d_diff", (7  - pl.col("d_diff"))).alias("d_dist"),
                 ])
                 .with_columns([
                     (1 - pl.col("h_dist") / 12).alias("hour_relevance"),
                     (1 - pl.col("d_dist") / 3.5).alias("day_of_week_relevance"),
                 ])
                 # drop helpers
                 .drop(["cur_hour", "cur_dow", "h_diff", "d_diff", "h_dist", "d_dist"])
        )
        return result
    

    # ========== SESSION FEATURES = ==========

    @FeatureFactory.register(
        'session_features',
        num_cols=[
            'session_length',
            'session_unique_products',
            'session_unique_stores',
            'session_duration_seconds'
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
                closed="both",
                group_by="user_id"
            )
            .agg([
                pl.len().alias("session_length"),
                pl.col("product_id").n_unique().alias("session_unique_products"),
                pl.col("product_id").n_unique().alias("session_unique_stores"),
                pl.col("product_id").n_unique().alias("session_unique_categories"),
                (pl.col("timestamp").max() - pl.col("timestamp").min()).dt.total_seconds().alias("session_duration_seconds")
            ]) 
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
                pl.col('action_type')
                    .is_in(['AT_Purchase', 'AT_Click', 'AT_CartUpdate'])
                    .sum()
                    .alias('purchases')
            ])
        
        early_popularity = get_period_popularity(early_period)
        late_popularity = get_period_popularity(late_period)
        
        popularity_trend = early_popularity.join(
            late_popularity, 
            on='product_id', 
            how='outer',
            suffix='_late'
        ).fill_null(0)
        
        popularity_trend = popularity_trend.with_columns([
            ((pl.col('interactions_late') - pl.col('interactions')) / 
            pl.max_horizontal(pl.lit(1), pl.col('interactions'))
            ).alias('interaction_trend'),
            
            ((pl.col('purchases_late') - pl.col('purchases')) / 
            pl.max_horizontal(pl.lit(1), pl.col('purchases'))
            ).alias('purchase_trend')
        ])
        
        return target_df.join(
            popularity_trend.select(['product_id', 'interaction_trend', 'purchase_trend']), 
            on='product_id', 
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
        
        result = target_df.with_columns([
            (pl.col('user_total_purchases') * pl.col('product_total_purchases')).alias('user_product_purchase_cross'),
            (pl.col('user_total_purchases') * pl.col('store_total_purchases')).alias('user_store_purchase_cross'),
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