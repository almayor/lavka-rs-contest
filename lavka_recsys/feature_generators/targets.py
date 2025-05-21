from ..feature_factory import FeatureFactory
from ..utils.config import Config
from ..utils.custom_logging import get_logger

import polars as pl

_logger_instance = get_logger(__name__) 


def _get_mask(target_df: pl.DataFrame, config: Config) -> pl.Series:
    """Returns a mask that would mask rows in the target_df which are acceptable as targets"""
    min_products_in_request = config.get('target.cleaning.min_products_in_request', -1)
    remove_view_only_requests = config.get('target.cleaning.remove_view_only_requests', False)
    accepted_source_types = config.get('target.cleaning.source_types', None)

    if min_products_in_request > 0:
        _logger_instance.info(f"Removing requests with <{min_products_in_request} unique products")
    if remove_view_only_requests:
        _logger_instance.info(f"Removing requests with only AT_View actions")
    if accepted_source_types is not None:
        _logger_instance.info(f"Ignoring rows with source types other than {', '.join(accepted_source_types)}")

    accepted_request_ids = set(
        target_df.group_by('request_id')
        .agg([
            pl.col("action_type").ne('AT_View').any().alias("has_non_view"),
            pl.col("product_id").n_unique().alias("unique_products")  
        ])
        .filter(
            pl.col("unique_products").ge(min_products_in_request) &
            (pl.col('has_non_view') if remove_view_only_requests else pl.lit(True))
        )
        .get_column('request_id')
        .unique()
    )
    mask = target_df.select(
        pl.col("request_id").is_in(accepted_request_ids).alias('mask')
    ).get_column('mask')

    accepted_source_types = config.get('target.cleaning.source_types', None)
    if accepted_source_types is not None:
        mask &= target_df.select(
            pl.col("source_type").is_in(accepted_source_types).alias('mask')
        ).get_column('mask')
    
    return mask
    
def register_target_fgens():
    @FeatureFactory.register_target('CartUpdate_vs_View')
    def target_cartupdate_view(history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config) -> pl.Series:
        """Assign 0 for 'AT_View' and 1 for 'AT_CartUpdate'."""
        mapping = {
            'AT_View': 0,
            'AT_CartUpdate': 1,
        }
        target_series = target_df.with_columns(
            target=pl.col("action_type").map_elements(
                lambda x: mapping.get(x, None),
                return_dtype=pl.Int64
            )
        )['target']

        if config.get('target.cleaning.enabled', False):
            mask = _get_mask(target_df, config)
            target_series[mask] = None
        return target_series

    @FeatureFactory.register_target('CartUpdate_Purchase_vs_View')
    def target_cartupdate_purchase(history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config) -> pl.Series:
        """Assign 0 for 'AT_View' and 1 for 'AT_CartUpdate' and 'AT_Purchase'."""
        mapping = {
            'AT_View': 0,
            'AT_CartUpdate': 1,
            'AT_Purchase': 1,
        }
        target_series = target_df.with_columns(
            target=pl.col("action_type").map_elements(
                lambda x: mapping.get(x, None),
                return_dtype=pl.Int64
            )
        )['target']

        if config.get('target.cleaning.enabled', False):
            mask = _get_mask(target_df, config)
            target_series[mask] = None
        return target_series
    
    @FeatureFactory.register_target('Weighted')
    def target_weighted(history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config) -> pl.Series:
        """Assign different weights for 'AT_View', 'AT_CartUpdate', 'AT_Purchase', 'AT_Click'."""
        mapping = {
            'AT_View': 0.0,
            'AT_CartUpdate': 0.85,
            'AT_Purchase': 1.0,
            'AT_Click': 0.3,
        }
        weighted = target_df.with_columns(
            pl.col('action_type').replace_strict(mapping, default=0.0).alias('target')
        )
        grouped = weighted.group_by(['request_id', 'product_id']).agg(
            pl.col('target').max().alias('max_target')
        )
        result = target_df.join(grouped, on=['request_id', 'product_id'], how='left')
        target_series = result.get_column('max_target')

        if config.get('target.cleaning.enabled', False):
            mask = _get_mask(target_df, config)
            target_series[mask] = None
        return target_series

    @FeatureFactory.register_target('CartUpdate_conversion_aware')
    def target_cartupdate_conversion_aware(history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config) -> pl.Series:
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
        if config.get('target.cleaning.enabled', False):
            mask = _get_mask(target_df, config)
            target_series[mask] = None
        return target_series
    
    @FeatureFactory.register_target('Weighted_InteractionAware')
    def target_weighted_interaction_aware(history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config) -> pl.Series:
        """
        Assigns weights to actions, differentiating views based on whether other
        items in the same request_id had a positive interaction.
        - AT_Purchase: 1.0
        - AT_CartUpdate: 0.85
        - AT_Click: 0.3
        - AT_View (in request with NO other positive interactions): 0.15
        - AT_View (in request WITH another positive interaction): 0.05
        """
        positive_interaction_types = ['AT_CartUpdate', 'AT_Purchase', 'AT_Click']
        
        base_mapping = {
            'AT_View': 0.0, # placeholder, will be overridden by nuanced logic
            'AT_CartUpdate': 0.85,
            'AT_Purchase': 1.0,
            'AT_Click': 0.3,
        }

        view_no_other_interaction_value = 0.15
        view_with_other_interaction_value = 0.05

        requests_with_pos_interaction = (
            target_df
            .filter(pl.col("action_type").is_in(positive_interaction_types))
            .select("request_id")
            .unique()
        )

        weighted_df = target_df.with_columns([
            pl.col('action_type').replace_strict(base_mapping, default=0.0).alias('base_target'),
            pl.col('request_id').is_in(requests_with_pos_interaction['request_id']).alias('request_had_pos_interaction')
        ])

        final_target_df = weighted_df.with_columns(
            target=pl.when(pl.col('action_type').is_in(positive_interaction_types))
                     .then(pl.col('base_target')) # base weight for positive interactions
                     .when((pl.col('action_type') == 'AT_View') & (pl.col('request_had_pos_interaction')))
                     .then(pl.lit(view_with_other_interaction_value, dtype=pl.Float64)) # viewed-but-ignored
                     .when((pl.col('action_type') == 'AT_View') & (~pl.col('request_had_pos_interaction')))
                     .then(pl.lit(view_no_other_interaction_value, dtype=pl.Float64)) # standard view
                     .otherwise(0.0) # default for any other unhandled cases (should ideally not happen with strict mapping)
        )
        
        # aggregate per (request_id, product_id) taking the max target
        grouped = final_target_df.group_by(['request_id', 'product_id']).agg(
            pl.col('target').max().alias('max_target_interaction_aware')
        )
        
        result_df = target_df.join(grouped, on=['request_id', 'product_id'], how='left')
        target_series = result_df.get_column('max_target_interaction_aware')

        if config.get('target.cleaning.enabled', False):
            mask = _get_mask(target_df, config)
            target_series[mask] = None
        return target_series
