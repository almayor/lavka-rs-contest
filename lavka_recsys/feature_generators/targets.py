from ..feature_factory import FeatureFactory
from ..utils.config import Config

import polars as pl

def register_target_fgens():
    @FeatureFactory.register_target('CartUpdate_vs_View')
    def target(history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config) -> pl.Series:
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
    def target_cart_update_purchase(history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config) -> pl.Series:
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
        return result.get_column('max_target')

    @FeatureFactory.register_target('CartUpdate_conversion_aware')
    def target(history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config) -> pl.Series:
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
        return result_df.get_column('max_target_interaction_aware')