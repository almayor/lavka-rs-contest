from ..feature_factory import FeatureFactory

import polars as pl

def register_target_fgens():
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
    def target_cart_update_purchase(history_df: pl.DataFrame, target_df: pl.DataFrame) -> pl.Series:
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
    def target_weighted(history_df: pl.DataFrame, target_df: pl.DataFrame) -> pl.Series:
        """Assign different weights for 'AT_View', 'AT_CartUpdate', 'AT_Purchase', 'AT_Click'."""
        mapping = {
            'AT_View': 0,
            'AT_CartUpdate': 0.85,
            'AT_Purchase': 1,
            'AT_Click': 0.3,
        }
        return (
            target_df
            .with_columns(
                pl.col('action_type').replace_strict(mapping, default=0).alias('target')
            )
            .group_by(['request_id', 'product_id'])
            .agg(
                #taking maximum score
                pl.col('target').max().alias('target')
            )
            .get_column('target')
        )

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