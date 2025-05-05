import polars as pl

from implicit.bpr import BayesianPersonalizedRanking
from tqdm.auto import tqdm

from ..feature_factory import FeatureFactory
from ..utils.matrix_operations import build_interaction_matrix


def register_bpr_fgens():

    @FeatureFactory.register('bpr-popular', num_cols=['bpr_popular_score'])
    def compute_bpr_popular_scores(
        history_df: pl.DataFrame, target_df: pl.DataFrame
    ) -> pl.DataFrame:
        
        df_interact = history_df.filter(
            pl.col("action_type").is_in(["AT_Purchase", "AT_CartUpdate", "AT_Click"])
        )
        interaction_matrix, user2idx, idx2user, item2idx, idx2item = \
            build_interaction_matrix(df_interact,
                                    user_col="user_id",
                                    item_col="product_id",
                                    binary=True)

        bpr_model = BayesianPersonalizedRanking(
            factors=100,
            random_state=42
        )
        bpr_model.fit(interaction_matrix)

        grouped = target_df.to_pandas().groupby('user_id')
        # Dictionary to store scores for each user_id
        request_scores = {}
        for user_id, group in tqdm(grouped, desc="BPR processing"):
            # Skip cold users
            if user_id not in user2idx:
                continue
            
            # Converting ids to idxs
            user_idx = user2idx[user_id]
            product_ids = group['product_id'].values
            product_idxs = [item2idx[pid] for pid in product_ids if pid in item2idx]
            # Removing duplicate entries
            product_idxs = list(set(product_idxs))

            # Nothing to score for this user
            if not product_idxs:
                continue

            # Get scores for the specified products
            product_idxs, scores = bpr_model.recommend(
                userid=user_idx,
                user_items=interaction_matrix,
                N=len(product_idxs),
                filter_already_liked_items=False,
                items=product_idxs
            )

            # Store the results
            scores_list = [(idx2item[pidx], score) for pidx, score in zip(product_idxs, scores)]
            request_scores[user_id] = scores_list
        
        # Flatten the dictionary into a list of records
        records = []
        for user_id, items in request_scores.items():
            for product_id, score in items:
                records.append({
                    'user_id': user_id,
                    'product_id': product_id,
                    'bpr_popular_score': score
                })

        # Create a Polars DataFrame
        schema = {
            'user_id': pl.UInt64,
            'product_id': pl.UInt64,
            'bpr_popular_score': float
        }
        result_df = pl.DataFrame(records, schema=schema)
        return target_df.join(
            result_df,
            on=['user_id', 'product_id'],
            how='left'
        )
        