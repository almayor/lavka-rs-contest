from typing import Tuple

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from lavka_recsys.feature_factory import FeatureFactory

def get_interaction_matrix(df_history: pl.DataFrame) -> Tuple[csr_matrix, dict, dict]:
    """
    Create an csr sparse interaction matrix from the history DataFrame. Each cell indicates the number of times
    a user has purchased a product. The matrix is in the form of (num_users, num_products).
    Also returns mappings user_id -> index and product_id -> index for easy lookup.
    """
    # Assuming your DataFrame is named `df` and has columns "user_id" and "product_id"
    df_counts = (
        df_history.group_by(["user_id", "product_id"])
        .agg(pl.len().alias("purchase_count"))
    )
    unique_users = df_counts["user_id"].unique().to_list()
    unique_products = df_counts["product_id"].unique().to_list()

    # Create mapping dictionaries: user -> index, product -> index
    user2idx = {user: i for i, user in enumerate(unique_users)}
    prod2idx = {prod: i for i, prod in enumerate(unique_products)}

    # Convert columns of the aggregated DataFrame to lists
    user_ids = df_counts["user_id"].to_list()
    product_ids = df_counts["product_id"].to_list()
    purchase_counts = df_counts["purchase_count"].to_list()

    # Build lists of row indices, column indices, and data values for the sparse matrix
    rows = [user2idx[user] for user in user_ids]
    cols = [prod2idx[prod] for prod in product_ids]
    data = [1 for _ in purchase_counts]

    # Determine the shape of the matrix
    num_users = len(unique_users)
    num_products = len(unique_products)

    interaction_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_products))
    return interaction_matrix, user2idx, prod2idx

def get_jaccard_similarity(interaction_matrix: csr_matrix) -> np.ndarray:
    """
    Compute Jaccard similarity between users based on their purchase history.
    
    For a given user u and another user u', the Jaccard similarity is defined as:
    
         Jaccard(u, u') = |I(u) ∩ I(u')| / |I(u) ∪ I(u')|
         
    where I(u) is the set of products purchased by user u.
    
    Parameters:
      interaction_matrix: csr_matrix of shape (n_users, n_products)
          Each entry is the number of times a user purchased a product.
    
    Returns:
      jaccard_similarity: np.ndarray of shape (n_users, n_users)
          The computed Jaccard similarity matrix.
    """
    binary_matrix = interaction_matrix.copy()
    binary_matrix.data = np.ones_like(binary_matrix.data)
    
    intersections = binary_matrix.dot(binary_matrix.T).toarray()
    row_sums = binary_matrix.getnnz(axis=1)
    unions = row_sums[..., np.newaxis] + row_sums[np.newaxis, ...] - intersections
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard_similarity = np.where(unions != 0, intersections / unions, 0) # shape: (n_target_users, n_users)
    
    return jaccard_similarity

# Add methods to FeatureFactory for collaborative filtering
def register_cf_features():

    @FeatureFactory.register('memory-based-cf')
    def compute_memory_based_cf_scores(
        history_df: pl.DataFrame, target_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Computes memory-based collaborative filtering scores.
        The scoring is based on Jaccard similarity computed on binary interaction data between users and products.
        
        For a given target user u and a candidate product p, the score is defined as:
        
            score(u, p) = sum_{u' who purchased p} [ Jaccard(u, u') * purchase_count(u', p) ]
            
        where Jaccard(u, u') = |I(u) ∩ I(u')| / |I(u) ∪ I(u')|.
        
        Parameters:
        interaction_matrix: csr_matrix of shape (n_users, n_products)
            Each entry is the number of times a user purchased a product.
        
        Returns:
        scores: np.ndarray of shape (n_users, n_products)
            The computed memory-based CF scores for each target user and product pair.
        """
        df_interact = history_df.filter(
            pl.col("action_type").is_in(["AT_Purchase", "AT_CartUpdate"])
        )
        interaction_matrix, user2idx, product2idx = get_interaction_matrix(df_interact)
        jaccard_similarity = get_jaccard_similarity(interaction_matrix)
        scores = interaction_matrix.T.dot(jaccard_similarity).T

        def get_scores(row: pl.Series) -> np.ndarray:
            user_idx = user2idx.get(row["user_id"])
            product_idx = product2idx.get(row["product_id"])
            if user_idx is None or product_idx is None:
                return np.nan
            return scores[user_idx, product_idx]

        return target_df.with_columns(
            pl.struct("user_id", "product_id")
                .map_elements(get_scores, return_dtype=pl.Float64)
                .alias("cf_score")
        )
