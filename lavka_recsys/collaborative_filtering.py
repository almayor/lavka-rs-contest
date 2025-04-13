import math
from typing import Tuple

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from lavka_recsys.feature_factory import FeatureFactory

def get_npmi_matrix(interaction_matrix: csr_matrix) -> np.ndarray:
    """
    Compute Normalized Pointwise Mutual Information (NPMI) between items.
    
    NPMI is defined as:
    NPMI(x, y) = PMI(x, y) / -log(P(x, y))
    
    where PMI(x, y) = log(P(x, y) / (P(x) * P(y)))
    
    Parameters:
        interaction_matrix: csr_matrix of shape (n_users, n_products)
            Binary matrix where each entry indicates if a user has purchased a product.
            
    Returns:
        npmi_matrix: np.ndarray of shape (n_products, n_products)
            The computed NPMI similarity matrix between products.
    """
    # Convert to binary matrix
    binary_matrix = interaction_matrix.copy()
    binary_matrix.data = np.ones_like(binary_matrix.data)
    
    n_users = binary_matrix.shape[0]
    n_products = binary_matrix.shape[1]
    
    # Compute item occurrence probabilities P(x)
    item_counts = np.array(binary_matrix.sum(axis=0)).flatten()
    item_probs = item_counts / n_users
    
    # Compute co-occurrence matrix (item-item)
    co_occurrence = binary_matrix.T @ binary_matrix
    co_occurrence_array = co_occurrence.toarray()
    
    # Compute co-occurrence probabilities P(x,y)
    co_occurrence_probs = co_occurrence_array / n_users
    
    # Compute PMI and NPMI matrices
    npmi_matrix = np.zeros((n_products, n_products))
    
    # Using broadcasting for vectorized operations
    outer_probs = np.outer(item_probs, item_probs)
    
    # Calculate PMI: log(P(x,y) / (P(x) * P(y)))
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log(co_occurrence_probs / outer_probs)
        
        # Normalize PMI to get NPMI
        denominator = -np.log(co_occurrence_probs)
        npmi_matrix = pmi / denominator
    
    # Replace NaN and inf values
    npmi_matrix[np.isnan(npmi_matrix) | np.isinf(npmi_matrix)] = 0
    
    # Set diagonal elements to 1 (maximum similarity with self)
    np.fill_diagonal(npmi_matrix, 1)
    
    return npmi_matrix


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

def get_npmi_item_scores(
    interaction_matrix: csr_matrix, 
    npmi_matrix: np.ndarray, 
    user_idx: int
) -> np.ndarray:
    """
    Compute recommendation scores for a user based on NPMI item similarities.
    
    Parameters:
        interaction_matrix: csr_matrix of shape (n_users, n_products)
            Binary matrix where each entry indicates if a user has purchased a product.
        npmi_matrix: np.ndarray of shape (n_products, n_products)
            NPMI similarity matrix between items.
        user_idx: int
            Index of the target user.
            
    Returns:
        scores: np.ndarray of shape (n_products,)
            Recommendation scores for each product for the target user.
    """
    # Get items purchased by the user
    user_items = interaction_matrix[user_idx].nonzero()[1]
    
    if len(user_items) == 0:
        return np.zeros(interaction_matrix.shape[1])
    
    # Compute scores as the sum of NPMI similarities to items the user has purchased
    scores = npmi_matrix[:, user_items].sum(axis=1)
    
    # Zero out scores for items the user has already purchased
    scores[user_items] = 0
    
    return scores


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

    @FeatureFactory.register('npmi-cf')
    def compute_npmi_cf_scores(
        history_df: pl.DataFrame, target_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Computes NPMI-based collaborative filtering scores.
        
        For a given target user u and a candidate product p, the score is defined as:
        
            score(u, p) = sum_{p' in items purchased by u} NPMI(p, p')
            
        where NPMI(p, p') is the normalized pointwise mutual information between products p and p'.
        
        Parameters:
            history_df: pl.DataFrame
                DataFrame containing user-product interaction history.
            target_df: pl.DataFrame
                DataFrame containing user-product pairs to score.
                
        Returns:
            target_df: pl.DataFrame
                The input DataFrame with an additional column 'npmi_cf_score'.
        """
        # Filter to relevant interactions
        df_interact = history_df.filter(
            pl.col("action_type").is_in(["AT_Purchase", "AT_CartUpdate"])
        )
        
        # Get interaction matrix and mappings
        interaction_matrix, user2idx, product2idx = get_interaction_matrix(df_interact)
        
        # Compute NPMI matrix
        npmi_matrix = get_npmi_matrix(interaction_matrix)
        
        # Precompute scores for all users
        all_user_scores = {}
        for user_id, user_idx in user2idx.items():
            all_user_scores[user_id] = get_npmi_item_scores(interaction_matrix, npmi_matrix, user_idx)
        
        def get_score(row: pl.Series) -> float:
            user_id = row["user_id"]
            product_id = row["product_id"]
            
            user_idx = user2idx.get(user_id)
            product_idx = product2idx.get(product_id)
            
            if user_idx is None or product_idx is None:
                return np.nan
                
            return all_user_scores[user_id][product_idx]
        
        # Add NPMI scores to target DataFrame
        return target_df.with_columns(
            pl.struct("user_id", "product_id")
                .map_elements(get_score, return_dtype=pl.Float64)
                .alias("npmi_cf_score")
        )
