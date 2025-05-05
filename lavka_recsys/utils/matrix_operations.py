import polars as pl
import scipy as sp


def build_matrix_with_mappings(ratings: pl.DataFrame, additive: bool = False):
    """
    Constructs a sparse user-item rating matrix from a Polars DataFrame and returns it 
    along with index mappings for users and items.
    Based on https://github.com/yandexdataschool/recsys_course.

    Parameters:
        ratings (pl.DataFrame): A DataFrame containing at least 'user_id', 'item_id', and 'rating' columns.
        additive (bool, optional): If True, ratings for the same (user, item) pair are summed. 
                                   If False (default), the last rating is used.

    Returns:
        tuple: A tuple containing:
            - R (scipy.sparse.lil_array): Sparse user-item rating matrix.
            - mappings (tuple): A tuple of four dictionaries:
                * user_id2idx: Maps user IDs to matrix row indices.
                * item_id2idx: Maps item IDs to matrix column indices.
                * user_idx2id: Reverse mapping from indices to user IDs.
                * item_idx2id: Reverse mapping from indices to item IDs.
    """
    mappings = build_mappings(ratings)
    user_id2idx, item_id2idx, _, _ = mappings
    num_users = len(user_id2idx)
    num_items = len(item_id2idx)
    R = sp.sparse.lil_array((num_users, num_items))
    for row in ratings.iter_rows(named=True):
        user_idx = user_id2idx[row["user_id"]]
        item_idx = item_id2idx[row["item_id"]]
        if additive:
            R[user_idx, item_idx] += row["rating"]
        else:
            R[user_idx, item_idx] = row["rating"]
    return R, mappings

def build_mappings(ratings: pl.DataFrame):
    """
    Generates mapping dictionaries to convert between user/item IDs and matrix indices.
    Based on https://github.com/yandexdataschool/recsys_course.

    Parameters:
        ratings (pl.DataFrame): A DataFrame containing 'user_id' and 'item_id' columns.

    Returns:
        tuple: A tuple of four dictionaries:
            - user_id2idx: Maps user IDs to matrix row indices.
            - item_id2idx: Maps item IDs to matrix column indices.
            - user_idx2id: Reverse mapping from row indices to user IDs.
            - item_idx2id: Reverse mapping from column indices to item IDs.
    """
    users = ratings.select(pl.col("user_id").unique())
    items = ratings.select(pl.col("item_id").unique())
    user_id2idx = {row["user_id"]: i for i, row in enumerate(users.iter_rows(named=True))}
    item_id2idx = {row["item_id"]: i for i, row in enumerate(items.iter_rows(named=True))}
    user_idx2id = {v: k for k, v in user_id2idx.items()}
    item_idx2id = {v: k for k, v in item_id2idx.items()}
    return user_id2idx, item_id2idx, user_idx2id, item_idx2id
