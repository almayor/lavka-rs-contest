import polars as pl
import scipy as sp


import polars as pl
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, Optional, Any


import polars as pl
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, Optional, Any


def build_interaction_matrix(
    df: pl.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    count_col: Optional[str] = None,
    binary: bool = False
) -> Tuple[csr_matrix, Dict[Any, int], Dict[int, Any], Dict[Any, int], Dict[int, Any]]:
    """
    Builds a user-item interaction matrix in CSR format, along with mappings.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing at least user and item columns.
    user_col : str
        Column name for user IDs.
    item_col : str
        Column name for item IDs.
    count_col : Optional[str]
        Column name for counts/weights. If None, each interaction is counted once.
    binary : bool
        If True, binarize interactions (entry = 1 if count > 0, else 0).

    Returns
    -------
    mat : csr_matrix
        Sparse interaction matrix of shape (n_users, n_items).
    user2idx : Dict[Any, int]
        Mapping from user ID to row index.
    idx2user : Dict[int, Any]
        Mapping from row index back to user ID.
    item2idx : Dict[Any, int]
        Mapping from item ID to column index.
    idx2item : Dict[int, Any]
        Mapping from column index back to item ID.
    """
    # 1) Aggregate counts or occurrences
    if count_col:
        df_counts = (
            df.lazy()
            .group_by([user_col, item_col])
            .agg(pl.col(count_col).alias("count"))
            .collect()
        )
    else:
        df_counts = (
            df.lazy()
            .group_by([user_col, item_col])
            .agg(pl.count().alias("count"))
            .collect()
        )

    # 2) Extract unique users and items
    users = df_counts[user_col].unique().to_list()
    items = df_counts[item_col].unique().to_list()

    # 3) Create mappings
    user2idx = {uid: idx for idx, uid in enumerate(users)}
    idx2user = {idx: uid for uid, idx in user2idx.items()}
    item2idx = {iid: idx for idx, iid in enumerate(items)}
    idx2item = {idx: iid for iid, idx in item2idx.items()}

    # 4) Build matrix components
    rows = [user2idx[uid] for uid in df_counts[user_col].to_list()]
    cols = [item2idx[iid] for iid in df_counts[item_col].to_list()]
    data = df_counts["count"].to_list()

    # 5) Optionally binarize: 1 if any interaction, else 0
    if binary:
        data = [1] * len(data)

    # 6) Construct CSR matrix
    mat = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

    return mat, user2idx, idx2user, item2idx, idx2item
