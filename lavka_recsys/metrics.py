import numpy as np
from sklearn.metrics import log_loss, ndcg_score, roc_auc_score
import json
import logging
import os
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.metrics import log_loss, ndcg_score, roc_auc_score
from tqdm.auto import tqdm

class RankingMetrics:
    """Ranking metrics for recommender systems"""
    
    @staticmethod
    def _convert_to_matrix(true_relevance: pl.Series, predicted_scores: pl.Series, group_idx: pl.Series, k=10):
        """
        Convert to padded matrices, where each row corresponds to a group id.
        """
        pred_df = pl.DataFrame({
            'request_id': group_idx,
            'target': true_relevance,
            'predict': predicted_scores
        })
        
        true_li = []
        pred_li = []
        
        # Process each group efficiently
        for _, group in pred_df.group_by('request_id'):
            # Skip if group is empty or has no positive examples
            if len(group) == 0 or group['target'].sum() == 0:
                continue
                
            # Sort by prediction score in descending order for the predicted rankings
            # This is what the model would rank items as
            sorted_by_pred = group.sort('predict', descending=True).head(k)
            
            # Get values as numpy arrays
            t_values = sorted_by_pred['target'].to_numpy()
            p_values = sorted_by_pred['predict'].to_numpy()
            
            # Pad to length k if needed
            padding = k - len(t_values)
            if padding > 0:
                t_values = np.pad(t_values, (0, padding), mode='constant', constant_values=0)
                p_values = np.pad(p_values, (0, padding), mode='constant', constant_values=0)
            
            true_li.append(t_values)
            pred_li.append(p_values)
        
        # Return matrices or None if no valid groups
        if not true_li:
            return None, None
        else:
            return np.array(true_li), np.array(pred_li)

    @staticmethod
    def map_at_k(true_relevance, predicted_scores, group_idx, k=10):
        """
        Calculate MAP@K (Mean Average Precision at K)
        
        Args:
            true_relevance: List of binary relevance values
            predicted_scores: List of predicted scores
            group_idx: List of group id for each row
            k: Cutoff for evaluation
        
        Returns:
            MAP@K score
        """
        def ap_at_k(y_true, y_pred, k):
            """Calculate AP@K for a single query"""
            if np.sum(y_true) == 0:
                return 0.0
                
            sorted_indices = np.argsort(y_pred)[::-1]
            top_k_indices = sorted_indices[:k]
            y_true_k = np.array(y_true)[top_k_indices]

            cumulative_precision = 0.0
            relevant_seen = 0
            
            for i in range(len(y_true_k)):
                if y_true_k[i]:
                    relevant_seen += 1
                    precision_at_i = relevant_seen / (i + 1)
                    cumulative_precision += precision_at_i

            return cumulative_precision / max(1, np.sum(y_true))
        
        # Calculate AP@K for each query
        aps = [ap_at_k(y_true, y_pred, k) 
              for y_true, y_pred in zip(true_relevance, predicted_scores)]
        
        # Return mean of AP@K values
        return np.mean(aps)
    
    @staticmethod
    def ndcg_at_k(true_relevance, predicted_scores, group_idx, k=10):
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain at K)
        
        Args:
            true_relevance: List of binary or graded relevance values
            predicted_scores: List of predicted scores
            group_idx: List of group id for each row
            k: Cutoff for evaluation
        
        Returns:
            NDCG@K score
        """
        true_array, pred_array = RankingMetrics._convert_to_matrix(
            true_relevance,
            predicted_scores,
            group_idx,
            k=k
        )
        
        # Return 0 if no valid groups found
        if true_array is None or pred_array is None or len(true_array) == 0:
            return 0.0
            
        # Calculate nDCG using sklearn
        try:
            return ndcg_score(true_array, pred_array, k=k)
        except Exception as e:
            # In case of error, return 0
            return 0.0
    
    @staticmethod
    def novelty_at_k(recommendations, popularity_df, k=10):
        """
        Calculate Novelty@K
        
        Args:
            recommendations: DataFrame with user_id, product_id, and predicted scores
            popularity_df: DataFrame with product_id and popularity
            k: Cutoff for evaluation
        
        Returns:
            Novelty@K score
        """
        # Join recommendations with popularity
        recs_with_pop = recommendations.join(
            popularity_df, on='product_id', how='left'
        ).fill_null(0)
        
        # Group by user and get top-K recommendations
        novelty_scores = []
        
        for user_id, user_recs in recs_with_pop.group_by('user_id'):
            # Sort by predicted score and take top K
            top_k = user_recs.sort('predict', descending=True).head(k)
            
            # Calculate novelty as 1 - popularity
            novelty = (1 - top_k['popularity']).mean()
            novelty_scores.append(novelty)
        
        # Return mean novelty across all users
        return np.mean(novelty_scores)
    
    @staticmethod
    def serendipity_at_k(recommendations, user_history, k=10):
        """
        Calculate Serendipity@K
        
        Args:
            recommendations: DataFrame with user_id, product_id, and predicted scores
            user_history: DataFrame with user purchase history
            k: Cutoff for evaluation
        
        Returns:
            Serendipity@K score
        """
        # Join recommendations with user history
        recs_with_history = recommendations.join(
            user_history, on=['user_id', 'product_id'], how='left'
        ).with_columns(
            has_purchased=pl.col('has_purchased').fill_null(0)
        )
        
        # Group by user and calculate serendipity
        serendipity_scores = []
        
        for user_id, user_recs in recs_with_history.group_by('user_id'):
            # Sort by predicted score and take top K
            top_k = user_recs.sort('predict', descending=True).head(k)
            
            # Calculate serendipity as prediction score * (1 - has_purchased)
            serendipity = (top_k['predict'] * (1 - top_k['has_purchased'])).mean()
            serendipity_scores.append(serendipity)
        
        # Return mean serendipity across all users
        return np.mean(serendipity_scores)