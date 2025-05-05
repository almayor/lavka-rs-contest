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


class Visualizer:
    """Utilities for visualizing experiment results"""
    
    @staticmethod
    def plot_metrics_comparison(experiments, metric='ndcg@10', figsize=(10, 6)):
        """
        Plot comparison of metrics across experiments
        
        Parameters:
        -----------
        experiments: Dictionary of experiment results
        metric: Metric to compare
        figsize: Figure size (width, height)
        save_path: Path to save graph (optional)
        
        Returns:
        --------
        Matplotlib figure
        """
        # Extract metric values
        exp_names = []
        metric_values = []
        
        for exp_name, exp_results in experiments.items():
            exp_names.append(exp_name)
            metric_values.append(exp_results['cv_results']['average_metrics'][metric])
        
        # Create plot
        plt.figure(figsize=figsize)
        bars = plt.bar(exp_names, metric_values)
        
        # Add labels and title
        plt.xlabel('Experiment')
        plt.ylabel(metric)
        plt.title(f'Comparison of {metric} Across Experiments')
        plt.xticks(rotation=45, ha='right')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_feature_importance(feature_importance, top_n=20, figsize=(12, 8)):
        """
        Plot feature importance
        
        Parameters:
        -----------
        feature_importance: Dictionary of feature importance values
        top_n: Number of top features to display
        figsize: Figure size (width, height)
        
        Returns:
        --------
        Matplotlib figure
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)
        
        # Take top N features
        top_features = reversed(sorted_features[:top_n])
        
        # Unpack feature names and importance values
        feature_names, importance_values = zip(*top_features)
        
        # Create horizontal bar plot
        plt.figure(figsize=figsize)
        plt.barh(range(len(feature_names)), importance_values, align='center')
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_metrics_heatmap(experiments, metrics=None, figsize=(12, 8)):
        """
        Plot heatmap of metrics across experiments
        
        Parameters:
        -----------
        experiments: Dictionary of experiment results
        metrics: List of metrics to include (defaults to all)
        figsize: Figure size (width, height)
        
        Returns:
        --------
        Matplotlib figure
        """
        # Extract experiment names and metrics
        exp_names = list(experiments.keys())
        
        if metrics is None:
            # Get metrics from first experiment
            first_exp = next(iter(experiments.values()))
            metrics = list(first_exp['cv_results']['average_metrics'].keys())
        
        # Create data matrix
        data = np.zeros((len(exp_names), len(metrics)))
        
        for i, exp_name in enumerate(exp_names):
            for j, metric in enumerate(metrics):
                data[i, j] = experiments[exp_name]['cv_results']['average_metrics'][metric]
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(data, annot=True, fmt='.4f', cmap='YlGnBu',
                   xticklabels=metrics, yticklabels=exp_names)
        plt.xlabel('Metrics')
        plt.ylabel('Experiments')
        plt.title('Metrics Comparison Heatmap')
        plt.tight_layout()
        
        return plt.gcf()