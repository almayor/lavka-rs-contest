from datetime import timedelta
from typing import Optional, Tuple, Generator
from enum import Enum

import os
import polars as pl

from .utils.config import Config
from .utils.custom_logging import get_logger


class DataLoader:
    """
    Loads and preprocesses data, and provides convenient train/validation/test splits.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        self.train_df: Optional[pl.DataFrame] = None
        self.test_df: Optional[pl.DataFrame] = None
        self.holdout_df: Optional[pl.DataFrame] = None

    def setup(self):
        """
        Reads train and test datasets from configured paths, converts timestamps.
        """
        train_path = self.config.get('data.train_path')
        if not train_path or not os.path.exists(train_path):
            raise FileNotFoundError(f"Training file not found: {train_path}")
        self.logger.info(f"Loading training data from {train_path}")
        self.train_df = pl.read_parquet(train_path)
        
        test_path = self.config.get('data.test_path')
        if not test_path or not os.path.exists(test_path):
            raise FileNotFoundError(f"Testing file not found: {test_path}")
        self.logger.info(f"Loading test data from {test_path}")
        self.test_df = pl.read_parquet(test_path)
        
        preprocessor = DataPreprocessor(self.config)
        self.train_df = preprocessor(self.train_df, clean=True)
        self.test_df  = preprocessor(self.test_df, clean=False)
        
        if self.config.get('data.sample.train_fraction', 1) < 1:
            fraction = self.config.get('data.sample.train_fraction', 1)
            self.train_df = self.train_df.sample(fraction=fraction)
        if self.config.get('data.holdout.enabled', False):
            self._create_holdout()

    def train_split(
        self,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Produce history/train/val.
        Returns train_history, train_target, val_history, val_target dataframes.
        """
        if self.train_df is None:
            raise ValueError("Train data not loaded. Call setup() first.")

        target_days = self.config.get('data.target_days', 30)
        val_history, val_target = self.split_timewise(self.train_df, target_days)
        train_history, train_target = self.split_timewise(val_history, target_days)
        self._log_split("Validation Split",
                        train_history=train_history,
                        train_target=train_target,
                        val_history=val_history,
                        val_target=val_target)
        return train_history, train_target, val_history, val_target
    
    def holdout_split(
        self,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Produce history/holdout.
        Returns history and holdout dataframes.
        """
        if self.train_df is None:
            raise ValueError("Train data not loaded. Call setup() first.")
        if self.holdout_df is None:
            raise ValueError("Holdout disabled.")
    
        return self.train_df, self.holdout_df

    def final_split(
        self,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:  
        """
        Produce history/train to train on the ENTIRE data.
        Returns history and target dataframes.
        """ 
        if self.train_df is None:
            raise ValueError("Train data not loaded. Call setup() first.")

        #Undo holdout split
        if self.holdout_df is not None:
            train_df = pl.concat([self.train_df, self.holdout_df])
            self.logger.info("Holdout data merged back into training data.")
        else:
            train_df = self.train_df

        target_days = self.config.get('data.target_days', 30)   
        history, target = self.split_timewise(train_df, target_days)
        self._log_split("Validation Split",
                        history=history,
                        target=target)
        return history, target
    
    def submission_split(
        self,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Produce history/test.
        Returns history and test dataframes.
        """
        if self.test_df is None:
            raise ValueError("Test data not loaded. Call setup() first.")
        
        #Undo holdout split
        if self.holdout_df is not None:
            train_df = pl.concat([self.train_df, self.holdout_df])
            self.logger.info("Holdout data merged back into training data.")
        else:
            train_df = self.train_df

        return train_df, self.test_df
    
    def _create_holdout(self):
        """
        Create a holdout dataset
        """
        holdout_days = self.config.get('data.holdout.holdout_days', 30)
        self.train_df, self.holdout_df = self.split_timewise(self.train_df, holdout_days)
        self._log_split(
            "Holdout Split", train=self.train_df, holdout=self.holdout_df
        )
        
    def _log_split(self, name: str, **parts: pl.DataFrame) -> None:
        """
        Log time range, row count, and span in days for each split component.
        """
        self.logger.info(f"{name}:")
        for part_name, df in parts.items():
            cnt = len(df)
            if cnt == 0:
                self.logger.info(f"  {part_name}: empty")
            else:
                t0 = df['timestamp'].min()
                t1 = df['timestamp'].max()
                # Compute full-day span
                days = (t1 - t0).days
                self.logger.info(
                    f"  {part_name}:\t{t0} → {t1} "
                    f"({cnt:_} rows, {days} days)"
                )
        

    @staticmethod  
    def split_timewise(df: pl.DataFrame, target_days: int):
        max_timestamp = df.select(pl.col("timestamp").max()).item()
        split_timestamp = max_timestamp - timedelta(days=target_days)
        early_df = df.filter(pl.col('timestamp') < split_timestamp)
        later_df = df.filter(pl.col('timestamp') >= split_timestamp)
        return early_df, later_df
    

class DataPreprocessor:

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    def __call__(self, df: pl.DataFrame, clean=True):
        """Preprocessing dataframe. If `clean = False`, no additional cleaning will be done."""
        df = self._convert_timestamps(df)
        if not clean:
            return df
        if self.config.get('data.cleaning.remove_view_only_requests'):
            df = self._remove_view_only_requests(df)
        return df

    def _convert_timestamps(self, df: pl.DataFrame):
        """
        Convert any epoch-based 'timestamp' columns to Polars Datetime type.
        """
        if 'timestamp' in df.columns:
            # Detect whether timestamp is in seconds or milliseconds
            sample = df['timestamp'].head(1).to_list()
            unit = 'ms' if sample and sample[0] > 1e12 else 's'
            self.logger.debug(f"Converting 'timestamp' from epoch[{unit}] to datetime")
            df = df.with_columns(
                pl.from_epoch(pl.col('timestamp'), time_unit=unit).alias('timestamp')
            )
        else:
            self.logger.warning(f"Skipping timestamp conversion as `timestamp` isn't available")
        return df
    
    def _remove_view_only_requests(self, df):
        """
        Remove sessions without a single non-view action.
        """
        if 'action_type' in df.columns and 'request_id' in df.columns:
            # 1. Find all request_ids with at least one action ≠ "AT_View"
            good_ids = (
                df
                .filter(pl.col("action_type") != "AT_View")
                .select("request_id")
                .unique()
            )
            # 2. Inner‐join back to keep _all_ rows for those requests
            df = df.join(good_ids, on="request_id", how="inner")
            self.logger.debug(f"Removing sessions without a single non-view action.")
        else:
            self.logger.warning(f"Skipping removing sessions without a single non-view action, as `action_type` and `request_id` aren't available.")
        return df
