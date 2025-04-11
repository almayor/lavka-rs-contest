class DataLoader:
    """Data loading and preprocessing"""
    
    def __init__(self, config: Config):
        """Initialize with configuration"""
        self.config = config
        self.train_df = None
        self.test_df = None
    
    def load_data(self):
        """Load training and testing data"""
        logger.info("Loading data...")
        
        # Load train data
        train_path = self.config.get('data', 'train_path')
        self.train_df = pl.read_parquet(train_path)
        
        # Load test data
        test_path = self.config.get('data', 'test_path')
        self.test_df = pl.read_parquet(test_path)
        
        # Sample if needed
        sample_size = self.config.get('data', 'sample_size')
        if sample_size is not None:
            self.train_df = self.train_df.sample(
                n=sample_size, 
                seed=self.config.get('data', 'random_seed')
            )
        
        logger.info(f"Loaded train data: {len(self.train_df)} rows")
        logger.info(f"Loaded test data: {len(self.test_df)} rows")
        
        return self.train_df, self.test_df
    
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply preprocessing steps based on configuration"""
        logger.info("Preprocessing data...")
        
        # Make a copy to avoid modifying the original
        processed_df = df.clone()
        
        # Apply preprocessing steps based on config
        if self.config.get('preprocessing', 'remove_duplicates'):
            processed_df = processed_df.unique()
            logger.info("Removed duplicates")
        
        if self.config.get('preprocessing', 'fill_nulls'):
            # Fill numerical nulls with 0
            processed_df = processed_df.fill_null(0)
            logger.info("Filled null values")
        
        if self.config.get('preprocessing', 'normalize_timestamps'):
            # Ensure timestamps are in a consistent format
            if 'timestamp' in processed_df.columns:
                # Convert to datetime if it's not already
                if processed_df['timestamp'].dtype != pl.Datetime:
                    processed_df = processed_df.with_columns(
                        pl.col('timestamp').cast(pl.Datetime)
                    )
                logger.info("Normalized timestamps")
        
        if self.config.get('preprocessing', 'clean_text'):
            # Apply text cleaning to relevant columns
            text_columns = [col for col in processed_df.columns 
                           if any(substr in col for substr in ['name', 'description', 'text'])]
            
            for col in text_columns:
                if col in processed_df.columns:
                    # Simple text cleaning example
                    processed_df = processed_df.with_columns(
                        pl.col(col).str.strip().str.to_lowercase()
                    )
            logger.info("Cleaned text columns")
        
        return processed_df
    
    def create_validation_splits(self):
        """Create training/validation splits based on config"""
        validation_method = self.config.get('validation', 'method')
        
        if validation_method == 'temporal':
            return self._create_temporal_splits()
        elif validation_method == 'kaggle':
            return self._create_kaggle_split()
        elif validation_method == 'random':
            return self._create_random_splits()
        else:
            raise ValueError(f"Unknown validation method: {validation_method}")
    
    def _create_temporal_splits(self):
        """Create time-based validation folds"""
        n_folds = self.config.get('validation', 'n_folds')
        gap_days = self.config.get('validation', 'gap_days')
        
        # Ensure data is sorted by timestamp
        df = self.train_df.sort('timestamp')
        
        # Calculate time range and fold duration
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        time_range = max_time - min_time
        fold_duration = time_range / (n_folds + 1)  # +1 to leave last fold for final validation
        
        folds = []
        for i in range(n_folds):
            # Calculate time boundaries
            train_end_time = min_time + fold_duration * (i + 1)
            
            # Add gap if specified
            if gap_days > 0:
                val_start_time = train_end_time + timedelta(days=gap_days)
            else:
                val_start_time = train_end_time
                
            val_end_time = val_start_time + fold_duration
            
            # Create train and validation sets
            train_df = df.filter(pl.col('timestamp') < train_end_time)
            val_df = df.filter((pl.col('timestamp') >= val_start_time) & 
                              (pl.col('timestamp') < val_end_time))
            
            folds.append((train_df, val_df))
        
        logger.info(f"Created {len(folds)} temporal validation folds")
        return folds
    
    def _create_kaggle_split(self):
        """Use Kaggle's predefined train/validation split"""
        # For Kaggle competitions, you might have a separate validation file
        # Here we'll simulate by taking the most recent data as validation
        
        df = self.train_df.sort('timestamp')
        
        # Use the most recent X% as validation
        test_size = self.config.get('validation', 'test_size')
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df[:split_idx]
        val_df = df[split_idx:]
        
        logger.info(f"Created Kaggle-style split: {len(train_df)} train, {len(val_df)} validation")
        return [(train_df, val_df)]
    
    def _create_random_splits(self):
        """Create random validation splits (not recommended for time series)"""
        n_folds = self.config.get('validation', 'n_folds')
        test_size = self.config.get('validation', 'test_size')
        seed = self.config.get('data', 'random_seed')
        
        df = self.train_df.clone()
        folds = []
        
        for i in range(n_folds):
            # Shuffle and split
            shuffled = df.sample(fraction=1.0, seed=seed + i)
            split_idx = int(len(shuffled) * (1 - test_size))
            
            train_df = shuffled[:split_idx]
            val_df = shuffled[split_idx:]
            
            folds.append((train_df, val_df))
        
        logger.info(f"Created {len(folds)} random validation folds")
        return folds
