import numpy as np
import polars as pl
from typing import List, Dict, Any
import re

from ..custom_logging import get_logger
from ..config import Config

class TextProcessor:
    """Simplified text processing with pretrained models"""
    
    def __init__(self, config: Config):
        """Initialize text processor with configuration"""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.model = None
        self.embedding_size = 0
        self.use_gpu = False
        self.device = None
        
        # Check for GPU availability early
        try:
            import torch
            # Set default GPU behavior
            use_gpu = self.config.get('text_processing.use_gpu', True)
            self.use_gpu = use_gpu and torch.cuda.is_available()
            if self.use_gpu:
                self.device = torch.device("cuda")
                self.logger.info("GPU is available and will be used for text processing")
            else:
                self.device = torch.device("cpu")
                if use_gpu:
                    self.logger.warning("GPU was requested but is not available, using CPU")
                else:
                    self.logger.info("Using CPU for text processing (by configuration)")
        except ImportError:
            self.logger.warning("PyTorch not available, GPU acceleration disabled")
            self.use_gpu = False
        
        # Load pretrained model
        model_type = self.config.get('text_processing.model_type', 'sentence-transformers')
        
        if model_type == 'sentence-transformers':
            self._load_sentence_transformers()
        elif model_type == 'word2vec':
            self._load_word2vec()
        elif model_type == 'fasttext':
            self._load_fasttext()
        else:
            self.logger.warning(
                f"Unknown model type: {model_type}. "
                "Falling back to sentence-transformers."
            )
            self._load_sentence_transformers()
    
    def _load_sentence_transformers(self):
        """Load SentenceTransformers model with GPU acceleration if available"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Get model name from config
            model_name = self.config.get(
                'text_processing.model_name', 
                'paraphrase-multilingual-MiniLM-L12-v2'  # Small but effective model
            )
            
            # Check if GPU is available and if user wants to use it
            use_gpu = self.config.get('text_processing.use_gpu', True)
            device = None
            
            if use_gpu and torch.cuda.is_available():
                device = "cuda"
                self.logger.info("GPU detected and will be used for text embedding")
            else:
                if use_gpu and not torch.cuda.is_available():
                    self.logger.warning("GPU requested but not available, using CPU instead")
                device = "cpu"
                self.logger.info("Using CPU for text embedding")
                
            # Load model on specified device
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_size = self.model.get_sentence_embedding_dimension()
            self.model_type = 'sentence-transformers'
            
            self.logger.info(f"Loaded sentence-transformers model: {model_name} on {device}")
            
        except ImportError:
            self.logger.error(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None
    
    def _load_word2vec(self):
        """Load Word2Vec model"""
        try:
            import gensim.downloader as api
            import torch
            
            # Check if GPU is available for post-processing
            use_gpu = self.config.get('text_processing.use_gpu', True)
            self.use_gpu = use_gpu and torch.cuda.is_available()
            
            if self.use_gpu:
                self.device = torch.device("cuda")
                self.logger.info("GPU will be used for vector operations with Word2Vec")
            else:
                self.device = torch.device("cpu")
                self.logger.info("CPU will be used for vector operations with Word2Vec")
                
            # Load the model (Word2Vec itself doesn't use GPU directly)
            model_name = self.config.get(
                'text_processing.model_name', 
                'word2vec-ruscorpora-300'  # Reasonable size/performance trade-off
            )
            self.model = api.load(model_name)
            self.embedding_size = self.model.vector_size
            self.model_type = 'word2vec'
            self.logger.info(f"Loaded word2vec model: {model_name}")
        except ImportError:
            self.logger.error(
                "gensim not available. "
                "Install with: pip install gensim"
            )
            self.model = None
    
    def _load_fasttext(self):
        """Load FastText model"""
        try:
            import fasttext
            import torch
            
            # Check if GPU is available for post-processing
            use_gpu = self.config.get('text_processing.use_gpu', True)
            self.use_gpu = use_gpu and torch.cuda.is_available()
            
            if self.use_gpu:
                self.device = torch.device("cuda")
                self.logger.info("GPU will be used for vector operations with FastText")
            else:
                self.device = torch.device("cpu")
                self.logger.info("CPU will be used for vector operations with FastText")
            
            # Load the model (FastText itself doesn't use GPU directly)
            model_path = self.config.get(
                'text_processing.model_path',
                'cc.ru.300.bin'  # Default Russian model
            )
            self.model = fasttext.load_model(model_path)
            self.embedding_size = self.model.get_dimension()
            self.model_type = 'fasttext'
            self.logger.info(f"Loaded fasttext model: {model_path}")
        except ImportError:
            self.logger.error(
                "fasttext not available. "
                "Install with: pip install fasttext-wheel"
            )
            self.model = None
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts using the pretrained model
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy.ndarray: Matrix of embeddings
        """
        import time
        import traceback
        start_time = time.time()
        
        self.logger.info(f"Getting embeddings for {len(texts)} texts using model type: {self.model_type}")
        
        # First, check if we have a model loaded
        if self.model is None:
            self.logger.error("No text model loaded. This should have been handled during initialization.")
            self.logger.error("Returning zero embeddings, but features will not be useful!")
            return np.zeros((len(texts), 1))
        
        # Early validation of input data
        if len(texts) == 0:
            self.logger.warning("Empty text list provided to get_embeddings. Returning empty array.")
            return np.zeros((0, self.embedding_size or 1))
        
        # Clean texts (remove empty strings, handle non-string inputs) - vectorized
        import numpy as np  # Import numpy here for this method's scope
        
        # Log detailed text stats
        empty_texts = sum(1 for t in texts if not t)
        if empty_texts > 0:
            self.logger.warning(f"Found {empty_texts} empty texts that will be replaced with space")
            self.logger.warning(f"Empty text percentage: {empty_texts/len(texts)*100:.2f}%")
            
        non_string_texts = sum(1 for t in texts if not isinstance(t, str))
        if non_string_texts > 0:
            self.logger.warning(f"Found {non_string_texts} non-string texts ({non_string_texts/len(texts)*100:.2f}%)")
            # Log some examples with more detail
            for i, t in enumerate(texts):
                if not isinstance(t, str):
                    self.logger.warning(f"Example non-string text at index {i}: {type(t)}, value: {t!r}")
                    if i >= 5:  # Log a few more examples
                        break
        
        # Log text length distribution
        text_lengths = [len(str(t)) for t in texts]
        self.logger.info(f"Text length stats - min: {min(text_lengths)}, max: {max(text_lengths)}, "  
                       f"mean: {sum(text_lengths)/len(text_lengths):.1f}, "  
                       f"number of texts < 5 chars: {sum(1 for l in text_lengths if l < 5)}")
            
        # Vectorized text cleaning
        self.logger.debug("Cleaning and preparing texts for embedding")
        cleaning_start = time.time()
        try:
            cleaned_texts = np.array([
                t if isinstance(t, str) and t else " " 
                for t in texts
            ])
            self.logger.debug(f"Text cleaning completed in {time.time() - cleaning_start:.2f} seconds")
        except Exception as e:
            self.logger.error(f"Error during text cleaning: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        
        # Log a few examples of cleaned texts with more detail
        if len(cleaned_texts) > 0:
            sample_size = min(5, len(cleaned_texts))
            sample_texts = cleaned_texts[:sample_size]
            self.logger.debug(f"Sample cleaned texts:")
            for i, text in enumerate(sample_texts):
                self.logger.debug(f"  Sample {i+1}: '{text[:50]}{'...' if len(text)>50 else ''}' (length: {len(text)})")
        
        try:
            if self.model_type == 'sentence-transformers':
                self.logger.info(f"Using SentenceTransformers model for embedding (batch_size=64)")
                
                try:
                    # Get embeddings directly from the model (already batch optimized)
                    model_start = time.time()
                    embeddings = self.model.encode(
                        cleaned_texts, 
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        batch_size=64  # Process in batches for better performance
                    )
                    encoding_time = time.time() - model_start
                    self.logger.info(f"SentenceTransformers encoding completed in {encoding_time:.2f} seconds")
                    self.logger.info(f"Embedding shape: {embeddings.shape}")
                    return embeddings
                except Exception as e:
                    self.logger.error(f"Error during SentenceTransformers encoding: {str(e)}")
                    self.logger.exception("Traceback:")
                    raise
                
            elif self.model_type == 'word2vec':
                self.logger.info("Using Word2Vec model for embedding with batched processing")
                
                # For word2vec, optimize with array operations
                embeddings = np.zeros((len(cleaned_texts), self.embedding_size))
                
                # Process in batches for better performance
                batch_size = 100
                num_batches = (len(cleaned_texts) + batch_size - 1) // batch_size
                self.logger.info(f"Processing {len(cleaned_texts)} texts in {num_batches} batches of size {batch_size}")
                
                missing_words = 0
                empty_vectors = 0
                
                for batch_idx in range(num_batches):
                    batch_start = time.time()
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(cleaned_texts))
                    batch_texts = cleaned_texts[start_idx:end_idx]
                    
                    self.logger.debug(f"Processing batch {batch_idx+1}/{num_batches}: items {start_idx}-{end_idx}")
                    
                    # Process each text in the batch
                    batch_missing_words = 0
                    for i, text in enumerate(batch_texts):
                        words = text.lower().split()
                        
                        # Count words not in vocabulary
                        words_not_in_vocab = sum(1 for word in words if word not in self.model)
                        batch_missing_words += words_not_in_vocab
                        
                        # Get embeddings for all words in a single list comprehension
                        valid_word_vectors = [self.model[word] for word in words if word in self.model]
                        
                        # Only compute mean if we have valid vectors
                        if valid_word_vectors:
                            # Convert to array and compute mean in one operation
                            word_vectors_array = np.array(valid_word_vectors)
                            embeddings[start_idx + i] = np.mean(word_vectors_array, axis=0)
                        else:
                            empty_vectors += 1
                            # Log when no valid words found
                            self.logger.warning(f"No valid words found in text: '{text[:50]}{'...' if len(text)>50 else ''}'")
                    
                    missing_words += batch_missing_words
                    batch_time = time.time() - batch_start
                    self.logger.debug(f"Batch {batch_idx+1}/{num_batches} completed in {batch_time:.2f} seconds. " + 
                                    f"Words not in vocabulary: {batch_missing_words}")
                
                total_time = time.time() - start_time
                self.logger.info(f"Word2Vec embedding completed in {total_time:.2f} seconds")
                self.logger.info(f"Total words not in vocabulary: {missing_words}")
                self.logger.info(f"Texts with no valid vectors: {empty_vectors} ({empty_vectors/len(cleaned_texts)*100:.1f}%)")
                self.logger.info(f"Embedding shape: {embeddings.shape}")
                return embeddings
                
            elif self.model_type == 'fasttext':
                self.logger.info("Using FastText model for embedding with batched processing")
                
                # Optimize fasttext embedding generation
                # Pre-allocate output array
                embeddings = np.zeros((len(cleaned_texts), self.embedding_size))
                
                # Process in batches for better performance
                batch_size = 100
                num_batches = (len(cleaned_texts) + batch_size - 1) // batch_size
                self.logger.info(f"Processing {len(cleaned_texts)} texts in {num_batches} batches of size {batch_size}")
                
                for batch_idx in range(num_batches):
                    batch_start = time.time()
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(cleaned_texts))
                    batch_texts = cleaned_texts[start_idx:end_idx]
                    
                    self.logger.debug(f"Processing batch {batch_idx+1}/{num_batches}: items {start_idx}-{end_idx}")
                    
                    try:
                        # Create a batch of embeddings
                        batch_embeddings = np.array([
                            self.model.get_sentence_vector(text) for text in batch_texts
                        ])
                        
                        # Store in result array
                        embeddings[start_idx:end_idx] = batch_embeddings
                        
                        batch_time = time.time() - batch_start
                        self.logger.debug(f"Batch {batch_idx+1}/{num_batches} completed in {batch_time:.2f} seconds")
                    except Exception as e:
                        self.logger.error(f"Error in FastText batch {batch_idx+1}: {str(e)}")
                        self.logger.error(f"Problematic batch content sample: {batch_texts[:3]}")
                        raise
                
                total_time = time.time() - start_time
                self.logger.info(f"FastText embedding completed in {total_time:.2f} seconds")
                self.logger.info(f"Embedding shape: {embeddings.shape}")
                
                # Check for NaNs
                nan_count = np.isnan(embeddings).sum()
                if nan_count > 0:
                    self.logger.warning(f"Found {nan_count} NaN values in embeddings")
                    
                return embeddings
                
            else:
                self.logger.warning(f"Unknown model type: {self.model_type}")
                return np.zeros((len(texts), 1))
                
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            self.logger.exception("Detailed traceback:")
            raise
    
    def reduce_dimensions(self, embeddings: np.ndarray, dimensions: int = 20) -> np.ndarray:
        """
        Reduce embedding dimensions for efficiency
        
        Args:
            embeddings: Original embeddings matrix
            dimensions: Target number of dimensions
            
        Returns:
            numpy.ndarray: Reduced embeddings matrix
        """
        if embeddings.shape[1] <= dimensions:
            return embeddings
            
        try:
            # Use cuML's PCA for GPU acceleration if available
            if self.use_gpu:
                try:
                    # First check if cuML is available
                    cuml_available = False
                    try:
                        import cuml
                        cuml_available = True
                    except (ImportError, ModuleNotFoundError):
                        self.logger.warning("cuML not available, falling back to CPU implementation")
                    
                    if cuml_available:
                        self.logger.info("Using GPU-accelerated PCA with cuML")
                        
                        # Initialize and fit cuML PCA
                        try:
                            gpu_pca = cuml.PCA(n_components=dimensions, random_state=42)
                            reduced = gpu_pca.fit_transform(embeddings)
                            
                            # Convert back to numpy array
                            if hasattr(reduced, 'get'):  # For cuML arrays
                                reduced = reduced.get()
                            else:
                                reduced = np.array(reduced)
                                
                            self.logger.info(
                                f"Reduced embeddings from {embeddings.shape[1]} to {dimensions} dimensions using GPU"
                            )
                            return reduced
                        except Exception as e:
                            self.logger.warning(f"Error using cuML PCA: {str(e)}")
                            self.logger.warning("Falling back to CPU implementation")
                except Exception as e:
                    self.logger.warning(f"Unexpected error with GPU acceleration: {str(e)}")
                    self.logger.warning("Falling back to CPU implementation")
            
            # If we're here, either GPU is not available or cuML failed to import
            from sklearn.decomposition import PCA
            
            # Use batch processing for large datasets
            batch_size = 10000
            num_samples = embeddings.shape[0]
            
            # For small datasets, just use standard PCA
            if num_samples <= batch_size:
                pca = PCA(n_components=dimensions, random_state=42)
                reduced = pca.fit_transform(embeddings)
            else:
                # For large datasets, use incremental PCA or batched processing
                try:
                    # Try to use IncrementalPCA which is more memory efficient
                    from sklearn.decomposition import IncrementalPCA
                    
                    # Initialize the model
                    ipca = IncrementalPCA(n_components=dimensions, batch_size=batch_size)
                    
                    # Process in batches
                    for i in range(0, num_samples, batch_size):
                        end = min(i + batch_size, num_samples)
                        ipca.partial_fit(embeddings[i:end])
                    
                    # Transform the data
                    reduced = ipca.transform(embeddings)
                    
                except ImportError:
                    # Fall back to regular PCA with batch processing if IncrementalPCA is not available
                    self.logger.info("IncrementalPCA not available, using regular PCA with batching")
                    
                    # Fit PCA on a smaller random subset for initialization
                    import numpy as np
                    sample_size = min(5000, num_samples)
                    sample_indices = np.random.choice(num_samples, size=sample_size, replace=False)
                    
                    pca = PCA(n_components=dimensions, random_state=42)
                    pca.fit(embeddings[sample_indices])
                    
                    # Transform in batches
                    reduced = np.zeros((num_samples, dimensions))
                    for i in range(0, num_samples, batch_size):
                        end = min(i + batch_size, num_samples)
                        reduced[i:end] = pca.transform(embeddings[i:end])
            
            self.logger.info(
                f"Reduced embeddings from {embeddings.shape[1]} to {dimensions} dimensions using CPU"
            )
            return reduced
            
        except ImportError:
            self.logger.warning(
                "sklearn not available for dimension reduction. "
                "Using original embeddings."
            )
            return embeddings


# Add methods to FeatureFactory for text embeddings
def register_text_embedding_features():
    """Register text embedding methods with FeatureFactory"""
    from ..feature_factory import FeatureFactory
    
    # Import numpy here once for all inner functions to use
    import numpy as np
    
    @FeatureFactory.register('product_embeddings')
    def generate_product_embeddings(
        history_df: pl.DataFrame, target_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Generate product name embeddings using pretrained model"""
        # Get a logger
        import logging
        import time
        import traceback
        import numpy as np
        logger = logging.getLogger("ProductEmbeddings")
        
        try:
            logger.info(f"==== STARTING PRODUCT EMBEDDINGS GENERATION ====")
            logger.info(f"History DataFrame: {len(history_df)} rows, {len(history_df.columns)} columns")
            logger.info(f"Target DataFrame: {len(target_df)} rows, {len(target_df.columns)} columns")
            
            # Check if required columns exist
            required_cols = ['product_id', 'product_name']
            missing_in_history = [col for col in required_cols if col not in history_df.columns]
            missing_in_target = [col for col in required_cols if col not in target_df.columns]
            
            if missing_in_history or missing_in_target:
                if missing_in_history:
                    logger.error(f"Missing columns in history_df: {missing_in_history}")
                    logger.error(f"Available columns in history_df: {history_df.columns}")
                if missing_in_target:
                    logger.error(f"Missing columns in target_df: {missing_in_target}")
                    logger.error(f"Available columns in target_df: {target_df.columns}")
                raise ValueError(f"Missing required columns for product embeddings")
            
            # Create text processor
            logger.info("Initializing TextProcessor")
            config_start = time.time()
            config = Config()
            text_processor = TextProcessor(config)
            logger.info(f"TextProcessor initialization completed in {time.time() - config_start:.2f} seconds")
            
            # Get unique product names
            logger.info("Collecting unique products from history and target dataframes")
            concat_start = time.time()
            products = pl.concat([
                history_df.select('product_id', 'product_name').unique(),
                target_df.select('product_id', 'product_name').unique()
            ]).unique(subset=['product_id'])
            logger.info(f"Product concatenation completed in {time.time() - concat_start:.2f} seconds")
            
            # Check for nulls in product data
            null_product_ids = products.filter(pl.col('product_id').is_null()).height
            null_product_names = products.filter(pl.col('product_name').is_null()).height
            
            if null_product_ids > 0:
                logger.warning(f"Found {null_product_ids} null product IDs")
            if null_product_names > 0:
                logger.warning(f"Found {null_product_names} null product names")
                # Show examples of products with null names
                null_examples = products.filter(pl.col('product_name').is_null()).sample(n=min(5, null_product_names))
                logger.warning(f"Examples of products with null names: {null_examples}")
            
            logger.info(f"Found {len(products)} unique products to embed")
            
            # Get embeddings
            product_names = products['product_name'].to_list()
            
            # More detailed logging of product names
            sample_size = min(5, len(product_names))
            logger.info(f"Sample of product names to embed:")
            for i, name in enumerate(product_names[:sample_size]):
                logger.info(f"  Sample {i+1}: '{name}' (type: {type(name)}, length: {len(str(name)) if name is not None else 0})")
            
            # Calculate length statistics
            name_lengths = [len(str(name)) if name is not None else 0 for name in product_names]
            logger.info(f"Product name length stats - min: {min(name_lengths)}, max: {max(name_lengths)}, " 
                       f"mean: {sum(name_lengths)/len(name_lengths):.1f}")
            
            logger.info(f"Getting embeddings for {len(product_names)} products")
            embedding_start = time.time()
            try:
                embeddings = text_processor.get_embeddings(product_names)
                embedding_time = time.time() - embedding_start
                logger.info(f"Embeddings generated in {embedding_time:.2f} seconds")
                if embeddings is None:
                    logger.error("Embedding generation returned None")
                    raise ValueError("Embedding generation failed with None result")
                logger.info(f"Embedding shape: {embeddings.shape}")
                
                # Check for NaN or zero values in embeddings
                nan_count = np.isnan(embeddings).sum()
                zero_rows = np.sum(np.all(embeddings == 0, axis=1))
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN values in raw embeddings")
                if zero_rows > 0:
                    logger.warning(f"Found {zero_rows} all-zero rows in raw embeddings")
                    
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        except Exception as e:
            logger.error(f"Fatal error in product embeddings generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Reduce dimensions if needed
        dimensions = config.get('text_processing.embedding_dimensions', 20)
        if embeddings.shape[1] > dimensions:
            logger.info(f"Reducing embedding dimensions from {embeddings.shape[1]} to {dimensions}")
            start_time = logging.Formatter.converter()
            embeddings = text_processor.reduce_dimensions(embeddings, dimensions)
            end_time = logging.Formatter.converter()
            reduction_time = (end_time[0] - start_time[0]) + (end_time[1] - start_time[1])/1000000
            logger.info(f"Dimension reduction completed in {reduction_time:.2f} seconds. New shape: {embeddings.shape}")
        
        # Create feature columns
        feature_names = [f'product_embed_{i}' for i in range(embeddings.shape[1])]
        logger.info(f"Created {len(feature_names)} embedding feature columns")
        
        # Convert to DataFrame
        import pandas as pd
        logger.info("Converting embeddings to DataFrame")
        embed_df = pd.DataFrame(
            embeddings, 
            columns=feature_names,
            index=products['product_id'].to_list()
        )
        embed_df['product_id'] = products['product_id'].to_list()
        
        # Convert to polars
        logger.info("Converting pandas DataFrame to polars")
        embed_pl = pl.from_pandas(embed_df)
        
        # Check for NaN values - using polars-specific approach
        # First sum null counts for each column, then sum the totals
        nan_count = embed_pl.null_count().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in embeddings")
        
        # Join with target DataFrame
        logger.info(f"Joining embeddings with target DataFrame (target has {len(target_df)} rows)")
        result = target_df.join(embed_pl, on='product_id', how='left')
        
        # Check for missing values after join
        # Correct polars syntax for summing null counts
        missing_count = result.select(feature_names).null_count().sum().sum()
        if missing_count > 0:
            logger.warning(f"After join, found {missing_count} missing embedding values")
            missing_products = target_df.filter(
                ~pl.col('product_id').is_in(products['product_id'])
            ).select('product_id').n_unique()
            logger.warning(f"Found {missing_products} products in target that weren't in the embedding set")
        
        logger.info(f"Product embeddings generation complete. Result shape: {result.shape}")
        return result
    
    @FeatureFactory.register('category_embeddings')
    def generate_category_embeddings(
        history_df: pl.DataFrame, target_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Generate category name embeddings using pretrained model"""
        # Get a logger
        import logging
        import time
        import traceback
        import numpy as np
        logger = logging.getLogger("CategoryEmbeddings")
        
        try:
            logger.info(f"==== STARTING CATEGORY EMBEDDINGS GENERATION ====")
            logger.info(f"History DataFrame: {len(history_df)} rows, {len(history_df.columns)} columns")
            logger.info(f"Target DataFrame: {len(target_df)} rows, {len(target_df.columns)} columns")
            
            # Check if the column exists with detailed logging
            logger.info(f"Checking for required columns")
            logger.info(f"History DataFrame columns: {history_df.columns}")
            logger.info(f"Target DataFrame columns: {target_df.columns}")
            
            if 'product_category' not in history_df.columns or 'product_category' not in target_df.columns:
                missing_in = []
                if 'product_category' not in history_df.columns:
                    missing_in.append('history')
                    logger.error(f"'product_category' column missing in history_df. Available columns: {history_df.columns}")
                    # Check for similarly named columns
                    similar_cols = [col for col in history_df.columns if 'category' in col.lower()]
                    if similar_cols:
                        logger.error(f"Found similar columns in history_df that might be what you want: {similar_cols}")
                if 'product_category' not in target_df.columns:
                    missing_in.append('target')
                    logger.error(f"'product_category' column missing in target_df. Available columns: {target_df.columns}")
                    # Check for similarly named columns
                    similar_cols = [col for col in target_df.columns if 'category' in col.lower()]
                    if similar_cols:
                        logger.error(f"Found similar columns in target_df that might be what you want: {similar_cols}")
                    
                error_msg = f"product_category column missing in {' and '.join(missing_in)} dataframes"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Check for null values in category column
            history_null_cats = history_df.filter(pl.col('product_category').is_null()).height
            target_null_cats = target_df.filter(pl.col('product_category').is_null()).height
            
            if history_null_cats > 0:
                logger.warning(f"Found {history_null_cats} null category values in history DataFrame ({history_null_cats/len(history_df)*100:.2f}%)")
            if target_null_cats > 0:
                logger.warning(f"Found {target_null_cats} null category values in target DataFrame ({target_null_cats/len(target_df)*100:.2f}%)")
            
            # Initialize text processor with extensive logging
            logger.info("Initializing TextProcessor")
            config_start = time.time()
            config = Config()
            text_processor = TextProcessor(config)
            logger.info(f"TextProcessor initialization completed in {time.time() - config_start:.2f} seconds")
            
            # Get unique categories with timing
            logger.info("Collecting unique categories from history and target dataframes")
            concat_start = time.time()
            try:
                categories = pl.concat([
                    history_df.select('product_category').unique(),
                    target_df.select('product_category').unique()
                ]).unique(subset=['product_category'])
                logger.info(f"Category concatenation completed in {time.time() - concat_start:.2f} seconds")
            except Exception as e:
                logger.error(f"Error during category concatenation: {str(e)}")
                logger.error(traceback.format_exc())
                raise
            
            logger.info(f"Found {len(categories)} unique categories to embed")
            
            # Get embeddings with extensive validation
            category_names = categories['product_category'].to_list()
            
            # Log detailed sample of category names for debugging
            sample_size = min(5, len(category_names))
            logger.info(f"Sample of category names to embed:")
            for i, name in enumerate(category_names[:sample_size]):
                logger.info(f"  Sample {i+1}: '{name}' (type: {type(name)}, length: {len(str(name)) if name is not None else 0})")
            
            # Check for null or invalid values with more detail
            null_categories = sum(1 for cat in category_names if cat is None)
            empty_categories = sum(1 for cat in category_names if isinstance(cat, str) and cat.strip() == '')
            if null_categories > 0:
                logger.warning(f"Found {null_categories} NULL category names")
            if empty_categories > 0:
                logger.warning(f"Found {empty_categories} EMPTY category names (just whitespace)")
            
            # Replace null/empty values with a placeholder for embedding
            if null_categories > 0 or empty_categories > 0:
                logger.info("Replacing null/empty category names with placeholder for embedding")
                category_names = [cat if (cat is not None and (not isinstance(cat, str) or cat.strip() != '')) else "unknown_category" 
                               for cat in category_names]
            
            # Calculate length statistics
            name_lengths = [len(str(name)) for name in category_names]
            logger.info(f"Category name length stats - min: {min(name_lengths)}, max: {max(name_lengths)}, " 
                      f"mean: {sum(name_lengths)/len(name_lengths):.1f}")
            
            logger.info(f"Getting embeddings for {len(category_names)} categories")
            embedding_start = time.time()
            try:
                embeddings = text_processor.get_embeddings(category_names)
                embedding_time = time.time() - embedding_start
                logger.info(f"Embeddings generated in {embedding_time:.2f} seconds")
                
                if embeddings is None:
                    logger.error("Embedding generation returned None")
                    raise ValueError("Embedding generation failed with None result")
                    
                logger.info(f"Embedding shape: {embeddings.shape}")
                
                # Check for NaN or zero values in embeddings
                nan_count = np.isnan(embeddings).sum()
                zero_rows = np.sum(np.all(embeddings == 0, axis=1))
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN values in raw embeddings")
                if zero_rows > 0:
                    logger.warning(f"Found {zero_rows} all-zero rows in raw embeddings")
                    
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        except Exception as e:
            logger.error(f"Fatal error in category embeddings generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Reduce dimensions if needed
        dimensions = config.get('text_processing.embedding_dimensions', 20)
        if embeddings.shape[1] > dimensions:
            logger.info(f"Reducing embedding dimensions from {embeddings.shape[1]} to {dimensions}")
            start_time = logging.Formatter.converter()
            embeddings = text_processor.reduce_dimensions(embeddings, dimensions)
            end_time = logging.Formatter.converter()
            reduction_time = (end_time[0] - start_time[0]) + (end_time[1] - start_time[1])/1000000
            logger.info(f"Dimension reduction completed in {reduction_time:.2f} seconds. New shape: {embeddings.shape}")
        
        # Create feature columns
        feature_names = [f'category_embed_{i}' for i in range(embeddings.shape[1])]
        logger.info(f"Created {len(feature_names)} embedding feature columns")
        
        # Convert to DataFrame
        import pandas as pd
        logger.info("Converting embeddings to DataFrame")
        embed_df = pd.DataFrame(
            embeddings, 
            columns=feature_names,
            index=categories['product_category'].to_list()
        )
        embed_df['product_category'] = categories['product_category'].to_list()
        
        # Convert to polars
        logger.info("Converting pandas DataFrame to polars")
        embed_pl = pl.from_pandas(embed_df)
        
        # Check for NaN values - using polars-specific approach
        # First sum null counts for each column, then sum the totals
        nan_count = embed_pl.null_count().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in embeddings")
        
        # Join with target DataFrame
        logger.info(f"Joining embeddings with target DataFrame (target has {len(target_df)} rows)")
        result = target_df.join(embed_pl, on='product_category', how='left')
        
        # Check for missing values after join
        # Correct polars syntax for summing null counts
        missing_count = result.select(feature_names).null_count().sum().sum()
        if missing_count > 0:
            logger.warning(f"After join, found {missing_count} missing embedding values")
            missing_categories = target_df.filter(
                ~pl.col('product_category').is_in(categories['product_category'])
            ).select('product_category').n_unique()
            logger.warning(f"Found {missing_categories} categories in target that weren't in the embedding set")
        
        logger.info(f"Category embeddings generation complete. Result shape: {result.shape}")
        return result
    
    @FeatureFactory.register('user_product_distance')
    def generate_user_product_distance(
        history_df: pl.DataFrame, target_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Generate features based on embedding distance between target products
        and products previously purchased by each user.
        
        This feature calculates:
        1. Distance from the target product to the user's weighted purchase history
        2. Distance from the target product to the user's weighted cart history
        3. Minimum distance to any previously purchased product
        
        Args:
            history_df: DataFrame containing user purchase history
            target_df: DataFrame containing target products to score
            
        Returns:
            DataFrame with added distance features
        """
        # Get a logger
        import logging
        logger = logging.getLogger("UserProductDistance")
        
        # Create text processor
        config = Config()
        text_processor = TextProcessor(config)
        
        # Step 1: Create embedding lookup for all products
        products = pl.concat([
            history_df.select('product_id', 'product_name').unique(),
            target_df.select('product_id', 'product_name').unique()
        ]).unique(subset=['product_id'])
        
        logger.info(f"Generating embeddings for {len(products)} unique products")
        
        # Get embeddings
        product_names = products['product_name'].to_list()
        product_embeddings = text_processor.get_embeddings(product_names)
        
        # Reduce dimensions if needed
        dimensions = config.get('text_processing.embedding_dimensions', 20)
        if product_embeddings.shape[1] > dimensions:
            product_embeddings = text_processor.reduce_dimensions(product_embeddings, dimensions)
        
        # Create embedding lookup dictionary and matrix
        product_id_list = products['product_id'].to_list()
        product_embedding_dict = {
            pid: embedding for pid, embedding in zip(product_id_list, product_embeddings)
        }
        
        logger.info(f"Created embedding lookup with {len(product_embedding_dict)} products")
        
        # Step 2: Calculate user purchase history embeddings (weighted by frequency)
        # Get purchase history
        purchase_history = history_df.filter(pl.col('action_type') == "AT_Purchase").select(
            'user_id', 'product_id', 'timestamp'
        )
        
        # Get cart history
        cart_history = history_df.filter(pl.col('action_type') == "AT_CartUpdate").select(
            'user_id', 'product_id', 'timestamp'
        )
        
        # Calculate purchase weights per user-product
        purchase_weights = purchase_history.group_by('user_id', 'product_id').agg([
            pl.count().alias('purchase_count'),
            pl.max('timestamp').alias('last_purchase_time')
        ])
        
        # Calculate cart weights per user-product
        cart_weights = cart_history.group_by('user_id', 'product_id').agg([
            pl.count().alias('cart_count'),
            pl.max('timestamp').alias('last_cart_time')
        ])
        
        logger.info(f"Calculated weights for {len(purchase_weights)} user-product purchase pairs")
        logger.info(f"Calculated weights for {len(cart_weights)} user-product cart pairs")
        
        # Convert to pandas for easier processing
        purchase_weights_pd = purchase_weights.to_pandas()
        cart_weights_pd = cart_weights.to_pandas()
        
        import pandas as pd
        
        # Step 2a: Optimize purchase history processing with matrix operations
        # Get all unique user IDs
        unique_user_ids = purchase_weights_pd['user_id'].unique()
        
        # Initialize dictionaries to store user embeddings and data
        user_purchase_history = {}
        
        # Create a DataFrame to map product_id to index in the embedding matrix
        product_id_to_idx = pd.DataFrame({
            'product_id': product_id_list,
            'idx': range(len(product_id_list))
        })
        
        # Process purchase history with matrix operations
        for user_id in unique_user_ids:
            # Get this user's purchases
            user_purchases = purchase_weights_pd[purchase_weights_pd['user_id'] == user_id]
            
            # Skip if no purchase history
            if len(user_purchases) == 0:
                continue
            
            # Get the product IDs and their weights
            user_product_ids = user_purchases['product_id'].tolist()
            user_product_weights = user_purchases['purchase_count'].tolist()
            
            # Filter to only include products in our embedding dictionary
            valid_indices = [i for i, pid in enumerate(user_product_ids) if pid in product_embedding_dict]
            if not valid_indices:
                continue
                
            filtered_product_ids = [user_product_ids[i] for i in valid_indices]
            filtered_weights = np.array([user_product_weights[i] for i in valid_indices])
            
            # Get product embeddings as a matrix (products × embedding_dim)
            user_product_embeddings = np.array([product_embedding_dict[pid] for pid in filtered_product_ids])
            
            # Normalize weights
            total_weight = filtered_weights.sum()
            if total_weight > 0:
                normalized_weights = filtered_weights / total_weight
                
                # Calculate weighted average embedding using matrix multiplication
                weighted_embedding = user_product_embeddings.T @ normalized_weights
                
                # Store in dictionary
                user_purchase_history[user_id] = {
                    'weighted_embedding': weighted_embedding,
                    'product_embeddings': user_product_embeddings
                }
        
        # Step 2b: Optimize cart history processing with matrix operations
        # Initialize cart history dictionary
        user_cart_history = {}
        
        # Process cart history with matrix operations
        for user_id in cart_weights_pd['user_id'].unique():
            # Get this user's cart items
            user_carts = cart_weights_pd[cart_weights_pd['user_id'] == user_id]
            
            # Skip if no cart history
            if len(user_carts) == 0:
                continue
            
            # Get the product IDs and their weights
            user_product_ids = user_carts['product_id'].tolist()
            user_product_weights = user_carts['cart_count'].tolist()
            
            # Filter to only include products in our embedding dictionary
            valid_indices = [i for i, pid in enumerate(user_product_ids) if pid in product_embedding_dict]
            if not valid_indices:
                continue
                
            filtered_product_ids = [user_product_ids[i] for i in valid_indices]
            filtered_weights = np.array([user_product_weights[i] for i in valid_indices])
            
            # Get product embeddings as a matrix (products × embedding_dim)
            user_product_embeddings = np.array([product_embedding_dict[pid] for pid in filtered_product_ids])
            
            # Normalize weights
            total_weight = filtered_weights.sum()
            if total_weight > 0:
                normalized_weights = filtered_weights / total_weight
                
                # Calculate weighted average embedding using matrix multiplication
                weighted_embedding = user_product_embeddings.T @ normalized_weights
                
                # Store in dictionary
                user_cart_history[user_id] = {
                    'weighted_embedding': weighted_embedding,
                    'product_embeddings': user_product_embeddings
                }
        
        logger.info(f"Created weighted embeddings for {len(user_purchase_history)} users' purchase history")
        logger.info(f"Created weighted embeddings for {len(user_cart_history)} users' cart history")
        
        # Step 3: Calculate distances efficiently using vectorized operations
        # Convert target data to pandas for processing
        target_pd = target_df.to_pandas()
        
        # Create matrices for batch processing
        # Get unique combinations of user_id and product_id
        user_product_pairs = target_pd[['user_id', 'product_id']].values
        unique_users = target_pd['user_id'].unique()
        unique_target_products = target_pd['product_id'].unique()
        
        # Compute target product embeddings matrix (only for products that exist in our dictionary)
        valid_target_products = [p for p in unique_target_products if p in product_embedding_dict]
        target_embeddings = np.array([product_embedding_dict[p] for p in valid_target_products])
        target_product_to_idx = {p: i for i, p in enumerate(valid_target_products)}
        
        # Helper function to calculate cosine similarity between matrices
        def batch_cosine_similarity(matrix_a, matrix_b):
            # Check if GPU acceleration should be used
            use_gpu = config.get('text_processing.use_gpu', True)
            
            # Try GPU acceleration for large matrices if cuML is available
            if use_gpu and (matrix_a.shape[0] > 1000 or matrix_b.shape[0] > 1000):
                try:
                    import cuml
                    # np is already imported globally
                    from cuml.metrics.pairwise_distances import cosine_similarity as cuml_cosine
                    logger.info("Using GPU-accelerated cosine similarity with cuML")
                    
                    # Execute on GPU
                    try:
                        similarity_matrix = cuml_cosine(matrix_a, matrix_b)
                        
                        # Convert back to numpy if needed
                        if hasattr(similarity_matrix, 'get'):
                            similarity_matrix = similarity_matrix.get()
                        else:
                            similarity_matrix = np.array(similarity_matrix)
                        
                        return similarity_matrix
                    except Exception as e:
                        logger.warning(f"Error with cuML cosine similarity: {str(e)}. Falling back to NumPy.")
                except ImportError:
                    logger.debug("cuML not available for cosine similarity, using NumPy instead")
            
            # Default CPU implementation
            # Normalize matrices
            norm_a = np.linalg.norm(matrix_a, axis=1, keepdims=True)
            norm_a = np.where(norm_a == 0, 1e-10, norm_a)  # Avoid division by zero
            
            norm_b = np.linalg.norm(matrix_b, axis=1, keepdims=True)
            norm_b = np.where(norm_b == 0, 1e-10, norm_b)  # Avoid division by zero
            
            matrix_a_normalized = matrix_a / norm_a
            matrix_b_normalized = matrix_b / norm_b
            
            # Calculate dot product
            similarity_matrix = np.dot(matrix_a_normalized, matrix_b_normalized.T)
            return similarity_matrix
        
        # Initialize array to store features
        target_features = []
        
        # Process targets more efficiently
        for _, row in target_pd.iterrows():
            user_id = row['user_id']
            product_id = row['product_id']
            
            # Initialize features
            purchase_weighted_similarity = 0
            cart_weighted_similarity = 0
            min_purchase_distance = 0
            min_cart_distance = 0
            
            # Process only if product is in our dictionary
            if product_id in product_embedding_dict:
                target_embedding = product_embedding_dict[product_id]
                
                # Helper function for cosine similarity calculation
                def calculate_cosine_similarity(vec_a, vec_b, use_cuml=False):
                    """Calculate cosine similarity with GPU acceleration option"""
                    if use_cuml:
                        try:
                            import cuml
                            from cuml.metrics.pairwise_distances import cosine_similarity as cuml_cosine
                            
                            # Reshape vectors to 2D for cuML
                            vec_a_2d = vec_a.reshape(1, -1)
                            vec_b_2d = vec_b.reshape(1, -1)
                            
                            # Calculate similarity on GPU
                            similarity = cuml_cosine(vec_a_2d, vec_b_2d)[0, 0]
                            
                            # Convert to numpy if needed
                            if hasattr(similarity, 'get'):
                                similarity = similarity.get()
                            
                            return similarity
                        except:
                            # Fall back to NumPy on error
                            pass
                    
                    # Default NumPy implementation
                    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-10)
                
                # Try to use GPU for vector similarity calculations if available
                use_gpu_for_vectors = False
                if config.get('text_processing.use_gpu', True):
                    try:
                        import cuml
                        use_gpu_for_vectors = True
                    except ImportError:
                        pass
                
                # Calculate similarity to purchase history
                if user_id in user_purchase_history:
                    # Calculate weighted similarity
                    purchase_history_embedding = user_purchase_history[user_id]['weighted_embedding']
                    purchase_weighted_similarity = calculate_cosine_similarity(
                        target_embedding, purchase_history_embedding, use_cuml=use_gpu_for_vectors
                    )
                    
                    # Calculate minimum distance to any purchased product using vectorized operations
                    user_product_matrix = user_purchase_history[user_id]['product_embeddings']
                    
                    # Try to use GPU for batch similarity if available and if we have many products
                    if use_gpu_for_vectors and len(user_product_matrix) > 100:
                        try:
                            import cuml
                            from cuml.metrics.pairwise_distances import cosine_similarity as cuml_cosine
                            
                            # Calculate on GPU
                            target_reshaped = target_embedding.reshape(1, -1)
                            similarity_matrix = cuml_cosine(user_product_matrix, target_reshaped)
                            
                            # Convert to numpy if needed
                            if hasattr(similarity_matrix, 'get'):
                                similarities = similarity_matrix.get().flatten()
                            else:
                                similarities = similarity_matrix.flatten()
                            
                            min_purchase_distance = similarities.max() if len(similarities) > 0 else 0
                        except Exception as e:
                            # Fall back to NumPy
                            similarities = np.dot(user_product_matrix, target_embedding) / (
                                np.linalg.norm(user_product_matrix, axis=1) * np.linalg.norm(target_embedding) + 1e-10
                            )
                            min_purchase_distance = similarities.max() if len(similarities) > 0 else 0
                    else:
                        # Default NumPy implementation
                        similarities = np.dot(user_product_matrix, target_embedding) / (
                            np.linalg.norm(user_product_matrix, axis=1) * np.linalg.norm(target_embedding) + 1e-10
                        )
                        min_purchase_distance = similarities.max() if len(similarities) > 0 else 0
                
                # Calculate similarity to cart history
                if user_id in user_cart_history:
                    # Calculate weighted similarity
                    cart_history_embedding = user_cart_history[user_id]['weighted_embedding']
                    cart_weighted_similarity = calculate_cosine_similarity(
                        target_embedding, cart_history_embedding, use_cuml=use_gpu_for_vectors
                    )
                    
                    # Calculate minimum distance to any carted product using vectorized operations
                    user_product_matrix = user_cart_history[user_id]['product_embeddings']
                    
                    # Try to use GPU for batch similarity if available and if we have many products
                    if use_gpu_for_vectors and len(user_product_matrix) > 100:
                        try:
                            import cuml
                            from cuml.metrics.pairwise_distances import cosine_similarity as cuml_cosine
                            
                            # Calculate on GPU
                            target_reshaped = target_embedding.reshape(1, -1)
                            similarity_matrix = cuml_cosine(user_product_matrix, target_reshaped)
                            
                            # Convert to numpy if needed
                            if hasattr(similarity_matrix, 'get'):
                                similarities = similarity_matrix.get().flatten()
                            else:
                                similarities = similarity_matrix.flatten()
                            
                            min_cart_distance = similarities.max() if len(similarities) > 0 else 0
                        except Exception as e:
                            # Fall back to NumPy
                            similarities = np.dot(user_product_matrix, target_embedding) / (
                                np.linalg.norm(user_product_matrix, axis=1) * np.linalg.norm(target_embedding) + 1e-10
                            )
                            min_cart_distance = similarities.max() if len(similarities) > 0 else 0
                    else:
                        # Default NumPy implementation
                        similarities = np.dot(user_product_matrix, target_embedding) / (
                            np.linalg.norm(user_product_matrix, axis=1) * np.linalg.norm(target_embedding) + 1e-10
                        )
                        min_cart_distance = similarities.max() if len(similarities) > 0 else 0
            
            # Store features
            target_features.append({
                'user_id': user_id,
                'product_id': product_id,
                'purchase_weighted_similarity': purchase_weighted_similarity,
                'cart_weighted_similarity': cart_weighted_similarity,
                'min_purchase_similarity': min_purchase_distance,
                'min_cart_similarity': min_cart_distance,
            })
        
        # Convert to DataFrame
        features_df = pd.DataFrame(target_features)
        
        # Convert to polars
        features_pl = pl.from_pandas(features_df)
        
        # Join with target DataFrame on user_id and product_id
        result = target_df.join(
            features_pl, 
            on=['user_id', 'product_id'], 
            how='left'
        ).fill_null(0)  # Fill missing values with 0
        
        logger.info(f"Generated user-product distance features for {len(result)} rows")
        
        return result
    
    @FeatureFactory.register('text_similarity_cluster', categorical_cols=['cluster'])
    def generate_text_similarity_cluster(
        history_df: pl.DataFrame, target_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Generate product similarity clusters based on text embeddings and
        calculate features based on user's interaction with these clusters.
        
        This feature aims to capture category-like patterns in product descriptions
        beyond the explicit category labels.
        
        Args:
            history_df: DataFrame containing user purchase history
            target_df: DataFrame containing target products to score
            
        Returns:
            DataFrame with added text similarity cluster features
        """
        # Get a logger
        import logging
        logger = logging.getLogger("TextSimilarityCluster")
        
        # Create text processor
        config = Config()
        text_processor = TextProcessor(config)
        
        # Step 1: Create embedding lookup for all products
        products = pl.concat([
            history_df.select('product_id', 'product_name').unique(),
            target_df.select('product_id', 'product_name').unique()
        ]).unique(subset=['product_id'])
        
        logger.info(f"Generating embeddings for {len(products)} unique products")
        
        # Get embeddings
        product_names = products['product_name'].to_list()
        product_embeddings = text_processor.get_embeddings(product_names)
        
        # Step 2: Cluster products using k-means (vectorized) with GPU acceleration if available
        try:
            import numpy as np
            import pandas as pd
            import time
            
            # Choose number of clusters
            n_clusters = config.get('text_processing.n_clusters', 15)
            
            # Check if GPU is available and if we should try to use cuML
            use_gpu = config.get('text_processing.use_gpu', True)
            use_cuml = False
            
            # Create and fit k-means model, with GPU acceleration if possible
            if use_gpu:
                logger.info("Checking for GPU support for K-means clustering...")
                try:
                    # First try to import cuML
                    import cuml
                    from cuml.cluster import KMeans as CuMLKMeans
                    use_cuml = True
                    logger.info("Using GPU-accelerated KMeans clustering with cuML")
                except ImportError:
                    logger.warning("cuML not available for KMeans, falling back to scikit-learn")
            
            # Track clustering time
            clustering_start = time.time()
            
            # Run the appropriate clustering algorithm
            if use_cuml:
                try:
                    # Use cuML's KMeans
                    kmeans = CuMLKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans.fit(product_embeddings)
                    # Get labels - with cuML we need to predict after fitting
                    cluster_labels = kmeans.predict(product_embeddings).astype(int)
                    if hasattr(cluster_labels, 'get'):  # Convert from GPU if needed
                        cluster_labels = cluster_labels.get()
                    else:
                        cluster_labels = np.array(cluster_labels)
                    logger.info(f"GPU KMeans clustering completed in {time.time() - clustering_start:.2f} seconds")
                except Exception as e:
                    logger.warning(f"Error with cuML KMeans: {str(e)}. Falling back to scikit-learn.")
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(product_embeddings)
                    logger.info(f"CPU KMeans clustering completed in {time.time() - clustering_start:.2f} seconds")
            else:
                # Use scikit-learn's KMeans
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(product_embeddings)
                logger.info(f"CPU KMeans clustering completed in {time.time() - clustering_start:.2f} seconds")
            
            # Create product_id to cluster mapping as a vectorized operation
            product_id_list = products['product_id'].to_list()
            
            # Create a DataFrame with product_id and cluster_label
            product_clusters_df = pd.DataFrame({
                'product_id': product_id_list,
                'cluster': cluster_labels
            })
            
            # Check cluster distribution
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            logger.info(f"Cluster distribution: {cluster_counts.to_dict()}")
            
            logger.info(f"Clustered {len(products)} products into {n_clusters} clusters")
            
            # Step 3: Calculate user interaction statistics with each cluster using vectorized operations
            # Get purchase history
            purchase_history = history_df.filter(pl.col('action_type') == "AT_Purchase").select(
                'user_id', 'product_id'
            )
            
            # Convert to pandas for faster operations
            purchase_history_pd = purchase_history.to_pandas()
            
            # Efficiently add cluster labels using merge instead of map
            purchase_history_with_clusters = purchase_history_pd.merge(
                product_clusters_df, on='product_id', how='left'
            )
            
            # Fill missing cluster values with -1
            purchase_history_with_clusters['cluster'] = purchase_history_with_clusters['cluster'].fillna(-1).astype(int)
            
            # Calculate statistics using optimized DataFrame operations
            # Get cluster counts per user-cluster pair (vectorized)
            user_cluster_matrix = pd.crosstab(
                purchase_history_with_clusters['user_id'], 
                purchase_history_with_clusters['cluster']
            ).reset_index()
            
            # Calculate total purchases per user (vectorized)
            user_total_purchases = user_cluster_matrix.set_index('user_id').sum(axis=1).reset_index(
                name='total_purchases'
            )
            
            # Convert the crosstab matrix to long format for easier merging
            user_cluster_counts = pd.melt(
                user_cluster_matrix, 
                id_vars=['user_id'], 
                var_name='cluster', 
                value_name='cluster_purchase_count'
            )
            
            # Merge to get total purchases
            user_cluster_counts = user_cluster_counts.merge(
                user_total_purchases, on='user_id', how='left'
            )
            
            # Calculate ratio vectorized
            user_cluster_counts['cluster_purchase_ratio'] = (
                user_cluster_counts['cluster_purchase_count'] / user_cluster_counts['total_purchases']
            )
            
            # Step 4: Add cluster labels and purchase ratios to target data
            # Convert target to pandas for processing
            target_pd = target_df.to_pandas()
            
            # Add cluster labels using merge instead of map (vectorized)
            target_with_clusters = target_pd.merge(
                product_clusters_df, on='product_id', how='left'
            )
            
            # Fill missing cluster values with -1
            target_with_clusters['cluster'] = target_with_clusters['cluster'].fillna(-1).astype(int)
            
            # Merge with user-cluster purchase ratios (vectorized)
            result_df = target_with_clusters.merge(
                user_cluster_counts[['user_id', 'cluster', 'cluster_purchase_ratio']], 
                on=['user_id', 'cluster'], 
                how='left'
            )
            
            # Fill missing values
            result_df['cluster_purchase_ratio'] = result_df['cluster_purchase_ratio'].fillna(0)
            
            # Convert back to polars
            result = pl.from_pandas(result_df)
            
            # Select only original columns + new features
            result = result.select(
                target_df.columns + ['cluster', 'cluster_purchase_ratio']
            )
            
            logger.info(f"Generated text similarity cluster features for {len(result)} rows")
            
            return result
            
        except ImportError:
            logger.warning("scikit-learn not available. Skipping text similarity clustering.")
            return target_df
    
    @FeatureFactory.register('text_diversity_features')
    def generate_text_diversity_features(
        history_df: pl.DataFrame, target_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Generate features measuring text diversity and novelty compared 
        to user's historical interactions.
        
        This feature aims to capture how different/novel a product is compared
        to what the user has already purchased or viewed.
        
        Args:
            history_df: DataFrame containing user purchase history
            target_df: DataFrame containing target products to score
            
        Returns:
            DataFrame with added text diversity features
        """
        # Get a logger
        import logging
        logger = logging.getLogger("TextDiversityFeatures")
        
        # Create text processor
        config = Config()
        text_processor = TextProcessor(config)
        
        # Step 1: Create embedding lookup for all products
        products = pl.concat([
            history_df.select('product_id', 'product_name').unique(),
            target_df.select('product_id', 'product_name').unique()
        ]).unique(subset=['product_id'])
        
        logger.info(f"Generating embeddings for {len(products)} unique products")
        
        # Get embeddings
        product_names = products['product_name'].to_list()
        product_embeddings = text_processor.get_embeddings(product_names)
        
        # Create embedding lookup dictionary and matrix
        product_id_list = products['product_id'].to_list()
        product_embedding_dict = {
            pid: embedding for pid, embedding in zip(product_id_list, product_embeddings)
        }
        
        # Step 2: Calculate user purchase history embedding statistics using vectorized operations
        # Get user purchase/view history
        user_history = history_df.filter(
            (pl.col('action_type') == "AT_Purchase") | (pl.col('action_type') == "AT_View")
        ).select('user_id', 'product_id', 'action_type')
        
        # Group by user to get unique products
        user_history_grouped = user_history.group_by('user_id', 'product_id').agg(
            pl.col('action_type').count().alias('interaction_count')
        )
        
        # Convert to pandas for vectorized operations
        user_history_pd = user_history_grouped.to_pandas()
        
        import numpy as np
        import pandas as pd
        
        # Create matrices for efficient computation
        user_stats = {}
        
        # Get unique users
        unique_user_ids = user_history_pd['user_id'].unique()
        
        # Prepare a dictionary mapping product_id to embedding array index
        product_id_to_idx = {pid: i for i, pid in enumerate(product_id_list)}
        
        # Create a master product embedding matrix
        product_embedding_matrix = np.array([product_embedding_dict[pid] for pid in product_id_list])
        
        # Compute user statistics using vectorized operations
        for user_id in unique_user_ids:
            # Get product IDs for this user
            user_products = user_history_pd[user_history_pd['user_id'] == user_id]['product_id'].tolist()
            
            # Get valid product IDs (ones that have embeddings)
            valid_product_ids = [pid for pid in user_products if pid in product_embedding_dict]
            
            if len(valid_product_ids) == 0:
                continue
                
            # Get indices of embeddings in the embedding matrix
            embedding_indices = [product_id_to_idx[pid] for pid in valid_product_ids]
            
            # Extract embeddings as a matrix
            user_product_matrix = product_embedding_matrix[embedding_indices]
            
            # Calculate centroid vectorized
            embedding_centroid = np.mean(user_product_matrix, axis=0)
            
            # Calculate distances from centroid vectorized
            # Reshape centroid for broadcasting
            centroid_reshaped = embedding_centroid.reshape(1, -1)
            
            # Calculate distances from all points to centroid at once
            distances = np.linalg.norm(user_product_matrix - centroid_reshaped, axis=1)
            
            # Calculate diversity (mean distance)
            embedding_diversity = np.mean(distances) if len(distances) > 0 else 0
            
            # Store in dictionary
            user_stats[user_id] = {
                'embedding_centroid': embedding_centroid,
                'embedding_diversity': embedding_diversity
            }
        
        logger.info(f"Calculated embedding statistics for {len(user_stats)} users")
        
        # Step 3: Calculate diversity features for target items using vectorized operations
        # Convert target data for easier processing
        target_pd = target_df.to_pandas()
        
        # Extract user_ids and product_ids
        target_user_ids = target_pd['user_id'].values
        target_product_ids = target_pd['product_id'].values
        
        # Initialize feature arrays
        num_targets = len(target_pd)
        distances_from_centroid = np.zeros(num_targets)
        relative_diversities = np.zeros(num_targets)
        
        # Process targets in batches based on user_id for more efficient computation
        for i, (user_id, product_id) in enumerate(zip(target_user_ids, target_product_ids)):
            if product_id in product_embedding_dict and user_id in user_stats:
                target_embedding = product_embedding_dict[product_id]
                centroid = user_stats[user_id]['embedding_centroid']
                
                # Calculate distance from centroid
                distance = np.linalg.norm(target_embedding - centroid)
                distances_from_centroid[i] = distance
                
                # Calculate relative diversity
                user_diversity = user_stats[user_id]['embedding_diversity']
                if user_diversity > 0:
                    relative_diversities[i] = distance / user_diversity
                else:
                    relative_diversities[i] = distance
        
        # Create feature DataFrame in one go
        features_df = pd.DataFrame({
            'user_id': target_user_ids,
            'product_id': target_product_ids,
            'distance_from_centroid': distances_from_centroid,
            'relative_diversity': relative_diversities
        })
        
        # Convert to polars
        features_pl = pl.from_pandas(features_df)
        
        # Join with target DataFrame on user_id and product_id
        result = target_df.join(
            features_pl, 
            on=['user_id', 'product_id'], 
            how='left'
        ).fill_null(0)  # Fill missing values with 0
        
        logger.info(f"Generated text diversity features for {len(result)} rows")
        
        return result

    return True
