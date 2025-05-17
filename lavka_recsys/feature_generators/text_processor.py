import numpy as np
import polars as pl
from typing import List, Dict, Any, Tuple, Optional, Union
import re
from sentence_transformers import SentenceTransformer
import gensim
import gensim.downloader as api
import fasttext
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from ..utils.custom_logging import get_logger
from ..utils.config import Config
from ..feature_factory import FeatureFactory # Assuming FeatureFactory is in the parent directory

# Module-level cache for TextProcessor instance and its configuration
_text_processor_instance: Optional[SentenceTransformer] = None
_text_processor_config_cache: Optional[Config] = None
_logger_instance = get_logger(__name__) # Module level logger

# --- TextProcessor Class (Handles Model Loading and Embedding Generation) ---
class TextProcessor:
    """Simplified text processing with pretrained models"""

    def __init__(self, config: Config):
        """Initialize text processor with configuration"""
        self.config = config
        self.logger = _logger_instance # Use module logger
        self.model: Union[SentenceTransformer, gensim.models.KeyedVectors, fasttext.FastText._FastText, None] = None
        self.embedding_size: int = 0
        self.model_type: str = ""

        model_type_config = self.config.get('text_processing.model_type', 'sentence-transformers')

        if model_type_config == 'sentence-transformers':
            self._load_sentence_transformers()
        elif model_type_config == 'word2vec':
            self._load_word2vec()
        elif model_type_config == 'fasttext':
            self._load_fasttext()
        else:
            self.logger.warning(
                f"Unknown model type: {model_type_config}. "
                "Falling back to sentence-transformers."
            )
            self._load_sentence_transformers()

    def _load_sentence_transformers(self):
        """Load SentenceTransformers model"""
        try
            model_name = self.config.get(
                'text_processing.model_name',
                'paraphrase-multilingual-MiniLM-L12-v2'
            )
            self.model = SentenceTransformer(model_name)
            # Type assertion for clarity, as get_sentence_embedding_dimension can return None
            embedding_dim = self.model.get_sentence_embedding_dimension()
            if embedding_dim is None:
                raise ValueError("SentenceTransformer model embedding dimension is None.")
            self.embedding_size = embedding_dim
            self.model_type = 'sentence-transformers'
            self.logger.info(f"Loaded sentence-transformers model: {model_name}")
        except ImportError:
            self.logger.error(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None
        except Exception as e:
            self.logger.error(f"Error loading SentenceTransformer model: {e}")
            self.model = None


    def _load_word2vec(self):
        """Load Word2Vec model"""
        try:
            model_name = self.config.get(
                'text_processing.model_name',
                'word2vec-ruscorpora-300'
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
        except Exception as e:
            self.logger.error(f"Error loading Word2Vec model: {e}")
            self.model = None

    def _load_fasttext(self):
        """Load FastText model"""
        try:
            model_path = self.config.get(
                'text_processing.model_path',
                'cc.ru.300.bin'
            )
            self.model = fasttext.load_model(model_path)
            self.embedding_size = self.model.get_dimension()
            self.model_type = 'fasttext'
            self.logger.info(f"Loaded fasttext model: {model_path}")
        except ImportError:
            self.logger.error(
                "fasttext not available. "
                "Install with: pip install fasttext-wheel or fasttext"
            )
            self.model = None
        except Exception as e:
            self.logger.error(f"Error loading FastText model: {e}")
            self.model = None
            
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts using the pretrained model.
        Returns zero embeddings of expected size if model fails.
        """
        if self.model is None or self.embedding_size == 0:
            self.logger.warning("No text model loaded or embedding size is zero. Returning zero embeddings.")
            # Return a 2D array with shape (len(texts), 1) or (len(texts), default_dim)
            # To avoid downstream errors, it's better to have a consistent (if arbitrary) dim.
            # However, if embedding_size is 0, it means model loading failed critically.
            # Let's return a (N,1) array of zeros to signal an issue but not break array operations.
            return np.zeros((len(texts), 1))


        cleaned_texts = [str(t) if t is not None and str(t).strip() else " " for t in texts]

        try:
            if self.model_type == 'sentence-transformers':
                assert isinstance(self.model, SentenceTransformer)
                embeddings = self.model.encode(
                    cleaned_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
            elif self.model_type == 'word2vec':
                assert isinstance(self.model, gensim.models.KeyedVectors)
                embeddings = np.zeros((len(cleaned_texts), self.embedding_size))
                for i, text in enumerate(cleaned_texts):
                    words = text.lower().split()
                    vectors = [self.model[word] for word in words if word in self.model]
                    if vectors:
                        embeddings[i] = np.mean(vectors, axis=0)
            elif self.model_type == 'fasttext':
                assert hasattr(self.model, 'get_sentence_vector')
                embeddings = np.array([
                    self.model.get_sentence_vector(text)
                    for text in cleaned_texts
                ])
            else:
                self.logger.warning(f"Unknown model type during embedding: {self.model_type}")
                return np.zeros((len(texts), 1))

            return embeddings if embeddings is not None else np.zeros((len(texts), self.embedding_size if self.embedding_size > 0 else 1))

        except Exception as e:
            self.logger.error(f"Error during embedding generation with {self.model_type}: {e}")
            return np.zeros((len(texts), self.embedding_size if self.embedding_size > 0 else 1))

    def reduce_dimensions(self, embeddings: np.ndarray, dimensions: int = 20) -> np.ndarray:
        """Reduce embedding dimensions using PCA."""
        if embeddings.shape[1] == 0: # No embeddings to reduce
            self.logger.warning("Attempted to reduce zero-dimensional embeddings. Returning as is.")
            return embeddings
        if embeddings.shape[1] <= dimensions:
            return embeddings
        if embeddings.shape[0] < dimensions: # Not enough samples for PCA
            self.logger.warning(
                f"Not enough samples ({embeddings.shape[0]}) to reduce to {dimensions} dimensions using PCA. "
                f"Returning original embeddings with {embeddings.shape[1]} dimensions."
            )
            return embeddings

        try:
            pca = PCA(n_components=dimensions, random_state=42)
            reduced = pca.fit_transform(embeddings)
            self.logger.info(
                f"Reduced embeddings from {embeddings.shape[1]} to {dimensions} dimensions using PCA."
            )
            return reduced
        except ImportError:
            self.logger.warning(
                "scikit-learn not available for dimension reduction. "
                "Install with: pip install scikit-learn. Using original embeddings."
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Error during PCA dimensionality reduction: {e}. Using original embeddings.")
            return embeddings

# --- Utility function to get a singleton TextProcessor instance ---
def get_text_processor(config: Config) -> TextProcessor:
    """
    Initializes and returns a singleton TextProcessor instance.
    Re-initializes if crucial model configuration changes.
    """
    global _text_processor_instance, _text_processor_config_cache

    current_model_type = config.get('text_processing.model_type')
    current_model_name = config.get('text_processing.model_name', config.get('text_processing.model_path'))

    if _text_processor_instance is None or \
       _text_processor_config_cache is None or \
       _text_processor_config_cache.get('text_processing.model_type') != current_model_type or \
       _text_processor_config_cache.get('text_processing.model_name', _text_processor_config_cache.get('text_processing.model_path')) != current_model_name:
        
        _logger_instance.info(f"Initializing TextProcessor with model: {current_model_type} - {current_model_name}")
        _text_processor_instance = TextProcessor(config)
        _text_processor_config_cache = config.copy() # Cache the config used for initialization
    
    return _text_processor_instance


# --- Helper to get product embeddings (name or category) ---
_product_embeddings_cache: Dict[str, pl.DataFrame] = {}

def _get_product_embeddings_df(
    history_df: pl.DataFrame,
    target_df: pl.DataFrame,
    text_processor: TextProcessor,
    config: Config,
    id_col: str, # 'product_id' or 'product_category'
    text_col: str # 'product_name' or 'product_category'
) -> pl.DataFrame:
    """
    Generates and caches embeddings for unique product IDs or categories.
    Returns a Polars DataFrame with [id_col, embed_0, embed_1, ...].
    """
    cache_key = f"{id_col}_{text_processor.model_type}_{text_processor.config.get('text_processing.model_name', text_processor.config.get('text_processing.model_path'))}"
    
    if cache_key in _product_embeddings_cache:
        _logger_instance.info(f"Using cached embeddings for {id_col}")
        return _product_embeddings_cache[cache_key]

    _logger_instance.info(f"Generating embeddings for unique {id_col} from {text_col}")

    # Get unique items and their text
    unique_items_hist = history_df.select(id_col, text_col).unique(subset=[id_col])
    unique_items_target = target_df.select(id_col, text_col).unique(subset=[id_col])
    
    all_unique_items = pl.concat([unique_items_hist, unique_items_target]).unique(subset=[id_col], keep='first')
    
    item_texts = all_unique_items[text_col].to_list()
    item_ids = all_unique_items[id_col]

    if not item_texts:
        _logger_instance.warning(f"No unique texts found for {id_col}. Returning empty embeddings DataFrame.")
        # Create an empty DataFrame with the expected structure if possible
        dim = config.get('text_processing.embedding_dimensions', 10) # Default to 10 if not reducible
        if text_processor.embedding_size > 0 and text_processor.embedding_size < dim:
            dim = text_processor.embedding_size
        elif text_processor.embedding_size == 0: # Model loading failed
            dim = 1 
        
        embed_cols = [f"{id_col.replace('_id','').replace('_category','cat')}_embed_{i}" for i in range(dim)]
        schema = {id_col: pl.Unknown} # Use Unknown or infer type if possible
        for col in embed_cols:
            schema[col] = pl.Float32 
        return pl.DataFrame(schema=schema)


    embeddings = text_processor.get_embeddings(item_texts)

    # Reduce dimensions if configured and embeddings are valid
    if embeddings.shape[1] > 1: # Only reduce if embeddings are not the dummy (N,1) zeros
        dimensions = config.get('text_processing.embedding_dimensions', 20) # Default from original code
        if embeddings.shape[1] > dimensions:
            embeddings = text_processor.reduce_dimensions(embeddings, dimensions)
    
    num_embedding_dims = embeddings.shape[1]
    feature_names = [f"{id_col.replace('_id','').replace('_category','cat')}_embed_{i}" for i in range(num_embedding_dims)]

    # Create Polars DataFrame
    data_dict = {id_col: item_ids}
    for i, name in enumerate(feature_names):
        data_dict[name] = embeddings[:, i]
    
    embed_df = pl.DataFrame(data_dict)
    _product_embeddings_cache[cache_key] = embed_df
    _logger_instance.info(f"Generated and cached embeddings for {id_col} with shape {embed_df.shape}")
    return embed_df

# --- Feature Generator Registration ---
def register_text_embedding_fgens():
    """Register text embedding methods with FeatureFactory"""

    # Note: The `num_cols` in @FeatureFactory.register are hardcoded lists.
    # The functions below generate column names dynamically based on config.
    # If `embedding_dimensions` in config changes, these lists in decorators MUST be updated,
    # or FeatureFactory needs to be adapted to handle dynamic column names/counts.

    @FeatureFactory.register(
        'product_embeddings',
        # This list must match the number of dimensions configured in 'text_processing.embedding_dimensions'
        # or the raw embedding size if reduction is skipped or fails.
        # Example for 20 dimensions:
        num_cols=[f'product_embed_{i}' for i in range(20)] 
    )
    def generate_product_embeddings(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate product name embeddings using pretrained model"""
        text_processor = get_text_processor(config)
        
        product_embed_df = _get_product_embeddings_df(
            history_df, target_df, text_processor, config, 'product_id', 'product_name'
        )
        
        # Ensure column names match the decorator if it's strict
        # This example assumes the decorator's num_cols is correctly set for 20 dims
        # If dimensions are different, the join might fail or have wrong columns.
        # It's safer if FeatureFactory infers columns from the returned df.
        
        return target_df.join(product_embed_df, on='product_id', how='left').fill_null(0)

    @FeatureFactory.register(
        'category_embeddings',
        # Example for 10 dimensions:
        num_cols=[f'cat_embed_{i}' for i in range(10)] 
    )
    def generate_category_embeddings(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        """Generate category name embeddings using pretrained model"""
        text_processor = get_text_processor(config)

        # Ensure 'product_category' exists and handle potential nulls before unique
        history_df = history_df.with_columns(pl.col("product_category").fill_null("UNKNOWN_CATEGORY"))
        target_df = target_df.with_columns(pl.col("product_category").fill_null("UNKNOWN_CATEGORY"))

        category_embed_df = _get_product_embeddings_df(
            history_df, target_df, text_processor, config, 'product_category', 'product_category'
        )
        return target_df.join(category_embed_df, on='product_category', how='left').fill_null(0)

    @FeatureFactory.register(
        'user_product_similarity', # Renamed from distance to similarity
        num_cols=[
            'purchase_weighted_similarity',
            'cart_weighted_similarity',
            'max_purchase_similarity_history', # Renamed
            'max_cart_similarity_history',     # Renamed
        ]
    )
    def generate_user_product_similarity(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        text_processor = get_text_processor(config)
        logger = _logger_instance

        logger.info("Starting user_product_similarity generation.")

        # 1. Get all product embeddings
        # Ensure 'product_name' exists and handle potential nulls
        history_df_p = history_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT"))
        target_df_p = target_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT"))

        all_product_embeddings_df = _get_product_embeddings_df(
            history_df_p, target_df_p, text_processor, config, 'product_id', 'product_name'
        )
        
        if all_product_embeddings_df.height == 0 or all_product_embeddings_df.width <=1 : # only id_col
            logger.warning("No product embeddings available for user_product_similarity. Returning target_df.")
            return target_df.with_columns([
                pl.lit(0.0).alias('purchase_weighted_similarity'),
                pl.lit(0.0).alias('cart_weighted_similarity'),
                pl.lit(0.0).alias('max_purchase_similarity_history'),
                pl.lit(0.0).alias('max_cart_similarity_history'),
            ])

        embedding_cols = [col for col in all_product_embeddings_df.columns if col.startswith('product_embed_')]
        if not embedding_cols:
            logger.warning("Embedding columns not found in all_product_embeddings_df. Returning target_df.")
            return target_df # Or add default zero columns as above

        # Convert to NumPy for easier vector operations, store in dict for lookup
        product_embedding_dict = {
            row[0]: np.array(row[1:]) 
            for row in all_product_embeddings_df.select(['product_id'] + embedding_cols).iter_rows()
        }
        logger.info(f"Created embedding lookup for {len(product_embedding_dict)} products.")

        # 2. Calculate user history embeddings (weighted by frequency or recency)
        def get_user_history_embeddings(action_type: str, weight_col_name: str, hist_embeddings_col_name: str):
            user_hist = history_df.filter(pl.col('action_type') == action_type) \
                .group_by(['user_id', 'product_id']) \
                .agg(pl.count().alias('interaction_count')) # Simple frequency weight

            # Join with product embeddings
            user_hist = user_hist.join(all_product_embeddings_df, on='product_id', how='inner')
            
            # Calculate weighted average embedding per user
            # This requires converting list of embeddings to something that can be averaged.
            # Polars' support for list aggregations can be tricky. Let's use apply.
            
            user_weighted_embeddings = []
            user_all_historical_embeddings = []

            for user_id, group_df in user_hist.group_by('user_id', maintain_order=False):
                embeddings_list = [product_embedding_dict.get(pid) for pid in group_df['product_id'].to_list()]
                valid_embeddings = [emb for emb in embeddings_list if emb is not None and emb.ndim > 0 and emb.shape[0] > 0]
                
                if not valid_embeddings:
                    continue

                weights = group_df['interaction_count'].to_numpy().astype(float) # Ensure float for division
                
                # Filter weights corresponding to valid_embeddings (if some products had no embeddings)
                # This alignment is tricky if embeddings_list has Nones.
                # Assuming all products in user_hist will have embeddings from all_product_embeddings_df due to inner join.
                
                weighted_avg_emb = np.average(np.array(valid_embeddings), axis=0, weights=weights)
                user_weighted_embeddings.append({'user_id': user_id, weight_col_name: weighted_avg_emb.tolist()})
                user_all_historical_embeddings.append({'user_id': user_id, hist_embeddings_col_name: valid_embeddings})

            weighted_df = pl.DataFrame(user_weighted_embeddings, schema={'user_id':pl.Utf8, weight_col_name: pl.List(pl.Float64)}) if user_weighted_embeddings else pl.DataFrame({'user_id': [], weight_col_name: []}, schema={'user_id':pl.Utf8, weight_col_name: pl.List(pl.Float64)})
            historical_df = pl.DataFrame(user_all_historical_embeddings, schema={'user_id':pl.Utf8, hist_embeddings_col_name: pl.Object}) if user_all_historical_embeddings else pl.DataFrame({'user_id': [], hist_embeddings_col_name: []}, schema={'user_id':pl.Utf8, hist_embeddings_col_name: pl.Object})
            
            return weighted_df, historical_df

        purchase_weighted_emb_df, purchase_hist_emb_df = get_user_history_embeddings(
            "AT_Purchase", "purchase_weighted_embedding", "purchase_historical_embeddings"
        )
        cart_weighted_emb_df, cart_hist_emb_df = get_user_history_embeddings(
            "AT_CartUpdate", "cart_weighted_embedding", "cart_historical_embeddings"
        )
        logger.info(f"Processed purchase history for {purchase_weighted_emb_df.height} users.")
        logger.info(f"Processed cart history for {cart_weighted_emb_df.height} users.")

        # 3. Join with target_df and calculate similarities
        target_with_embeddings = target_df.join(all_product_embeddings_df, on='product_id', how='left')
        
        # Join user history embeddings
        target_with_embeddings = target_with_embeddings.join(purchase_weighted_emb_df, on='user_id', how='left')
        target_with_embeddings = target_with_embeddings.join(purchase_hist_emb_df, on='user_id', how='left')
        target_with_embeddings = target_with_embeddings.join(cart_weighted_emb_df, on='user_id', how='left')
        target_with_embeddings = target_with_embeddings.join(cart_hist_emb_df, on='user_id', how='left')

        # Prepare for similarity calculation (convert to NumPy for batch processing)
        # This part can be slow if target_with_embeddings is huge.
        # Consider doing it in chunks or using more advanced Polars UDFs if performance is an issue.
        
        results = []
        for row_tuple in target_with_embeddings.select(['user_id', 'product_id'] + embedding_cols + [
            'purchase_weighted_embedding', 'purchase_historical_embeddings',
            'cart_weighted_embedding', 'cart_historical_embeddings'
        ]).iter_rows(named=True): # iter_rows(named=True) for dicts
            
            target_emb_list = [row_tuple[col] for col in embedding_cols]
            if any(x is None for x in target_emb_list): # Skip if target embedding is missing
                results.append({'user_id': row_tuple['user_id'], 'product_id': row_tuple['product_id'],
                                'purchase_weighted_similarity': 0.0, 'cart_weighted_similarity': 0.0,
                                'max_purchase_similarity_history': 0.0, 'max_cart_similarity_history': 0.0})
                continue
            target_emb = np.array(target_emb_list)


            sim_purchase_weighted = 0.0
            sim_cart_weighted = 0.0
            max_sim_purchase_hist = 0.0
            max_sim_cart_hist = 0.0

            if row_tuple['purchase_weighted_embedding']:
                user_purchase_emb = np.array(row_tuple['purchase_weighted_embedding'])
                if target_emb.shape == user_purchase_emb.shape and np.linalg.norm(target_emb) > 0 and np.linalg.norm(user_purchase_emb) > 0:
                     sim_purchase_weighted = sklearn_cosine_similarity(target_emb.reshape(1, -1), user_purchase_emb.reshape(1, -1))[0, 0]
            
            if row_tuple['purchase_historical_embeddings']:
                hist_embeds = [np.array(e) for e in row_tuple['purchase_historical_embeddings'] if np.array(e).size > 0 and np.array(e).shape == target_emb.shape]
                if hist_embeds and np.linalg.norm(target_emb) > 0:
                    similarities = sklearn_cosine_similarity(target_emb.reshape(1, -1), np.array(hist_embeds))
                    if similarities.size > 0:
                        max_sim_purchase_hist = np.max(similarities)
            
            if row_tuple['cart_weighted_embedding']:
                user_cart_emb = np.array(row_tuple['cart_weighted_embedding'])
                if target_emb.shape == user_cart_emb.shape and np.linalg.norm(target_emb) > 0 and np.linalg.norm(user_cart_emb) > 0:
                    sim_cart_weighted = sklearn_cosine_similarity(target_emb.reshape(1, -1), user_cart_emb.reshape(1, -1))[0, 0]

            if row_tuple['cart_historical_embeddings']:
                hist_embeds = [np.array(e) for e in row_tuple['cart_historical_embeddings'] if np.array(e).size > 0 and np.array(e).shape == target_emb.shape]
                if hist_embeds and np.linalg.norm(target_emb) > 0:
                    similarities = sklearn_cosine_similarity(target_emb.reshape(1, -1), np.array(hist_embeds))
                    if similarities.size > 0:
                        max_sim_cart_hist = np.max(similarities)
            
            results.append({
                'user_id': row_tuple['user_id'], 
                'product_id': row_tuple['product_id'],
                'purchase_weighted_similarity': sim_purchase_weighted,
                'cart_weighted_similarity': sim_cart_weighted,
                'max_purchase_similarity_history': max_sim_purchase_hist,
                'max_cart_similarity_history': max_sim_cart_hist
            })
        
        features_df = pl.DataFrame(results)
        final_df = target_df.join(features_df, on=['user_id', 'product_id'], how='left').fill_null(0)
        logger.info("Finished user_product_similarity generation.")
        return final_df


    @FeatureFactory.register(
        'text_similarity_cluster',
        cat_cols=['product_text_cluster'], # Renamed for clarity
        num_cols=['cluster_purchase_ratio', 'cluster_cart_ratio', 'cluster_view_ratio']
    )
    def generate_text_similarity_cluster(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        text_processor = get_text_processor(config)
        logger = _logger_instance
        logger.info("Starting text_similarity_cluster generation.")

        history_df_p = history_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT"))
        target_df_p = target_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT"))

        all_product_embeddings_df = _get_product_embeddings_df(
            history_df_p, target_df_p, text_processor, config, 'product_id', 'product_name'
        )
        
        if all_product_embeddings_df.height == 0 or all_product_embeddings_df.width <=1:
            logger.warning("No product embeddings for text_similarity_cluster. Returning target_df.")
            return target_df.with_columns([
                pl.lit(-1).alias('product_text_cluster').cast(pl.Int32), # Use -1 for unknown cluster
                pl.lit(0.0).alias('cluster_purchase_ratio'),
                pl.lit(0.0).alias('cluster_cart_ratio'),
                pl.lit(0.0).alias('cluster_view_ratio'),
            ])

        embedding_cols = [col for col in all_product_embeddings_df.columns if col.startswith('product_embed_')]
        if not embedding_cols:
             logger.warning("Embedding columns not found for text_similarity_cluster. Returning target_df.")
             return target_df # Or add default zero columns as above

        product_embeddings_np = all_product_embeddings_df[embedding_cols].to_numpy()
        product_ids_series = all_product_embeddings_df['product_id']

        if product_embeddings_np.shape[0] == 0: # No products with embeddings
            logger.warning("No product embeddings numpy array for text_similarity_cluster. Returning target_df.")
            return target_df # Or add default zero columns as above


        try:
            n_clusters = config.get('text_processing.n_clusters', 15)
            if product_embeddings_np.shape[0] < n_clusters:
                logger.warning(f"Number of products with embeddings ({product_embeddings_np.shape[0]}) is less than n_clusters ({n_clusters}). Adjusting n_clusters.")
                n_clusters = max(1, product_embeddings_np.shape[0]) # Ensure at least 1 cluster

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto' if hasattr(KMeans(), 'n_init') and KMeans().n_init == 'auto' else 10) # n_init='auto' for newer sklearn
            cluster_labels = kmeans.fit_predict(product_embeddings_np)
            
            product_cluster_df = pl.DataFrame({
                'product_id': product_ids_series,
                'product_text_cluster': cluster_labels.astype(np.int32)
            })
            logger.info(f"Clustered {len(product_ids_series)} products into {n_clusters} clusters.")

            # Calculate user interaction stats with clusters
            history_with_clusters = history_df.join(product_cluster_df, on='product_id', how='left') \
                                            .fill_null({'product_text_cluster': -1}) # Handle products not in cluster_df

            def calculate_cluster_ratios(action_type: str, ratio_col_name: str) -> pl.DataFrame:
                user_cluster_action = history_with_clusters.filter(pl.col('action_type') == action_type) \
                    .group_by(['user_id', 'product_text_cluster']) \
                    .agg(pl.count().alias('action_in_cluster_count'))
                
                user_total_actions = user_cluster_action.group_by('user_id') \
                    .agg(pl.sum('action_in_cluster_count').alias('total_user_actions'))
                
                user_cluster_ratios = user_cluster_action.join(user_total_actions, on='user_id', how='left') \
                    .with_columns(
                        (pl.col('action_in_cluster_count') / pl.col('total_user_actions')).alias(ratio_col_name)
                    ) \
                    .select(['user_id', 'product_text_cluster', ratio_col_name])
                return user_cluster_ratios

            purchase_ratios = calculate_cluster_ratios("AT_Purchase", "cluster_purchase_ratio")
            cart_ratios = calculate_cluster_ratios("AT_CartUpdate", "cluster_cart_ratio")
            view_ratios = calculate_cluster_ratios("AT_View", "cluster_view_ratio") # Added view ratio

            # Add cluster labels and ratios to target data
            result_df = target_df.join(product_cluster_df, on='product_id', how='left')
            result_df = result_df.join(purchase_ratios, on=['user_id', 'product_text_cluster'], how='left')
            result_df = result_df.join(cart_ratios, on=['user_id', 'product_text_cluster'], how='left')
            result_df = result_df.join(view_ratios, on=['user_id', 'product_text_cluster'], how='left')
            
            final_df = result_df.fill_null({
                'product_text_cluster': -1, # Default for products not clustered
                'cluster_purchase_ratio': 0.0,
                'cluster_cart_ratio': 0.0,
                'cluster_view_ratio': 0.0
            }).with_columns(pl.col('product_text_cluster').cast(pl.Int32)) # Ensure correct type for cat feature

            logger.info(f"Generated text similarity cluster features for {final_df.height} rows.")
            return final_df

        except ImportError:
            logger.warning("scikit-learn not available for KMeans. Skipping text similarity clustering.")
            return target_df.with_columns([
                pl.lit(-1).alias('product_text_cluster').cast(pl.Int32),
                pl.lit(0.0).alias('cluster_purchase_ratio'),
                pl.lit(0.0).alias('cluster_cart_ratio'),
                pl.lit(0.0).alias('cluster_view_ratio'),
            ])
        except Exception as e:
            logger.error(f"Error during text_similarity_cluster: {e}")
            return target_df # Or add default zero columns as above

    @FeatureFactory.register(
        'text_diversity_features',
        num_cols=['distance_from_centroid', 'relative_diversity']
    )
    def generate_text_diversity_features(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        text_processor = get_text_processor(config)
        logger = _logger_instance
        logger.info("Starting text_diversity_features generation.")

        history_df_p = history_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT"))
        target_df_p = target_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT"))

        all_product_embeddings_df = _get_product_embeddings_df(
            history_df_p, target_df_p, text_processor, config, 'product_id', 'product_name'
        )
        
        if all_product_embeddings_df.height == 0 or all_product_embeddings_df.width <=1:
            logger.warning("No product embeddings for text_diversity_features. Returning target_df.")
            return target_df.with_columns([
                pl.lit(0.0).alias('distance_from_centroid'),
                pl.lit(0.0).alias('relative_diversity'),
            ])

        embedding_cols = [col for col in all_product_embeddings_df.columns if col.startswith('product_embed_')]
        if not embedding_cols:
             logger.warning("Embedding columns not found for text_diversity_features. Returning target_df.")
             return target_df # Or add default zero columns as above


        # Calculate user history embedding statistics (centroid, diversity)
        user_history_actions = history_df.filter(
            pl.col('action_type').is_in(["AT_Purchase", "AT_View"]) # Consider CartUpdate too?
        )
        user_history_with_embeds = user_history_actions.join(all_product_embeddings_df, on='product_id', how='inner')

        # Define a function to calculate stats for a group of embeddings
        def calculate_embedding_stats(embeddings_list_col: pl.Series) -> pl.Series:
            stats_list = []
            for embeddings_list in embeddings_list_col: # embeddings_list is a list of lists (embeddings)
                if not embeddings_list or not any(embeddings_list): # Check if list is empty or contains only empty/None
                    stats_list.append({'centroid': None, 'diversity': 0.0})
                    continue
                
                # Convert list of lists/tuples to NumPy array
                valid_embeddings = [np.array(e) for e in embeddings_list if e is not None and len(e) > 0]
                if not valid_embeddings:
                    stats_list.append({'centroid': None, 'diversity': 0.0})
                    continue

                embeddings_np = np.array(valid_embeddings)
                centroid = np.mean(embeddings_np, axis=0)
                if embeddings_np.shape[0] > 1:
                    distances = np.linalg.norm(embeddings_np - centroid, axis=1)
                    diversity = np.mean(distances)
                else: # Single item history, diversity is 0
                    diversity = 0.0
                stats_list.append({'centroid': centroid.tolist(), 'diversity': diversity})
            return pl.Series(stats_list)


        # Group by user and collect all their historical embeddings
        user_historical_embeddings_collected = user_history_with_embeds.group_by('user_id').agg(
            pl.col(embedding_cols[0]).alias(embedding_cols[0]+"_list"), # Hacky way to get list of lists
            # A proper way would be to aggregate into a list of structs/tuples if Polars supports it easily
            # For now, we'll reconstruct the list of embeddings per user
        )
        
        # This part is tricky in Polars directly. We might need to iterate user groups or use a UDF that's efficient.
        # Let's simplify by iterating user groups from pandas for stats calculation
        user_stats_list = []
        for user_id, group_df in user_history_with_embeds.group_by('user_id'):
            user_embeds_np = group_df[embedding_cols].to_numpy()
            if user_embeds_np.size == 0:
                user_stats_list.append({'user_id': user_id, 'embedding_centroid': None, 'embedding_diversity': 0.0})
                continue
            
            centroid = np.mean(user_embeds_np, axis=0)
            diversity = 0.0
            if user_embeds_np.shape[0] > 1: # Need at least 2 points for diversity from centroid
                 distances = np.linalg.norm(user_embeds_np - centroid, axis=1)
                 diversity = np.mean(distances)

            user_stats_list.append({
                'user_id': user_id,
                'embedding_centroid': centroid.tolist(),
                'embedding_diversity': diversity
            })
        
        user_stats_df = pl.DataFrame(user_stats_list, schema_overrides={'embedding_centroid': pl.List(pl.Float64)}) if user_stats_list else pl.DataFrame({'user_id':[], 'embedding_centroid':[], 'embedding_diversity':[]}, schema_overrides={'embedding_centroid': pl.List(pl.Float64), 'user_id':pl.Utf8, 'embedding_diversity':pl.Float64})
        logger.info(f"Calculated embedding statistics for {user_stats_df.height} users.")

        # Join with target data and calculate diversity features
        target_with_user_stats = target_df.join(all_product_embeddings_df, on='product_id', how='left') \
                                          .join(user_stats_df, on='user_id', how='left')
        
        # Calculate features row by row (can be slow, consider UDF or expression-based if possible)
        # This is similar to the user_product_similarity loop.
        diversity_features_list = []
        for row_tuple in target_with_user_stats.select(['user_id', 'product_id'] + embedding_cols + 
                                                       ['embedding_centroid', 'embedding_diversity']).iter_rows(named=True):
            dist_centroid = 0.0
            rel_diversity = 0.0

            target_emb_list = [row_tuple[col] for col in embedding_cols]
            if any(x is None for x in target_emb_list) or not row_tuple['embedding_centroid']:
                diversity_features_list.append({'user_id': row_tuple['user_id'], 'product_id': row_tuple['product_id'],
                                                'distance_from_centroid': dist_centroid, 'relative_diversity': rel_diversity})
                continue
            
            target_emb = np.array(target_emb_list)
            centroid_emb = np.array(row_tuple['embedding_centroid'])
            user_diversity_val = row_tuple['embedding_diversity'] if row_tuple['embedding_diversity'] is not None else 0.0

            if target_emb.shape == centroid_emb.shape: # Ensure embeddings are compatible
                dist_centroid = np.linalg.norm(target_emb - centroid_emb)
                if user_diversity_val > 1e-6: # Avoid division by zero or very small number
                    rel_diversity = dist_centroid / user_diversity_val
                else: # If user has no diversity (e.g., 1 item history), use raw distance
                    rel_diversity = dist_centroid 
            
            diversity_features_list.append({
                'user_id': row_tuple['user_id'],
                'product_id': row_tuple['product_id'],
                'distance_from_centroid': dist_centroid,
                'relative_diversity': rel_diversity
            })

        features_df = pl.DataFrame(diversity_features_list)
        final_df = target_df.join(features_df, on=['user_id', 'product_id'], how='left').fill_null(0)
        
        logger.info(f"Generated text diversity features for {final_df.height} rows.")
        return final_df

    # Return True or some status if FeatureFactory expects it from register functions
    return True

# Example of how to call the registration (e.g., in your lavka_recsys/__init__.py or feature_factory.py)
# if __name__ == '__main__':
#     register_text_embedding_fgens()
#     print("Text feature generators registered.")
#     # You would then need a FeatureFactory instance to use them.
