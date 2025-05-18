import numpy as np
import polars as pl
from typing import List, Dict, Any, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm # tqdm is used for progress bars
import gensim
import gensim.downloader as api # type: ignore
import fasttext # type: ignore
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import torch # Added for CUDA device check
import os # For path operations
from pathlib import Path # For robust path handling

from ..utils.custom_logging import get_logger # Assuming this is correctly importable
from ..utils.config import Config # Assuming this is correctly importable
from ..feature_factory import FeatureFactory # Assuming this is correctly importable

# Module-level logger and caches
_logger_instance = get_logger(__name__) 
_text_processor_instance: Optional['TextProcessor'] = None
_text_processor_config_cache: Optional[Config] = None
_product_embeddings_cache: Dict[str, pl.DataFrame] = {} # In-memory cache

class TextProcessor:
    """Simplified text processing with pretrained models, with GPU support for SentenceTransformers."""

    def __init__(self, config: Config):
        """Initialize text processor with configuration."""
        self.config = config
        self.logger = _logger_instance
        self.model: Union[SentenceTransformer, gensim.models.KeyedVectors, fasttext.FastText._FastText, None] = None
        self.embedding_size: int = 0
        self.model_type: str = ""
        self.device: str = "cpu"  # Device for PyTorch models (SentenceTransformers)

        model_type_config = self.config.get('feature_config.text_processing.model_type', 'sentence-transformers')

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
        """Load SentenceTransformers model and place it on GPU if available."""
        try:
            model_name = self.config.get(
                'feature_config.text_processing.model_name',
                'paraphrase-multilingual-MiniLM-L12-v2'
            )
            if torch.cuda.is_available():
                self.device = "cuda"
                self.logger.info("CUDA GPU is available. Using GPU for SentenceTransformer.")
            else:
                self.device = "cpu"
                self.logger.info("CUDA GPU not available. Using CPU for SentenceTransformer.")
            self.model = SentenceTransformer(model_name, device=self.device)
            embedding_dim = self.model.get_sentence_embedding_dimension()
            if embedding_dim is None:
                raise ValueError("SentenceTransformer model embedding dimension is None.")
            self.embedding_size = embedding_dim
            self.model_type = 'sentence-transformers'
            self.logger.info(f"Loaded sentence-transformers model: {model_name} on device: {self.device}")
        except ImportError:
            self.logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            self.logger.error(f"Error loading SentenceTransformer model: {e}")
            self.model = None

    def _load_word2vec(self):
        """Load Word2Vec model."""
        try:
            model_name = self.config.get('feature_config.text_processing.model_name', 'word2vec-ruscorpora-300')
            self.model = api.load(model_name)
            self.embedding_size = self.model.vector_size
            self.model_type = 'word2vec'
            self.logger.info(f"Loaded word2vec model: {model_name}")
        except ImportError:
            self.logger.error("gensim not available. Install with: pip install gensim")
            self.model = None
        except Exception as e:
            self.logger.error(f"Error loading Word2Vec model: {e}")
            self.model = None

    def _load_fasttext(self):
        """Load FastText model."""
        try:
            model_path = self.config.get('feature_config.text_processing.model_path', 'cc.ru.300.bin')
            self.model = fasttext.load_model(model_path)
            self.embedding_size = self.model.get_dimension()
            self.model_type = 'fasttext'
            self.logger.info(f"Loaded fasttext model: {model_path}")
        except ImportError:
            self.logger.error("fasttext not available. Install with: pip install fasttext-wheel or fasttext")
            self.model = None
        except Exception as e:
            self.logger.error(f"Error loading FastText model: {e}")
            self.model = None
            
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts."""
        if self.model is None or self.embedding_size == 0:
            self.logger.warning("No text model loaded or embedding size is zero. Returning zero embeddings.")
            return np.zeros((len(texts), 1 if self.embedding_size == 0 else self.embedding_size))
        cleaned_texts = [str(t) if t is not None and str(t).strip() else " " for t in texts]
        try:
            if self.model_type == 'sentence-transformers':
                assert isinstance(self.model, SentenceTransformer)
                embeddings = self.model.encode(
                    cleaned_texts,
                    show_progress_bar=self.config.get('feature_config.text_processing.show_encode_progress', False),
                    convert_to_numpy=True,
                    batch_size=self.config.get('feature_config.text_processing.batch_size', 32)
                )
            elif self.model_type == 'word2vec':
                assert isinstance(self.model, gensim.models.KeyedVectors)
                embeddings = np.zeros((len(cleaned_texts), self.embedding_size))
                for i, text in enumerate(cleaned_texts):
                    words = text.lower().split()
                    vectors = [self.model[word] for word in words if word in self.model.key_to_index]
                    if vectors: embeddings[i] = np.mean(vectors, axis=0)
            elif self.model_type == 'fasttext':
                assert hasattr(self.model, 'get_sentence_vector')
                embeddings = np.array([self.model.get_sentence_vector(text) for text in cleaned_texts]) # type: ignore
            else:
                self.logger.warning(f"Unknown model type during embedding: {self.model_type}")
                return np.zeros((len(texts), 1))
            return embeddings if embeddings is not None else np.zeros((len(texts), self.embedding_size if self.embedding_size > 0 else 1))
        except Exception as e:
            self.logger.error(f"Error during embedding generation with {self.model_type}: {e}")
            return np.zeros((len(texts), self.embedding_size if self.embedding_size > 0 else 1))

    def reduce_dimensions(self, embeddings: np.ndarray, dimensions: int = 20) -> np.ndarray:
        """Reduce embedding dimensions using PCA."""
        if embeddings.ndim == 1:
             if embeddings.shape[0] == 0: return embeddings
             embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] == 0 or embeddings.shape[1] <= dimensions: return embeddings
        if embeddings.shape[0] < dimensions:
            self.logger.warning(f"Samples ({embeddings.shape[0]}) < PCA dimensions ({dimensions}). Returning original.")
            return embeddings
        try:
            pca = PCA(n_components=dimensions, random_state=self.config.get('general.seed', 42))
            reduced = pca.fit_transform(embeddings)
            self.logger.info(f"Reduced embeddings from {embeddings.shape[1]} to {dimensions} via PCA.")
            return reduced
        except ImportError:
            self.logger.warning("scikit-learn not available for PCA. Using original embeddings.")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error during PCA: {e}. Using original embeddings.")
            return embeddings

def get_text_processor(config: Config) -> TextProcessor:
    """Singleton accessor for TextProcessor."""
    global _text_processor_instance, _text_processor_config_cache
    current_model_config_sig = (
        config.get('feature_config.text_processing.model_type'),
        config.get('feature_config.text_processing.model_name', config.get('feature_config.text_processing.model_path'))
    )
    cached_model_config_sig = None
    if _text_processor_config_cache:
        cached_model_config_sig = (
            _text_processor_config_cache.get('feature_config.text_processing.model_type'),
            _text_processor_config_cache.get('feature_config.text_processing.model_name', _text_processor_config_cache.get('feature_config.text_processing.model_path'))
        )

    if _text_processor_instance is None or current_model_config_sig != cached_model_config_sig:
        _logger_instance.info(f"Initializing TextProcessor with model: {current_model_config_sig[0]} - {current_model_config_sig[1]}")
        _text_processor_instance = TextProcessor(config)
        _text_processor_config_cache = config.copy()
    return _text_processor_instance

def _get_product_embeddings_df(
    history_df: pl.DataFrame,
    target_df: pl.DataFrame,
    text_processor: TextProcessor,
    config: Config,
    id_col: str,
    text_col: str,
    embed_col_prefix: Optional[str] = None
) -> pl.DataFrame:
    """
    Helper to get or generate product embeddings, with in-memory and disk caching.
    Embeddings are saved as Parquet files.
    """
    model_name_or_path = text_processor.config.get(
        'feature_config.text_processing.model_name', 
        text_processor.config.get('feature_config.text_processing.model_path')
    )
    sanitized_model_name = model_name_or_path.replace('/', '_') if model_name_or_path else "unknown_model"
    pca_enabled = config.get('feature_config.text_processing.enable_pca', True)
    final_embedding_dim = config.get('feature_config.text_processing.embedding_dimensions_pca', 20) if pca_enabled else config.get('feature_config.text_processing.embedding_dimensions', 20)


    cache_key_parts = [
        id_col,
        text_processor.model_type,
        sanitized_model_name,
        str(final_embedding_dim), 
        str(pca_enabled) 
    ]
    cache_key = "_".join(filter(None, cache_key_parts))
    
    if cache_key in _product_embeddings_cache:
        _logger_instance.info(f"Using in-memory cached embeddings for {id_col} with key {cache_key}")
        return _product_embeddings_cache[cache_key]

    disk_cache_path_str = config.get('feature_config.text_processing.disk_cache_path', './.cache/embeddings')
    disk_cache_path = Path(disk_cache_path_str)
    disk_cache_path.mkdir(parents=True, exist_ok=True)
    cache_filename = f"{cache_key}.parquet"
    cache_file_path = disk_cache_path / cache_filename

    if cache_file_path.exists():
        try:
            _logger_instance.info(f"Loading embeddings from disk: {cache_file_path}")
            embed_df = pl.read_parquet(cache_file_path)
            _product_embeddings_cache[cache_key] = embed_df
            _logger_instance.info(f"Successfully loaded embeddings for {id_col} from disk. Shape: {embed_df.shape}")
            return embed_df
        except Exception as e:
            _logger_instance.warning(f"Failed to load embeddings from {cache_file_path}: {e}. Regenerating.")

    _logger_instance.info(f"Generating embeddings for unique {id_col} from {text_col} (key: {cache_key})")
    select_cols = list(set([id_col, text_col]))
    unique_items_hist = history_df.select(select_cols).unique(subset=[id_col], keep='first')
    unique_items_target = target_df.select(select_cols).unique(subset=[id_col], keep='first')
    all_unique_items = pl.concat([unique_items_hist, unique_items_target]).unique(subset=[id_col], keep='first')
    
    item_texts = all_unique_items.get_column(text_col).to_list()
    item_ids = all_unique_items.get_column(id_col)

    if not item_texts:
        _logger_instance.warning(f"No unique texts found for {id_col}. Returning empty embeddings DataFrame.")
        dim_to_use = final_embedding_dim
        if text_processor.embedding_size > 0 and text_processor.embedding_size < dim_to_use and not pca_enabled: 
            dim_to_use = text_processor.embedding_size
        elif text_processor.embedding_size == 0: dim_to_use = 1
        
        prefix = embed_col_prefix or id_col.replace('_id','').replace('_category','cat')
        feature_names = [f"{prefix}_embed_{i}" for i in range(dim_to_use)]
        schema_dict: Dict[str, Any] = {id_col: item_ids.dtype}
        for name in feature_names: schema_dict[name] = pl.Float32
        return pl.DataFrame(schema=schema_dict)

    embeddings_raw = text_processor.get_embeddings(item_texts)
    embeddings_processed = embeddings_raw

    if embeddings_raw.shape[1] > 1:
        if pca_enabled:
            dimensions_pca = config.get('feature_config.text_processing.embedding_dimensions_pca', 20)
            if embeddings_raw.shape[1] > dimensions_pca:
                 _logger_instance.info(f"Applying PCA to reduce {id_col} embeddings from {embeddings_raw.shape[1]} to {dimensions_pca} dimensions.")
                 embeddings_processed = text_processor.reduce_dimensions(embeddings_raw, dimensions_pca)
            else:
                _logger_instance.info(f"PCA not applied for {id_col}: original dim ({embeddings_raw.shape[1]}) <= target PCA dim ({dimensions_pca}).")
        else:
            _logger_instance.info(f"PCA disabled for {id_col} embeddings.")
    
    num_embedding_dims = embeddings_processed.shape[1]
    prefix = embed_col_prefix or id_col.replace('_id','').replace('_category','cat')
    feature_names = [f"{prefix}_embed_{i}" for i in range(num_embedding_dims)]
    
    data_dict: Dict[str, Any] = {id_col: item_ids}
    for i, name in enumerate(feature_names):
        data_dict[name] = embeddings_processed[:, i]
    embed_df = pl.DataFrame(data_dict)
    
    try:
        embed_df.write_parquet(cache_file_path)
        _logger_instance.info(f"Saved embeddings for {id_col} to disk: {cache_file_path}. Shape: {embed_df.shape}")
    except Exception as e:
        _logger_instance.error(f"Failed to save embeddings to {cache_file_path}: {e}")
    _product_embeddings_cache[cache_key] = embed_df
    _logger_instance.info(f"Generated and cached embeddings for {id_col}. Final shape: {embed_df.shape}")
    return embed_df

def _calculate_single_cosine_similarity_udf(target_emb_list: Optional[list], user_emb_list: Optional[list]) -> float:
    if target_emb_list is None or user_emb_list is None: return 0.0
    target_emb = np.array(target_emb_list, dtype=np.float32)
    user_emb = np.array(user_emb_list, dtype=np.float32)
    if target_emb.size == 0 or user_emb.size == 0 or \
       target_emb.shape != user_emb.shape or \
       np.linalg.norm(target_emb) < 1e-9 or np.linalg.norm(user_emb) < 1e-9:
        return 0.0
    return sklearn_cosine_similarity(target_emb.reshape(1, -1), user_emb.reshape(1, -1))[0, 0]

def _calculate_max_historical_similarity_udf(target_emb_list: Optional[list], historical_embeddings_obj: Optional[list]) -> float:
    if target_emb_list is None or historical_embeddings_obj is None: return 0.0
    target_emb = np.array(target_emb_list, dtype=np.float32)
    if target_emb.size == 0 or np.linalg.norm(target_emb) < 1e-9: return 0.0
    valid_hist_embeds = []
    if isinstance(historical_embeddings_obj, list):
        for emb_data in historical_embeddings_obj:
            if emb_data is not None and isinstance(emb_data, list):
                emb = np.array(emb_data, dtype=np.float32)
                if emb.ndim > 0 and emb.size > 0 and emb.shape == target_emb.shape and np.linalg.norm(emb) > 1e-9:
                    valid_hist_embeds.append(emb)
    if not valid_hist_embeds: return 0.0
    historical_embeds_matrix = np.array(valid_hist_embeds)
    similarities = sklearn_cosine_similarity(target_emb.reshape(1, -1), historical_embeds_matrix)
    return np.max(similarities) if similarities.size > 0 else 0.0

def _batch_process_single_similarity(batch_struct_series: pl.Series, target_emb_field: str, user_emb_field: str, desc: str) -> pl.Series:
    results = [
        _calculate_single_cosine_similarity_udf(s[target_emb_field], s[user_emb_field])
        for s in tqdm(batch_struct_series, desc=desc, leave=False, mininterval=1.0, miniters=max(1, len(batch_struct_series)//100))
    ]
    return pl.Series(results, dtype=pl.Float64)

def _batch_process_max_historical_similarity(batch_struct_series: pl.Series, target_emb_field: str, hist_emb_field: str, desc: str) -> pl.Series:
    results = [
        _calculate_max_historical_similarity_udf(s[target_emb_field], s[hist_emb_field])
        for s in tqdm(batch_struct_series, desc=desc, leave=False, mininterval=1.0, miniters=max(1, len(batch_struct_series)//100))
    ]
    return pl.Series(results, dtype=pl.Float64)

def register_text_embedding_fgens() -> bool:
    PRODUCT_EMBED_DIMS_HINT = 20 
    CATEGORY_EMBED_DIMS_HINT = 10

    @FeatureFactory.register(
        'product_embeddings',
        num_cols=[f'product_embed_{i}' for i in range(PRODUCT_EMBED_DIMS_HINT)]
    )
    def generate_product_embeddings(history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config) -> pl.DataFrame:
        text_processor = get_text_processor(config)
        product_embed_df = _get_product_embeddings_df(
            history_df, target_df, text_processor, config, 
            'product_id', 'product_name', 'product'
        )
        return target_df.join(product_embed_df, on='product_id', how='left').fill_null(0.0) # Fill with float

    @FeatureFactory.register(
        'category_embeddings',
        num_cols=[f'cat_embed_{i}' for i in range(CATEGORY_EMBED_DIMS_HINT)]
    )
    def generate_category_embeddings(history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config) -> pl.DataFrame:
        text_processor = get_text_processor(config)
        h_df = history_df.with_columns(pl.col("product_category").fill_null("UNKNOWN_CATEGORY").cast(pl.String))
        t_df = target_df.with_columns(pl.col("product_category").fill_null("UNKNOWN_CATEGORY").cast(pl.String))
        category_embed_df = _get_product_embeddings_df(
            h_df, t_df, text_processor, config, 
            'product_category', 'product_category', 'cat'
        )
        return target_df.join(category_embed_df, on='product_category', how='left').fill_null(0.0) # Fill with float

    @FeatureFactory.register(
        'user_product_similarity', 
        num_cols=[
            'purchase_weighted_similarity', 'cart_weighted_similarity',
            'max_purchase_similarity_history', 'max_cart_similarity_history',     
        ]
    )
    def generate_user_product_similarity(history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config) -> pl.DataFrame:
        text_processor = get_text_processor(config)
        logger = _logger_instance
        logger.info("Starting user_product_similarity generation.")

        h_df_p = history_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT").cast(pl.String))
        t_df_p = target_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT").cast(pl.String))

        all_product_embeddings_df = _get_product_embeddings_df(
            h_df_p, t_df_p, text_processor, config, 'product_id', 'product_name', 'product'
        )
        
        default_feature_cols = {
            'purchase_weighted_similarity': 0.0, 'cart_weighted_similarity': 0.0,
            'max_purchase_similarity_history': 0.0, 'max_cart_similarity_history': 0.0
        }
        if all_product_embeddings_df.height == 0 or all_product_embeddings_df.width <=1 :
            logger.warning("No product embeddings for user_product_similarity. Returning defaults.")
            return target_df.with_columns([pl.lit(v).cast(pl.Float64).alias(k) for k,v in default_feature_cols.items()])

        embedding_cols = [col for col in all_product_embeddings_df.columns if col.startswith('product_embed_')]
        if not embedding_cols:
            logger.warning("Embedding columns not found. Returning defaults.")
            return target_df.with_columns([pl.lit(v).cast(pl.Float64).alias(k) for k,v in default_feature_cols.items()])

        product_embedding_dict = {
            row[0]: np.array(row[1:]) 
            for row in all_product_embeddings_df.select(['product_id'] + embedding_cols).iter_rows()
        }
        logger.info(f"Created embedding lookup for {len(product_embedding_dict)} products.")

        def get_user_history_embeddings(action_type: str, weight_col_name: str, hist_embeddings_col_name: str):
            user_hist_actions = history_df.filter(pl.col('action_type') == action_type)
            empty_weighted_schema = {'user_id': pl.Utf8, weight_col_name: pl.List(pl.Float64)}
            empty_historical_schema = {'user_id': pl.Utf8, hist_embeddings_col_name: pl.Object}
            if user_hist_actions.height == 0:
                return pl.DataFrame([], schema=empty_weighted_schema), pl.DataFrame([], schema=empty_historical_schema)

            user_hist = user_hist_actions.group_by(['user_id', 'product_id']).agg(pl.count().alias('interaction_count'))
            user_weighted_embeddings_list = []
            user_all_historical_embeddings_list = []

            for user_id_key, group_df in tqdm(user_hist.group_by('user_id', maintain_order=False), desc=f"Processing {action_type} history", leave=False):
                scalar_user_id_val: Any
                if isinstance(user_id_key, tuple): scalar_user_id_val = user_id_key[0]
                elif isinstance(user_id_key, list): scalar_user_id_val = user_id_key[0]
                else: scalar_user_id_val = user_id_key
                if not isinstance(scalar_user_id_val, (int, str, np.integer)): continue
                user_id_as_string = str(scalar_user_id_val)
                embeddings_np_list = [product_embedding_dict.get(pid) for pid in group_df['product_id']]
                valid_embeddings_info = [(emb, group_df['interaction_count'][i]) for i, emb in enumerate(embeddings_np_list) if emb is not None and emb.ndim > 0 and emb.size > 0]
                if not valid_embeddings_info: continue
                valid_embeddings = [info[0] for info in valid_embeddings_info]
                weights = np.array([info[1] for info in valid_embeddings_info], dtype=float)
                weighted_avg_emb = np.average(np.array(valid_embeddings), axis=0, weights=weights)
                user_weighted_embeddings_list.append({'user_id': user_id_as_string, weight_col_name: weighted_avg_emb.tolist()})
                user_all_historical_embeddings_list.append({'user_id': user_id_as_string, hist_embeddings_col_name: [e.tolist() for e in valid_embeddings]})
            
            weighted_df = pl.DataFrame(user_weighted_embeddings_list, schema=empty_weighted_schema) if user_weighted_embeddings_list else pl.DataFrame([], schema=empty_weighted_schema)
            historical_df = pl.DataFrame(user_all_historical_embeddings_list, schema=empty_historical_schema) if user_all_historical_embeddings_list else pl.DataFrame([], schema=empty_historical_schema)
            return weighted_df, historical_df

        purchase_weighted_emb_df, purchase_hist_emb_df = get_user_history_embeddings("AT_Purchase", "purchase_weighted_embedding", "purchase_historical_embeddings")
        cart_weighted_emb_df, cart_hist_emb_df = get_user_history_embeddings("AT_CartUpdate", "cart_weighted_embedding", "cart_historical_embeddings")
        logger.info(f"Processed purchase history for {purchase_weighted_emb_df.height} users.")
        logger.info(f"Processed cart history for {cart_weighted_emb_df.height} users.")

        target_with_embeddings_eager = target_df.join(all_product_embeddings_df.select(['product_id'] + embedding_cols), on='product_id', how='left')
        target_with_embeddings_eager = target_with_embeddings_eager.with_columns(pl.concat_list([pl.col(c) for c in embedding_cols]).alias("target_embedding_list"))
        
        original_user_id_dtype = target_df['user_id'].dtype 
        target_with_embeddings_eager = target_with_embeddings_eager.with_columns(pl.col('user_id').cast(pl.Utf8))

        target_with_embeddings_eager = target_with_embeddings_eager.join(purchase_weighted_emb_df, on='user_id', how='left')
        target_with_embeddings_eager = target_with_embeddings_eager.join(purchase_hist_emb_df, on='user_id', how='left')
        target_with_embeddings_eager = target_with_embeddings_eager.join(cart_weighted_emb_df, on='user_id', how='left')
        target_with_embeddings_eager = target_with_embeddings_eager.join(cart_hist_emb_df, on='user_id', how='left')
        
        logger.info("Applying optimized similarity calculations with progress bars (Lazy Evaluation)...")
        lf = target_with_embeddings_eager.lazy() 
        
        new_feature_names = list(default_feature_cols.keys())

        logger.info("Step 1/4: Defining purchase_weighted_similarity...")
        lf = lf.with_columns(
            pl.struct(["target_embedding_list", "purchase_weighted_embedding"])
            .map_batches(lambda batch_s: _batch_process_single_similarity(batch_s, "target_embedding_list", "purchase_weighted_embedding", "PurchWeightedSim"))
            .alias(new_feature_names[0]) # purchase_weighted_similarity
        )
        logger.info("Step 2/4: Defining max_purchase_similarity_history...")
        lf = lf.with_columns(
            pl.struct(["target_embedding_list", "purchase_historical_embeddings"])
            .map_batches(lambda batch_s: _batch_process_max_historical_similarity(batch_s, "target_embedding_list", "purchase_historical_embeddings", "MaxPurchHistSim"))
            .alias(new_feature_names[2]) # max_purchase_similarity_history
        )
        logger.info("Step 3/4: Defining cart_weighted_similarity...")
        lf = lf.with_columns(
            pl.struct(["target_embedding_list", "cart_weighted_embedding"])
            .map_batches(lambda batch_s: _batch_process_single_similarity(batch_s, "target_embedding_list", "cart_weighted_embedding", "CartWeightedSim"))
            .alias(new_feature_names[1]) # cart_weighted_similarity
        )
        logger.info("Step 4/4: Defining max_cart_similarity_history...")
        lf = lf.with_columns(
            pl.struct(["target_embedding_list", "cart_historical_embeddings"])
            .map_batches(lambda batch_s: _batch_process_max_historical_similarity(batch_s, "target_embedding_list", "cart_historical_embeddings", "MaxCartHistSim"))
            .alias(new_feature_names[3]) # max_cart_similarity_history
        )
        
        # Fill nulls for new feature columns within the lazy plan
        for col_name in new_feature_names:
            lf = lf.with_columns(pl.col(col_name).fill_null(0.0).cast(pl.Float64))

        # Define the final set of columns to select for collection
        select_expressions = []
        for original_col_name in target_df.columns:
            if original_col_name == 'user_id':
                select_expressions.append(pl.col('user_id').cast(original_user_id_dtype).alias(original_col_name))
            else:
                select_expressions.append(pl.col(original_col_name))
        for new_feature_col_name in new_feature_names:
            select_expressions.append(pl.col(new_feature_col_name))
        
        lf = lf.select(select_expressions)
        
        logger.info("Collecting final selected results from lazy evaluation...")
        final_df = lf.collect()
        logger.info("Finished collecting final results.")
            
        logger.info(f"Finished user_product_similarity generation. Output shape: {final_df.shape}")
        return final_df

    @FeatureFactory.register(
        'text_similarity_cluster',
        cat_cols=['product_text_cluster'], 
        num_cols=['cluster_purchase_ratio', 'cluster_cart_ratio', 'cluster_view_ratio']
    )
    def generate_text_similarity_cluster(history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config) -> pl.DataFrame:
        text_processor = get_text_processor(config)
        logger = _logger_instance
        logger.info("Starting text_similarity_cluster generation.")

        h_df_p = history_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT").cast(pl.String))
        t_df_p = target_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT").cast(pl.String))

        all_product_embeddings_df = _get_product_embeddings_df(
            h_df_p, t_df_p, text_processor, config, 'product_id', 'product_name', 'product'
        )
        
        default_cluster_cols = {'product_text_cluster': -1, 'cluster_purchase_ratio': 0.0, 'cluster_cart_ratio': 0.0, 'cluster_view_ratio': 0.0}
        if all_product_embeddings_df.height == 0 or all_product_embeddings_df.width <=1:
            logger.warning("No product embeddings for text_similarity_cluster.")
            return target_df.with_columns([pl.lit(v).cast(pl.Int32 if k == 'product_text_cluster' else pl.Float64).alias(k) for k,v in default_cluster_cols.items()])

        embedding_cols = [col for col in all_product_embeddings_df.columns if col.startswith('product_embed_')]
        if not embedding_cols:
             logger.warning("Embedding columns not found for text_similarity_cluster.")
             return target_df.with_columns([pl.lit(v).cast(pl.Int32 if k == 'product_text_cluster' else pl.Float64).alias(k) for k,v in default_cluster_cols.items()])

        product_embeddings_np = all_product_embeddings_df.select(embedding_cols).to_numpy()
        product_ids_series = all_product_embeddings_df['product_id']

        if product_embeddings_np.shape[0] == 0: 
            logger.warning("No product embeddings numpy array for text_similarity_cluster.")
            return target_df.with_columns([pl.lit(v).cast(pl.Int32 if k == 'product_text_cluster' else pl.Float64).alias(k) for k,v in default_cluster_cols.items()])
        try:
            n_clusters = config.get('feature_config.text_processing.n_clusters', 15)
            if product_embeddings_np.shape[0] < n_clusters:
                logger.warning(f"Products ({product_embeddings_np.shape[0]}) < n_clusters ({n_clusters}). Adjusting.")
                n_clusters = max(1, product_embeddings_np.shape[0]) 
            kmeans = KMeans(n_clusters=n_clusters, random_state=config.get('general.seed', 42), n_init='auto')
            cluster_labels = kmeans.fit_predict(product_embeddings_np)
            product_cluster_df = pl.DataFrame({'product_id': product_ids_series, 'product_text_cluster': cluster_labels.astype(np.int32)})
            logger.info(f"Clustered {len(product_ids_series)} products into {n_clusters} clusters.")
            history_with_clusters = history_df.join(product_cluster_df, on='product_id', how='left').with_columns(pl.col('product_text_cluster').fill_null(-1).cast(pl.Int32))

            def calculate_cluster_ratios(action_type: str, ratio_col_name: str) -> pl.DataFrame:
                user_cluster_action = history_with_clusters.filter(pl.col('action_type') == action_type).group_by(['user_id', 'product_text_cluster']).agg(pl.count().alias('action_in_cluster_count'))
                user_total_actions = user_cluster_action.group_by('user_id').agg(pl.sum('action_in_cluster_count').alias('total_user_actions'))
                return user_cluster_action.join(user_total_actions, on='user_id', how='left').with_columns((pl.col('action_in_cluster_count') / pl.col('total_user_actions')).fill_null(0.0).alias(ratio_col_name)).select(['user_id', 'product_text_cluster', ratio_col_name])

            purchase_ratios_df = calculate_cluster_ratios("AT_Purchase", "cluster_purchase_ratio")
            cart_ratios_df = calculate_cluster_ratios("AT_CartUpdate", "cluster_cart_ratio")
            view_ratios_df = calculate_cluster_ratios("AT_View", "cluster_view_ratio") 

            result_df = target_df.join(product_cluster_df, on='product_id', how='left').with_columns(pl.col('product_text_cluster').fill_null(-1).cast(pl.Int32))
            result_df = result_df.join(purchase_ratios_df, on=['user_id', 'product_text_cluster'], how='left')
            result_df = result_df.join(cart_ratios_df, on=['user_id', 'product_text_cluster'], how='left')
            result_df = result_df.join(view_ratios_df, on=['user_id', 'product_text_cluster'], how='left')
            
            # Use a loop for fill_null and cast for default_cluster_cols
            final_df = result_df
            for k, v_default in default_cluster_cols.items():
                final_df = final_df.with_columns(
                    pl.col(k).fill_null(v_default).cast(pl.Int32 if k == 'product_text_cluster' else pl.Float64)
                )

            logger.info(f"Generated text similarity cluster features for {final_df.height} rows.")
            return final_df.select(target_df.columns + list(default_cluster_cols.keys()))
        except ImportError: 
            logger.warning("scikit-learn not available for KMeans. Skipping text similarity clustering.")
        except Exception as e: 
            logger.error(f"Error during text_similarity_cluster: {e}")
        return target_df.with_columns([pl.lit(v).cast(pl.Int32 if k == 'product_text_cluster' else pl.Float64).alias(k) for k,v in default_cluster_cols.items()])


    @FeatureFactory.register(
        'text_diversity_features',
        num_cols=['distance_from_centroid', 'relative_diversity']
    )
    def generate_text_diversity_features(history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config) -> pl.DataFrame:
        text_processor = get_text_processor(config)
        logger = _logger_instance
        logger.info("Starting text_diversity_features generation.")

        h_df_p = history_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT").cast(pl.String))
        t_df_p = target_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT").cast(pl.String))

        all_product_embeddings_df = _get_product_embeddings_df(
            h_df_p, t_df_p, text_processor, config, 'product_id', 'product_name', 'product'
        )
        
        default_diversity_cols = {'distance_from_centroid': 0.0, 'relative_diversity': 0.0}
        if all_product_embeddings_df.height == 0 or all_product_embeddings_df.width <=1:
            logger.warning("No product embeddings for text_diversity_features.")
            return target_df.with_columns([pl.lit(v).cast(pl.Float64).alias(k) for k,v in default_diversity_cols.items()])

        embedding_cols = [col for col in all_product_embeddings_df.columns if col.startswith('product_embed_')]
        if not embedding_cols:
             logger.warning("Embedding columns not found for text_diversity_features.")
             return target_df.with_columns([pl.lit(v).cast(pl.Float64).alias(k) for k,v in default_diversity_cols.items()])

        user_history_actions = history_df.filter(pl.col('action_type').is_in(["AT_Purchase", "AT_View"]))
        user_history_with_embeds = user_history_actions.join(all_product_embeddings_df.select(['product_id'] + embedding_cols), on='product_id', how='inner')
        
        user_stats_list = []
        for user_id_key, group_df in tqdm(user_history_with_embeds.group_by('user_id'), desc="Calculating user embedding stats", leave=False):
            scalar_user_id_val: Any
            if isinstance(user_id_key, tuple): scalar_user_id_val = user_id_key[0]
            elif isinstance(user_id_key, list): scalar_user_id_val = user_id_key[0]
            else: scalar_user_id_val = user_id_key
            if not isinstance(scalar_user_id_val, (int, str, np.integer)): continue
            user_id_as_string = str(scalar_user_id_val)
            user_embeds_np = group_df.select(embedding_cols).to_numpy()
            centroid_val_list, diversity_val = None, 0.0
            if user_embeds_np.size > 0 and user_embeds_np.shape[1] > 0:
                centroid_np_arr = np.mean(user_embeds_np, axis=0)
                centroid_val_list = centroid_np_arr.tolist()
                if user_embeds_np.shape[0] > 1:
                     distances = np.linalg.norm(user_embeds_np - centroid_np_arr, axis=1)
                     diversity_val = np.mean(distances)
            user_stats_list.append({'user_id': user_id_as_string, 'embedding_centroid_list': centroid_val_list, 'embedding_diversity': diversity_val})
        
        user_stats_schema = {'user_id':pl.Utf8, 'embedding_centroid_list':pl.List(pl.Float64), 'embedding_diversity':pl.Float64}
        user_stats_df = pl.DataFrame(user_stats_list, schema=user_stats_schema) if user_stats_list else pl.DataFrame([], schema=user_stats_schema)
        logger.info(f"Calculated embedding statistics for {user_stats_df.height} users.")
        
        target_for_join = target_df.with_columns(pl.col('user_id').cast(pl.Utf8).alias('user_id_str_for_join'))
        target_with_user_stats = target_for_join.join(all_product_embeddings_df.select(['product_id'] + embedding_cols), on='product_id', how='left')
        target_with_user_stats = target_with_user_stats.join(user_stats_df, left_on='user_id_str_for_join', right_on='user_id', how='left', suffix='_stats')
        
        diversity_features_list = []
        original_target_user_id_dtype, original_target_product_id_dtype = target_df['user_id'].dtype, target_df['product_id'].dtype

        for row_d in tqdm(target_with_user_stats.iter_rows(named=True), total=target_with_user_stats.height, desc="Calculating diversity features", leave=False):
            dist_centroid, rel_diversity = 0.0, 0.0
            output_user_id, output_product_id = row_d['user_id'], row_d['product_id'] # These are original types from target_df
            target_emb_list = [row_d.get(col) for col in embedding_cols]
            centroid_list = row_d.get('embedding_centroid_list') 
            if any(x is None for x in target_emb_list) or centroid_list is None:
                diversity_features_list.append({'user_id': output_user_id, 'product_id': output_product_id, **default_diversity_cols})
                continue
            target_emb_np, centroid_np = np.array(target_emb_list), np.array(centroid_list)
            user_diversity_val = row_d.get('embedding_diversity', 0.0) 
            if user_diversity_val is None: user_diversity_val = 0.0
            if target_emb_np.shape == centroid_np.shape and target_emb_np.size > 0 and centroid_np.size > 0 : 
                dist_centroid = np.linalg.norm(target_emb_np - centroid_np)
                if user_diversity_val > 1e-9: rel_diversity = dist_centroid / user_diversity_val
                else: rel_diversity = dist_centroid if dist_centroid > 1e-9 else 0.0 
            diversity_features_list.append({'user_id': output_user_id, 'product_id': output_product_id, 'distance_from_centroid': dist_centroid, 'relative_diversity': rel_diversity})

        div_features_schema = {'user_id': original_target_user_id_dtype, 'product_id': original_target_product_id_dtype, **{k: pl.Float64 for k in default_diversity_cols}}
        features_df = pl.DataFrame(diversity_features_list, schema=div_features_schema) if diversity_features_list else pl.DataFrame([], schema=div_features_schema)
        features_df = features_df.unique(subset=['user_id', 'product_id'], keep='last')
        
        final_df = target_df.join(features_df, on=['user_id', 'product_id'], how='left')
        for col_name in default_diversity_cols:
            final_df = final_df.with_columns(pl.col(col_name).fill_null(0.0).cast(pl.Float64))
        
        logger.info(f"Generated text diversity features for {final_df.height} rows.")
        return final_df.select(target_df.columns + list(default_diversity_cols.keys()))

    return True
