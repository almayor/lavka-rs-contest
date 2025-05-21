import numpy as np
import polars as pl
from typing import List, Dict, Any, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm, trange
import gensim
import gensim.downloader as api # type: ignore
import fasttext # type: ignore
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from ..utils.custom_logging import get_logger
from ..utils.config import Config
from ..feature_factory import FeatureFactory

_text_processor_instance: Optional['TextProcessor'] = None # Forward declaration
_text_processor_config_cache: Optional[Config] = None
_logger_instance = get_logger(__name__) 
_product_embeddings_cache: Dict[str, pl.DataFrame] = {}


class TextProcessor:
    """Simplified text processing with pretrained models"""

    def __init__(self, config: Config):
        """Initialize text processor with configuration"""
        self.config = config
        self.logger = _logger_instance 
        self.model: Union[SentenceTransformer, gensim.models.KeyedVectors, fasttext.FastText._FastText, None] = None
        self.embedding_size: int = 0
        self.model_type: str = ""

        # Using the config path structure from the user's canvas version
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
        """Load SentenceTransformers model"""
        try:
            from sentence_transformers import SentenceTransformer 
            model_name = self.config.get(
                'feature_config.text_processing.model_name',
                'paraphrase-multilingual-MiniLM-L12-v2'
            )
            self.model = SentenceTransformer(model_name)
            embedding_dim = self.model.get_sentence_embedding_dimension()
            if embedding_dim is None:
                raise ValueError("SentenceTransformer model embedding dimension is None.")
            self.embedding_size = embedding_dim
            self.model_type = 'sentence-transformers'
            self.logger.info(f"Loaded sentence-transformers model: {model_name}")
        except ImportError as e:
            self.logger.error(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
            raise e


    def _load_word2vec(self):
        """Load Word2Vec model"""
        try:
            import gensim.downloader as api 
            model_name = self.config.get(
                'feature_config.text_processing.model_name',
                'word2vec-ruscorpora-300'
            )
            self.model = api.load(model_name)
            self.embedding_size = self.model.vector_size
            self.model_type = 'word2vec'
            self.logger.info(f"Loaded word2vec model: {model_name}")
        except ImportError as e:
            self.logger.error(
                "gensim not available. "
                "Install with: pip install gensim"
            )
            raise e

    def _load_fasttext(self):
        """Load FastText model"""
        try:
            import fasttext 
            model_path = self.config.get(
                'feature_config.text_processing.model_path',
                'cc.ru.300.bin'
            )
            self.model = fasttext.load_model(model_path)
            self.embedding_size = self.model.get_dimension()
            self.model_type = 'fasttext'
            self.logger.info(f"Loaded fasttext model: {model_path}")
        except ImportError as e:
            self.logger.error(
                "fasttext not available. "
                "Install with: pip install fasttext-wheel or fasttext"
            )
            raise e
            
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts using the pretrained model.
        Returns zero embeddings of expected size if model fails.
        """
        if self.model is None or self.embedding_size == 0:
            self.logger.warning("No text model loaded or embedding size is zero. Returning zero embeddings.")
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
                    vectors = [self.model[word] for word in words if word in self.model.key_to_index] 
                    if vectors:
                        embeddings[i] = np.mean(vectors, axis=0)
            elif self.model_type == 'fasttext':
                assert hasattr(self.model, 'get_sentence_vector')
                embeddings = np.array([
                    self.model.get_sentence_vector(text) # type: ignore
                    for text in cleaned_texts
                ])
            else:
                self.logger.warning(f"Unknown model type during embedding: {self.model_type}")
                return np.zeros((len(texts), 1))
            
            return embeddings if embeddings is not None else np.zeros((len(texts), self.embedding_size if self.embedding_size > 0 else 1))

        except Exception as e:
            self.logger.error(f"Error during embedding generation with {self.model_type}: {e}")
            raise e

    def reduce_dimensions(self, embeddings: np.ndarray, dimensions: int = 20) -> np.ndarray:
        """Reduce embedding dimensions using PCA."""
        if embeddings.ndim == 1: 
             if embeddings.shape[0] == 0:
                self.logger.warning("Attempted to reduce zero-length 1D embeddings. Returning as is.")
                return embeddings
             embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] == 0: 
            self.logger.warning("Attempted to reduce zero-dimensional embeddings (shape[1] is 0). Returning as is.")
            return embeddings
        if embeddings.shape[1] <= dimensions:
            return embeddings
        if embeddings.shape[0] < dimensions: 
            self.logger.warning(
                f"Not enough samples ({embeddings.shape[0]}) to reduce to {dimensions} dimensions using PCA. "
                f"Returning original embeddings with {embeddings.shape[1]} dimensions."
            )
            return embeddings

        try:
            from sklearn.decomposition import PCA 
            pca = PCA(n_components=dimensions, random_state=42)
            reduced = pca.fit_transform(embeddings)
            self.logger.info(
                f"Reduced embeddings from {embeddings.shape[1]} to {dimensions} dimensions using PCA."
            )
            return reduced
        except ImportError as e:
            self.logger.warning(
                "scikit-learn not available for dimension reduction. "
                "Install with: pip install scikit-learn. Using original embeddings."
            )
            raise e

def get_text_processor(config: Config) -> TextProcessor:
    global _text_processor_instance, _text_processor_config_cache
    current_model_type = config.get('feature_config.text_processing.model_type')
    current_model_name_or_path = config.get('feature_config.text_processing.model_name', config.get('feature_config.text_processing.model_path'))

    if _text_processor_instance is None or \
       _text_processor_config_cache is None or \
       _text_processor_config_cache.get('feature_config.text_processing.model_type') != current_model_type or \
       _text_processor_config_cache.get('feature_config.text_processing.model_name', _text_processor_config_cache.get('feature_config.text_processing.model_path')) != current_model_name_or_path:
        
        _logger_instance.info(f"Initializing TextProcessor with model: {current_model_type} - {current_model_name_or_path}")
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
    cache_key = f"{id_col}_{text_processor.model_type}_{text_processor.config.get('feature_config.text_processing.model_name', text_processor.config.get('feature_config.text_processing.model_path'))}_{config.get('feature_config.text_processing.embedding_dimensions')}"
    
    if cache_key in _product_embeddings_cache:
        _logger_instance.info(f"Using cached embeddings for {id_col} with key {cache_key}")
        return _product_embeddings_cache[cache_key]

    _logger_instance.info(f"Generating embeddings for unique {id_col} from {text_col}")

    # Using set for select columns to avoid issues if id_col and text_col are the same (e.g. for category)
    select_cols = list(set([id_col, text_col]))
    unique_items_hist = history_df.select(select_cols).unique(subset=[id_col])
    unique_items_target = target_df.select(select_cols).unique(subset=[id_col])
    
    all_unique_items = pl.concat([unique_items_hist, unique_items_target]).unique(subset=[id_col], keep='first')
    
    item_texts = all_unique_items[text_col].to_list()
    item_ids = all_unique_items[id_col] 

    if not item_texts:
        _logger_instance.warning(f"No unique texts found for {id_col}. Returning empty embeddings DataFrame.")
        dim = config.get('feature_config.text_processing.embedding_dimensions', 20) 
        if text_processor.embedding_size > 0 and text_processor.embedding_size < dim:
            dim = text_processor.embedding_size
        elif text_processor.embedding_size == 0: 
            dim = 1 
        
        embed_col_prefix = embed_col_prefix or id_col.replace('_id','').replace('_category','cat')
        feature_names = [f"{embed_col_prefix}_embed_{i}" for i in range(dim)]
        
        schema_dict: Dict[str, Any] = {id_col: item_ids.dtype} 
        for name in feature_names:
            schema_dict[name] = pl.Float32
        return pl.DataFrame(schema=schema_dict)


    embeddings = text_processor.get_embeddings(item_texts)

    if embeddings.shape[1] > 1: 
        dimensions = config.get('feature_config.text_processing.embedding_dimensions', 20) 
        if embeddings.shape[1] > dimensions: 
            embeddings = text_processor.reduce_dimensions(embeddings, dimensions)
    
    num_embedding_dims = embeddings.shape[1]
    embed_col_prefix = embed_col_prefix or id_col.replace('_id','').replace('_category','cat')
    feature_names = [f"{embed_col_prefix}_embed_{i}" for i in range(num_embedding_dims)]
    
    data_dict: Dict[str, Any] = {id_col: item_ids} 
    for i, name in enumerate(feature_names):
        data_dict[name] = embeddings[:, i] 
    
    embed_df = pl.DataFrame(data_dict)
    _product_embeddings_cache[cache_key] = embed_df
    _logger_instance.info(f"Generated and cached embeddings for {id_col} with shape {embed_df.shape}")
    return embed_df


def register_text_embedding_fgens():
    # These DIM constants should ideally be read from the config when the decorators are processed,
    # or the FeatureFactory should be made more dynamic. For now, they are placeholders.
    # Ensure these match the 'feature_config.text_processing.embedding_dimensions' for relevant features.
    # It's better to fetch this from config inside the generator or make FeatureFactory smarter.
    # For now, assuming a default or that FeatureFactory handles mismatched num_cols gracefully.
    PRODUCT_EMBED_DIMS = 20 
    CATEGORY_EMBED_DIMS = 20 

    @FeatureFactory.register(
        'product_embeddings',
        num_cols=[f'product_embed_{i}' for i in range(PRODUCT_EMBED_DIMS)] 
    )
    def generate_product_embeddings(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config 
    ) -> pl.DataFrame:
        text_processor = get_text_processor(config)
        product_embed_df = _get_product_embeddings_df(
            history_df, target_df, text_processor, config, 'product_id', 'product_name', 'product'
        )
        # Note: The actual number of columns in product_embed_df depends on config's embedding_dimensions.
        # The num_cols in decorator must align.
        return target_df.join(product_embed_df, on='product_id', how='left').fill_null(0)

    @FeatureFactory.register(
        'category_embeddings',
        num_cols=[f'cat_embed_{i}' for i in range(CATEGORY_EMBED_DIMS)] 
    )
    def generate_category_embeddings(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config 
    ) -> pl.DataFrame:
        text_processor = get_text_processor(config)
        h_df = history_df.with_columns(pl.col("product_category").fill_null("UNKNOWN_CATEGORY"))
        t_df = target_df.with_columns(pl.col("product_category").fill_null("UNKNOWN_CATEGORY"))
        category_embed_df = _get_product_embeddings_df(
            h_df, t_df, text_processor, config, 'product_category', 'product_category', 'cat'
        )
        return target_df.join(category_embed_df, on='product_category', how='left').fill_null(0)

    @FeatureFactory.register(
        'user_product_similarity', 
        num_cols=[
            'purchase_weighted_similarity',
            'cart_weighted_similarity',
            'max_purchase_similarity_history', 
            'max_cart_similarity_history',     
        ]
    )
    def generate_user_product_similarity(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        text_processor = get_text_processor(config)
        logger = _logger_instance
        logger.info("Starting user_product_similarity generation.")

        h_df_p = history_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT"))
        t_df_p = target_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT"))

        all_product_embeddings_df = _get_product_embeddings_df(
            h_df_p, t_df_p, text_processor, config, 'product_id', 'product_name'
        )
        
        default_feature_cols = {
            'purchase_weighted_similarity': 0.0, 'cart_weighted_similarity': 0.0,
            'max_purchase_similarity_history': 0.0, 'max_cart_similarity_history': 0.0
        }
        if all_product_embeddings_df.height == 0 or all_product_embeddings_df.width <=1 :
            logger.warning("No product embeddings available for user_product_similarity.")
            return target_df.with_columns([pl.lit(v).cast(pl.Float64).alias(k) for k,v in default_feature_cols.items()])

        embedding_cols = [col for col in all_product_embeddings_df.columns if col.startswith('product_embed_')]
        if not embedding_cols:
            logger.warning("Embedding columns not found in all_product_embeddings_df.")
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

            user_hist = user_hist_actions.group_by(['user_id', 'product_id']) \
                .agg(pl.count().alias('interaction_count'))
            user_hist = user_hist.join(all_product_embeddings_df.select(['product_id'] + embedding_cols), on='product_id', how='inner')
            
            user_weighted_embeddings_list = []
            user_all_historical_embeddings_list = []

            for user_id_key, group_df in user_hist.group_by('user_id', maintain_order=False):
                scalar_user_id_val: Any
                if isinstance(user_id_key, tuple): # Check for tuple first
                    if len(user_id_key) == 1:
                        scalar_user_id_val = user_id_key[0]
                    else:
                        logger.error(f"User ID key is multi-element tuple: {user_id_key}. Skipping."); continue
                elif isinstance(user_id_key, list): 
                    if len(user_id_key) == 1: scalar_user_id_val = user_id_key[0]
                    else: logger.error(f"User ID key is multi-element list: {user_id_key}. Skipping."); continue
                else: scalar_user_id_val = user_id_key
                
                if not isinstance(scalar_user_id_val, (int, str, np.integer)): 
                    logger.error(f"Scalar user_id {scalar_user_id_val} (type: {type(scalar_user_id_val)}) not int/str/np.integer. Skipping.")
                    continue
                user_id_as_string = str(scalar_user_id_val)

                embeddings_np_list = [product_embedding_dict.get(pid) for pid in group_df['product_id']]
                valid_embeddings = [emb for emb in embeddings_np_list if emb is not None and emb.ndim > 0 and emb.size > 0]
                
                if not valid_embeddings:
                    logger.debug(f"No valid embeddings for user {user_id_as_string} for action {action_type}")
                    continue

                weights = group_df['interaction_count'].to_numpy().astype(float)
                if len(valid_embeddings) != len(weights):
                    logger.warning(f"Mismatch valid_embeddings/weights for user {user_id_as_string}. Skipping.")
                    continue
                
                weighted_avg_emb = np.average(np.array(valid_embeddings), axis=0, weights=weights)
                user_weighted_embeddings_list.append({'user_id': user_id_as_string, weight_col_name: weighted_avg_emb.tolist()})
                user_all_historical_embeddings_list.append({'user_id': user_id_as_string, hist_embeddings_col_name: valid_embeddings})
            
            weighted_df = pl.DataFrame(user_weighted_embeddings_list, schema=empty_weighted_schema) if user_weighted_embeddings_list else pl.DataFrame([], schema=empty_weighted_schema)
            historical_df = pl.DataFrame(user_all_historical_embeddings_list, schema=empty_historical_schema) if user_all_historical_embeddings_list else pl.DataFrame([], schema=empty_historical_schema)
            return weighted_df, historical_df

        purchase_weighted_emb_df, purchase_hist_emb_df = get_user_history_embeddings(
            "AT_Purchase", "purchase_weighted_embedding", "purchase_historical_embeddings"
        )
        cart_weighted_emb_df, cart_hist_emb_df = get_user_history_embeddings(
            "AT_CartUpdate", "cart_weighted_embedding", "cart_historical_embeddings"
        )
        logger.info(f"Processed purchase history for {purchase_weighted_emb_df.height} users.")
        logger.info(f"Processed cart history for {cart_weighted_emb_df.height} users.")

        target_with_embeddings = target_df.join(all_product_embeddings_df.select(['product_id'] + embedding_cols), on='product_id', how='left')
        
        # Cast user_id to Utf8 for all joins with user-specific history DFs
        target_with_embeddings = target_with_embeddings.with_columns(pl.col('user_id').cast(pl.Utf8))

        target_with_embeddings = target_with_embeddings.join(purchase_weighted_emb_df, on='user_id', how='left')
        target_with_embeddings = target_with_embeddings.join(purchase_hist_emb_df, on='user_id', how='left')
        target_with_embeddings = target_with_embeddings.join(cart_weighted_emb_df, on='user_id', how='left')
        target_with_embeddings = target_with_embeddings.join(cart_hist_emb_df, on='user_id', how='left')
        
        results_data = []
        # Ensure original user_id and product_id types are preserved for the final features_df construction
        original_user_id_dtype = target_df['user_id'].dtype
        original_product_id_dtype = target_df['product_id'].dtype

        for row_dict in tqdm(target_with_embeddings.iter_rows(named=True), total=len(target_with_embeddings)):
            target_emb_list = [row_dict.get(col) for col in embedding_cols] 
            
            current_features = default_feature_cols.copy()
            # Use original user_id from row_dict which comes from target_df before Utf8 casting for join
            current_features['user_id'] = row_dict['user_id']
            current_features['product_id'] = row_dict['product_id']


            if any(x is None for x in target_emb_list):
                results_data.append(current_features); continue
            target_emb = np.array(target_emb_list)
            if target_emb.size == 0 or np.linalg.norm(target_emb) < 1e-9: 
                results_data.append(current_features); continue

            user_purchase_emb_list = row_dict.get('purchase_weighted_embedding')
            if user_purchase_emb_list:
                user_purchase_emb = np.array(user_purchase_emb_list)
                if target_emb.shape == user_purchase_emb.shape and np.linalg.norm(user_purchase_emb) > 1e-9:
                     current_features['purchase_weighted_similarity'] = sklearn_cosine_similarity(target_emb.reshape(1, -1), user_purchase_emb.reshape(1, -1))[0, 0]
            
            purchase_hist_embeds_list_of_np = row_dict.get('purchase_historical_embeddings') 
            if purchase_hist_embeds_list_of_np:
                valid_hist_embeds = [e for e in purchase_hist_embeds_list_of_np if isinstance(e, np.ndarray) and e.shape == target_emb.shape and np.linalg.norm(e) > 1e-9]
                if valid_hist_embeds:
                    similarities = sklearn_cosine_similarity(target_emb.reshape(1, -1), np.array(valid_hist_embeds))
                    if similarities.size > 0: current_features['max_purchase_similarity_history'] = np.max(similarities)
            
            user_cart_emb_list = row_dict.get('cart_weighted_embedding')
            if user_cart_emb_list:
                user_cart_emb = np.array(user_cart_emb_list)
                if target_emb.shape == user_cart_emb.shape and np.linalg.norm(user_cart_emb) > 1e-9:
                    current_features['cart_weighted_similarity'] = sklearn_cosine_similarity(target_emb.reshape(1, -1), user_cart_emb.reshape(1, -1))[0, 0]

            cart_hist_embeds_list_of_np = row_dict.get('cart_historical_embeddings')
            if cart_hist_embeds_list_of_np:
                valid_hist_embeds = [e for e in cart_hist_embeds_list_of_np if isinstance(e, np.ndarray) and e.shape == target_emb.shape and np.linalg.norm(e) > 1e-9]
                if valid_hist_embeds:
                    similarities = sklearn_cosine_similarity(target_emb.reshape(1, -1), np.array(valid_hist_embeds))
                    if similarities.size > 0: current_features['max_cart_similarity_history'] = np.max(similarities)
            
            results_data.append(current_features)
        
        features_schema = {'user_id': original_user_id_dtype, 'product_id': original_product_id_dtype}
        features_schema.update({k: pl.Float64 for k in default_feature_cols})
        
        features_df = pl.DataFrame(results_data, schema=features_schema) if results_data else pl.DataFrame([], schema=features_schema)
        
        final_df = target_df.join(features_df, on=['user_id', 'product_id'], how='left')
        for col_name in default_feature_cols:
            final_df = final_df.with_columns(pl.col(col_name).fill_null(0.0).cast(pl.Float64))
        
        logger.info("Finished user_product_similarity generation.")
        # Return only original target_df columns + the new feature columns
        return final_df.select(target_df.columns + list(default_feature_cols.keys()))


    @FeatureFactory.register(
        'text_similarity_cluster',
        cat_cols=['product_text_cluster'], 
        num_cols=['cluster_purchase_ratio', 'cluster_cart_ratio', 'cluster_view_ratio']
    )
    def generate_text_similarity_cluster(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config
    ) -> pl.DataFrame:
        text_processor = get_text_processor(config)
        logger = _logger_instance
        logger.info("Starting text_similarity_cluster generation.")

        h_df_p = history_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT"))
        t_df_p = target_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT"))

        all_product_embeddings_df = _get_product_embeddings_df(
            h_df_p, t_df_p, text_processor, config, 'product_id', 'product_name'
        )
        
        default_cluster_cols = {
            'product_text_cluster': -1, 'cluster_purchase_ratio': 0.0,
            'cluster_cart_ratio': 0.0, 'cluster_view_ratio': 0.0
        }
        if all_product_embeddings_df.height == 0 or all_product_embeddings_df.width <=1:
            logger.warning("No product embeddings for text_similarity_cluster.")
            return target_df.with_columns([
                pl.lit(default_cluster_cols['product_text_cluster']).cast(pl.Int32).alias('product_text_cluster'),
                pl.lit(default_cluster_cols['cluster_purchase_ratio']).cast(pl.Float64).alias('cluster_purchase_ratio'),
                pl.lit(default_cluster_cols['cluster_cart_ratio']).cast(pl.Float64).alias('cluster_cart_ratio'),
                pl.lit(default_cluster_cols['cluster_view_ratio']).cast(pl.Float64).alias('cluster_view_ratio'),
            ])

        embedding_cols = [col for col in all_product_embeddings_df.columns if col.startswith('product_embed_')]
        if not embedding_cols:
             logger.warning("Embedding columns not found for text_similarity_cluster.")
             return target_df.with_columns([pl.lit(v).cast(pl.Float64 if isinstance(v, float) else pl.Int32).alias(k) for k,v in default_cluster_cols.items()])


        product_embeddings_np = all_product_embeddings_df.select(embedding_cols).to_numpy()
        product_ids_series = all_product_embeddings_df['product_id']

        if product_embeddings_np.shape[0] == 0: 
            logger.warning("No product embeddings numpy array for text_similarity_cluster.")
            return target_df.with_columns([pl.lit(v).cast(pl.Float64 if isinstance(v, float) else pl.Int32).alias(k) for k,v in default_cluster_cols.items()])

        try:
            from sklearn.cluster import KMeans 
            n_clusters = config.get('feature_config.text_processing.n_clusters', 15)
            if product_embeddings_np.shape[0] < n_clusters:
                logger.warning(f"Products ({product_embeddings_np.shape[0]}) < n_clusters ({n_clusters}). Adjusting.")
                n_clusters = max(1, product_embeddings_np.shape[0]) 

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto' if hasattr(KMeans, 'n_init') and KMeans().n_init == 'auto' else 10) 
            cluster_labels = kmeans.fit_predict(product_embeddings_np)
            
            product_cluster_df = pl.DataFrame({
                'product_id': product_ids_series,
                'product_text_cluster': cluster_labels.astype(np.int32)
            })
            logger.info(f"Clustered {len(product_ids_series)} products into {n_clusters} clusters.")

            history_with_clusters = history_df.join(product_cluster_df, on='product_id', how='left') \
                                            .with_columns(pl.col('product_text_cluster').fill_null(-1).cast(pl.Int32))

            def calculate_cluster_ratios(action_type: str, ratio_col_name: str) -> pl.DataFrame:
                user_cluster_action = history_with_clusters.filter(pl.col('action_type') == action_type) \
                    .group_by(['user_id', 'product_text_cluster']) \
                    .agg(pl.count().alias('action_in_cluster_count'))
                
                user_total_actions = user_cluster_action.group_by('user_id') \
                    .agg(pl.sum('action_in_cluster_count').alias('total_user_actions'))
                
                user_cluster_ratios_df = user_cluster_action.join(user_total_actions, on='user_id', how='left') \
                    .with_columns(
                        (pl.col('action_in_cluster_count') / pl.col('total_user_actions')).fill_null(0.0).alias(ratio_col_name)
                    ) \
                    .select(['user_id', 'product_text_cluster', ratio_col_name])
                return user_cluster_ratios_df

            purchase_ratios_df = calculate_cluster_ratios("AT_Purchase", "cluster_purchase_ratio")
            cart_ratios_df = calculate_cluster_ratios("AT_CartUpdate", "cluster_cart_ratio")
            view_ratios_df = calculate_cluster_ratios("AT_View", "cluster_view_ratio") 

            result_df = target_df.join(product_cluster_df, on='product_id', how='left')
            # Cast user_id to string for joining with ratio DFs if their user_id became string (it shouldn't here)
            # However, product_text_cluster in ratio DFs is Int32, ensure result_df matches or cast.
            result_df = result_df.with_columns(pl.col('product_text_cluster').fill_null(-1).cast(pl.Int32))


            result_df = result_df.join(purchase_ratios_df, on=['user_id', 'product_text_cluster'], how='left')
            result_df = result_df.join(cart_ratios_df, on=['user_id', 'product_text_cluster'], how='left')
            result_df = result_df.join(view_ratios_df, on=['user_id', 'product_text_cluster'], how='left')
            
            final_df = result_df.with_columns([
                pl.col('product_text_cluster').fill_null(-1).cast(pl.Int32), # Already done but good to ensure
                pl.col('cluster_purchase_ratio').fill_null(0.0).cast(pl.Float64),
                pl.col('cluster_cart_ratio').fill_null(0.0).cast(pl.Float64),
                pl.col('cluster_view_ratio').fill_null(0.0).cast(pl.Float64)
            ])

            logger.info(f"Generated text similarity cluster features for {final_df.height} rows.")
            return final_df.select(target_df.columns + ['product_text_cluster', 'cluster_purchase_ratio', 'cluster_cart_ratio', 'cluster_view_ratio'])

        except ImportError:
            logger.warning("scikit-learn not available for KMeans. Skipping text similarity clustering.")
            return target_df.with_columns([pl.lit(v).cast(pl.Float64 if isinstance(v, float) else pl.Int32).alias(k) for k,v in default_cluster_cols.items()])
        except Exception as e:
            logger.error(f"Error during text_similarity_cluster: {e}")
            return target_df.with_columns([pl.lit(v).cast(pl.Float64 if isinstance(v, float) else pl.Int32).alias(k) for k,v in default_cluster_cols.items()])


    @FeatureFactory.register( # type: ignore
        'text_diversity_features',
        num_cols=['distance_from_centroid', 'relative_diversity']
    )
    def generate_text_diversity_features_vectorized(
        history_df: pl.DataFrame, target_df: pl.DataFrame, config: Config # type: ignore
    ) -> pl.DataFrame:
        text_processor = get_text_processor(config)
        logger = _logger_instance

        logger.info("Starting text_diversity_features generation (vectorized).")

        h_df_p = history_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT"))
        t_df_p = target_df.with_columns(pl.col("product_name").fill_null("UNKNOWN_PRODUCT"))

        all_product_embeddings_df = _get_product_embeddings_df(
            h_df_p, t_df_p, text_processor, config, 'product_id', 'product_name'
        )
        
        default_diversity_cols = {'distance_from_centroid': 0.0, 'relative_diversity': 0.0}
        
        # Validate product embeddings DataFrame
        if all_product_embeddings_df.height == 0 or all_product_embeddings_df.width <= 1: # Needs product_id + at least one embed col
            logger.warning("No product embeddings available for text_diversity_features. Returning defaults.")
            return target_df.with_columns([pl.lit(v).cast(pl.Float64).alias(k) for k,v in default_diversity_cols.items()])

        embedding_cols = [col for col in all_product_embeddings_df.columns if col.startswith('product_embed_')]
        if not embedding_cols:
            logger.warning("Embedding columns not found in all_product_embeddings_df. Returning defaults.")
            return target_df.with_columns([pl.lit(v).cast(pl.Float64).alias(k) for k,v in default_diversity_cols.items()])
        
        dim = len(embedding_cols)

        # --- Part 1: Calculate user embedding statistics (vectorized) ---
        user_history_actions = history_df.filter(
            pl.col('action_type').is_in(["AT_Purchase", "AT_View"])
        )

        user_stats_df_schema = {'user_id':pl.Utf8, 'embedding_centroid_list':pl.List(pl.Float64), 'embedding_diversity':pl.Float64}
        if user_history_actions.height == 0:
            logger.info("No relevant user actions in history_df. Diversity features will use defaults.")
            user_stats_df = pl.DataFrame(schema=user_stats_df_schema)
        else:
            user_history_with_embeds = user_history_actions.join(
                all_product_embeddings_df.select(['product_id'] + embedding_cols), 
                on='product_id', 
                how='inner' # Only consider history items for which we have embeddings
            )
            
            if user_history_with_embeds.height == 0 or user_history_with_embeds.select(embedding_cols).null_count().sum_horizontal()[0]: # Check if all embedding columns are null
                logger.info("No historical interactions could be mapped to valid embeddings. Diversity features will use defaults.")
                user_stats_df = pl.DataFrame(schema=user_stats_df_schema)
            else:
                # Calculate centroids per user: list of mean values for each embedding dimension
                centroid_component_exprs = [pl.mean(col_name).alias(f"centroid_{col_name}") for col_name in embedding_cols]
                user_centroids_grouped = user_history_with_embeds.group_by('user_id', maintain_order=False).agg(centroid_component_exprs)
                
                # Join centroids back to calculate distances within each group
                df_with_centroids = user_history_with_embeds.join(user_centroids_grouped, on='user_id', how='left')

                # Calculate squared L2 distance for each product embedding from its user's centroid
                sq_dist_to_centroid_exprs = [(pl.col(col_name) - pl.col(f"centroid_{col_name}")).pow(2) for col_name in embedding_cols]
                dist_to_centroid_expr = pl.sum_horizontal(sq_dist_to_centroid_exprs).sqrt().alias("dist_to_centroid")
                
                df_with_distances = df_with_centroids.with_columns(dist_to_centroid_expr)
                
                # Aggregate stats per user: centroid list and mean diversity
                user_stats_df = df_with_distances.group_by('user_id', maintain_order=False).agg(
                    # Create a list of centroid components
                    pl.concat_list([pl.first(f"centroid_{col_name}") for col_name in embedding_cols]).alias("embedding_centroid_list"),
                    # Calculate diversity: mean of distances to centroid, only if more than one item
                    pl.when(pl.count() > 1)
                        .then(pl.mean("dist_to_centroid"))
                        .otherwise(0.0) # Default diversity for users with 1 item
                        .fill_null(0.0) # Fill null if mean is null (e.g., all dist_to_centroid were null)
                        .alias("embedding_diversity")
                ).with_columns(
                    pl.col('user_id').cast(pl.Utf8) # Ensure user_id is Utf8 for consistent joining
                ).select(['user_id', 'embedding_centroid_list', 'embedding_diversity'])
                logger.info(f"Calculated embedding statistics for {user_stats_df.height} users.")

        # --- Part 2: Prepare target_df, join with stats, and calculate diversity features (vectorized) ---
        original_target_user_id_dtype = target_df['user_id'].dtype
        original_target_product_id_dtype = target_df['product_id'].dtype

        target_for_join = target_df.with_columns(pl.col('user_id').cast(pl.Utf8).alias('user_id_str_for_join'))
        
        target_with_embeds = target_for_join.join(
            all_product_embeddings_df.select(['product_id'] + embedding_cols), 
            on='product_id', 
            how='left' # Keep all target rows, get nulls if no embedding
        )

        target_with_user_stats = target_with_embeds.join(
            user_stats_df, 
            left_on='user_id_str_for_join', 
            right_on='user_id', 
            how='left' # Keep all target rows, get nulls if no user stats
        )
        
        # Calculate distance_from_centroid and relative_diversity
        sq_diff_exprs_target = []
        for i, emb_col_name in enumerate(embedding_cols):
            sq_diff_exprs_target.append(
                (pl.col(emb_col_name) - pl.col('embedding_centroid_list').list.get(i)).pow(2)
            )
        dist_centroid_calc_expr = pl.sum_horizontal(sq_diff_exprs_target).sqrt()

        # Define mask for when calculations are possible
        target_embeds_not_null_mask = pl.all_horizontal([pl.col(col).is_not_null() for col in embedding_cols])
        centroid_list_valid_mask = (
            pl.col('embedding_centroid_list').is_not_null() &
            (pl.col('embedding_centroid_list').list.len() == dim) &
            # Ensure all elements within the centroid list needed for calculation are not null
            pl.all_horizontal([pl.col('embedding_centroid_list').list.get(i).is_not_null() for i in range(dim)])
        )
        valid_calculation_mask = target_embeds_not_null_mask & centroid_list_valid_mask

        features_calculated_df = target_with_user_stats.with_columns(
            pl.when(valid_calculation_mask)
                .then(dist_centroid_calc_expr)
                .otherwise(0.0) # Default distance
                .alias('distance_from_centroid')
        ).with_columns(
            # Fill nulls for user_diversity (from left join if user had no stats)
            pl.col('embedding_diversity').fill_null(0.0).alias('user_diversity_filled')
        )

        features_calculated_df = features_calculated_df.with_columns(
            pl.when(valid_calculation_mask & (pl.col('user_diversity_filled') > 1e-9))
                .then(pl.col('distance_from_centroid') / pl.col('user_diversity_filled'))
            .when(valid_calculation_mask & (pl.col('distance_from_centroid') > 1e-9)) # Executed if user_diversity_filled <= 1e-9
                .then(pl.col('distance_from_centroid'))
            .otherwise(0.0) # Default relative diversity
            .alias('relative_diversity')
        )

        # Select final features, ensuring original user_id and product_id types for the join
        features_df = features_calculated_df.select(
            pl.col('user_id').cast(original_target_user_id_dtype), 
            pl.col('product_id').cast(original_target_product_id_dtype), 
            pl.col('distance_from_centroid').cast(pl.Float64),
            pl.col('relative_diversity').cast(pl.Float64)
        )
        
        # Apply unique operation, similar to original logic
        # This is important if target_df could have (user_id, product_id) duplicates and you need one feature set per pair
        features_df = features_df.unique(
            subset=['user_id', 'product_id'],
            keep='last' 
        )
            
        final_df = target_df.join(features_df, on=['user_id', 'product_id'], how='left')
        
        # Fill nulls for features that couldn't be calculated or for rows not in features_df after unique
        for col_name, default_val in default_diversity_cols.items():
            final_df = final_df.with_columns(pl.col(col_name).fill_null(default_val).cast(pl.Float64))
        
        logger.info(f"Generated text diversity features for {final_df.height} rows.")
        # Ensure final selection matches original requirement (target_df columns + new feature columns)
        return final_df.select(target_df.columns + list(default_diversity_cols.keys()))