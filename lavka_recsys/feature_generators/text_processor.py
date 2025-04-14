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
        """Load SentenceTransformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = self.config.get(
                'text_processing.model_name', 
                'paraphrase-multilingual-MiniLM-L12-v2'  # Small but effective model
            )
            self.model = SentenceTransformer(model_name)
            self.embedding_size = self.model.get_sentence_embedding_dimension()
            self.model_type = 'sentence-transformers'
            self.logger.info(f"Loaded sentence-transformers model: {model_name}")
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
        if self.model is None:
            self.logger.warning("No text model loaded. Returning zero embeddings.")
            return np.zeros((len(texts), 1))
            
        # Clean texts (remove empty strings, handle non-string inputs)
        cleaned_texts = [
            t if isinstance(t, str) and t else " " 
            for t in texts
        ]
        
        if self.model_type == 'sentence-transformers':
            # Get embeddings directly from the model
            embeddings = self.model.encode(
                cleaned_texts, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings
            
        elif self.model_type == 'word2vec':
            # For word2vec, average word vectors in each text
            embeddings = np.zeros((len(cleaned_texts), self.embedding_size))
            
            for i, text in enumerate(cleaned_texts):
                words = text.lower().split()
                vectors = [
                    self.model[word] for word in words 
                    if word in self.model
                ]
                
                if vectors:
                    embeddings[i] = np.mean(vectors, axis=0)
            
            return embeddings
            
        elif self.model_type == 'fasttext':
            # Get embeddings directly from the model
            embeddings = np.array([
                self.model.get_sentence_vector(text) 
                for text in cleaned_texts
            ])
            return embeddings
            
        else:
            self.logger.warning(f"Unknown model type: {self.model_type}")
            return np.zeros((len(texts), 1))
    
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
            from sklearn.decomposition import PCA
            pca = PCA(n_components=dimensions)
            reduced = pca.fit_transform(embeddings)
            self.logger.info(
                f"Reduced embeddings from {embeddings.shape[1]} to {dimensions} dimensions"
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
    
    @FeatureFactory.register('product_embeddings')
    def generate_product_embeddings(
        history_df: pl.DataFrame, target_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Generate product name embeddings using pretrained model"""
        config = Config()
        text_processor = TextProcessor(config)
        
        # Get unique product names
        products = pl.concat([
            history_df.select('product_id', 'product_name').unique(),
            target_df.select('product_id', 'product_name').unique()
        ]).unique(subset=['product_id'])
        
        # Get embeddings
        product_names = products['product_name'].to_list()
        embeddings = text_processor.get_embeddings(product_names)
        
        # Reduce dimensions if needed
        dimensions = config.get('text_processing.embedding_dimensions', 20)
        if embeddings.shape[1] > dimensions:
            embeddings = text_processor.reduce_dimensions(embeddings, dimensions)
        
        # Create feature columns
        feature_names = [f'product_embed_{i}' for i in range(embeddings.shape[1])]
        
        # Convert to DataFrame
        import pandas as pd
        embed_df = pd.DataFrame(
            embeddings, 
            columns=feature_names,
            index=products['product_id'].to_list()
        )
        embed_df['product_id'] = products['product_id'].to_list()
        
        # Convert to polars
        embed_pl = pl.from_pandas(embed_df)
        
        # Join with target DataFrame
        result = target_df.join(embed_pl, on='product_id', how='left')
        
        return result
    
    @FeatureFactory.register('category_embeddings')
    def generate_category_embeddings(
        history_df: pl.DataFrame, target_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Generate category name embeddings using pretrained model"""
        config = Config()
        text_processor = TextProcessor(config)
        
        # Get unique category names
        categories = pl.concat([
            history_df.select('product_category').unique(),
            target_df.select('product_category').unique()
        ]).unique(subset=['product_category'])
        
        # Get embeddings
        category_names = categories['product_category'].to_list()
        embeddings = text_processor.get_embeddings(category_names)
        
        # Reduce dimensions if needed
        dimensions = config.get('text_processing.embedding_dimensions', 20)
        if embeddings.shape[1] > dimensions:
            embeddings = text_processor.reduce_dimensions(embeddings, dimensions)
        
        # Create feature columns
        feature_names = [f'category_embed_{i}' for i in range(embeddings.shape[1])]
        
        # Convert to DataFrame
        import pandas as pd
        embed_df = pd.DataFrame(
            embeddings, 
            columns=feature_names,
            index=categories['product_category'].to_list()
        )
        embed_df['product_category'] = categories['product_category'].to_list()
        
        # Convert to polars
        embed_pl = pl.from_pandas(embed_df)
        
        # Join with target DataFrame
        result = target_df.join(embed_pl, on='product_category', how='left')
        
        return result
    
    @FeatureFactory.register('user_product_distance')
    def generate_user_product_distance(
        history_df: pl.DataFrame, target_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Generate features based on embedding distance between target products
        and products previously purchased by each user.
        
        Args:
            history_df: DataFrame containing user purchase history
            target_df: DataFrame containing target products to score
            
        Returns:
            DataFrame with added distance features
        """
        raise NotImplementedError()

    return True
