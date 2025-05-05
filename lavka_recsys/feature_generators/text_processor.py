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
def register_text_embedding_fgens():
    """Register text embedding methods with FeatureFactory"""
    from ..feature_factory import FeatureFactory
    
    @FeatureFactory.register(
        'product_embeddings',
        num_cols=[
            'product_embed_0',
            'product_embed_1',
            'product_embed_2',
            'product_embed_3',
            'product_embed_4',
            'product_embed_5',
            'product_embed_6',
            'product_embed_7',
            'product_embed_8',
            'product_embed_9',
        ]
    )
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
    
    @FeatureFactory.register(
        'category_embeddings',
        num_cols=[
            'category_embed_0',
            'category_embed_1',
            'category_embed_2',
            'category_embed_3',
            'category_embed_4',
            'category_embed_5',
            'category_embed_6',
            'category_embed_7',
            'category_embed_8',
            'category_embed_9',
        ]
    )
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
    
    @FeatureFactory.register(
        'user_product_distance',
        num_cols=[
            'purchase_weighted_similarity',
            'cart_weighted_similarity',
            'min_purchase_similarity',
            'min_cart_similarity',
        ]
    )
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
        
        # Create embedding lookup dictionary
        product_id_list = products['product_id'].to_list()
        product_embedding_dict = {
            pid: embedding for pid, embedding in zip(product_id_list, product_embeddings)
        }
        
        logger.info(f"Created embedding lookup with {len(product_embedding_dict)} products")
        
        # Step 2: Calculate user purchase history embeddings (weighted by frequency)
        user_purchase_history = {}
        user_cart_history = {}
        
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
        
        # Create weighted embeddings for each user's purchase history
        for user_id in purchase_weights_pd['user_id'].unique():
            user_purchases = purchase_weights_pd[purchase_weights_pd['user_id'] == user_id]
            
            # Skip if no purchase history (shouldn't happen but just in case)
            if len(user_purchases) == 0:
                continue
                
            # Get embeddings and weights for products this user has purchased
            user_product_embeddings = []
            user_product_weights = []
            
            for _, row in user_purchases.iterrows():
                product_id = row['product_id']
                purchase_count = row['purchase_count']
                
                if product_id in product_embedding_dict:
                    user_product_embeddings.append(product_embedding_dict[product_id])
                    user_product_weights.append(purchase_count)
            
            # Skip if no valid embeddings
            if len(user_product_embeddings) == 0:
                continue
                
            # Normalize weights
            total_weight = sum(user_product_weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in user_product_weights]
                
                # Calculate weighted average embedding
                weighted_embedding = np.average(
                    user_product_embeddings, 
                    axis=0, 
                    weights=normalized_weights
                )
                
                # Store in dictionary
                user_purchase_history[user_id] = {
                    'weighted_embedding': weighted_embedding,
                    'product_embeddings': user_product_embeddings
                }
        
        # Same for cart history
        for user_id in cart_weights_pd['user_id'].unique():
            user_carts = cart_weights_pd[cart_weights_pd['user_id'] == user_id]
            
            # Skip if no cart history
            if len(user_carts) == 0:
                continue
                
            # Get embeddings and weights for products this user has added to cart
            user_product_embeddings = []
            user_product_weights = []
            
            for _, row in user_carts.iterrows():
                product_id = row['product_id']
                cart_count = row['cart_count']
                
                if product_id in product_embedding_dict:
                    user_product_embeddings.append(product_embedding_dict[product_id])
                    user_product_weights.append(cart_count)
            
            # Skip if no valid embeddings
            if len(user_product_embeddings) == 0:
                continue
                
            # Normalize weights
            total_weight = sum(user_product_weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in user_product_weights]
                
                # Calculate weighted average embedding
                weighted_embedding = np.average(
                    user_product_embeddings, 
                    axis=0, 
                    weights=normalized_weights
                )
                
                # Store in dictionary
                user_cart_history[user_id] = {
                    'weighted_embedding': weighted_embedding,
                    'product_embeddings': user_product_embeddings
                }
        
        logger.info(f"Created weighted embeddings for {len(user_purchase_history)} users' purchase history")
        logger.info(f"Created weighted embeddings for {len(user_cart_history)} users' cart history")
        
        # Step 3: Calculate distances from target products to user's history
        # Helper function to calculate cosine similarity
        def cosine_similarity(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0
            return np.dot(a, b) / (norm_a * norm_b)
        
        # Helper function to calculate minimum distance to any product in a list
        def min_distance(embedding, product_embeddings):
            if not product_embeddings:
                return 0
            similarities = [cosine_similarity(embedding, p_embed) for p_embed in product_embeddings]
            return max(similarities)  # Max similarity = min distance
        
        # Process target data in batches 
        result_dfs = []
        
        # Convert to pandas for easier processing
        target_pd = target_df.to_pandas()
        
        target_features = []
        
        for _, row in target_pd.iterrows():
            user_id = row['user_id']
            product_id = row['product_id']
            
            # Initialize features
            purchase_weighted_similarity = 0
            cart_weighted_similarity = 0
            min_purchase_distance = 0
            min_cart_distance = 0
            
            # Get target product embedding
            if product_id in product_embedding_dict:
                target_embedding = product_embedding_dict[product_id]
                
                # Calculate similarity to purchase history
                if user_id in user_purchase_history:
                    purchase_history_embedding = user_purchase_history[user_id]['weighted_embedding']
                    purchase_weighted_similarity = cosine_similarity(target_embedding, purchase_history_embedding)
                    
                    # Min distance to any purchased product
                    min_purchase_distance = min_distance(
                        target_embedding, 
                        user_purchase_history[user_id]['product_embeddings']
                    )
                
                # Calculate similarity to cart history
                if user_id in user_cart_history:
                    cart_history_embedding = user_cart_history[user_id]['weighted_embedding']
                    cart_weighted_similarity = cosine_similarity(target_embedding, cart_history_embedding)
                    
                    # Min distance to any carted product
                    min_cart_distance = min_distance(
                        target_embedding, 
                        user_cart_history[user_id]['product_embeddings']
                    )
            
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
        import pandas as pd
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
    
    @FeatureFactory.register(
        'text_similarity_cluster',
        cat_cols=['cluster'],
        num_cols=['cluster_purchase_ratio']
    )
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
        
        # Step 2: Cluster products using k-means
        try:
            from sklearn.cluster import KMeans
            
            # Choose number of clusters
            n_clusters = config.get('text_processing.n_clusters', 15)
            
            # Create and fit k-means model
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(product_embeddings)
            
            # Create product_id to cluster mapping
            product_id_list = products['product_id'].to_list()
            product_cluster_dict = {
                pid: label for pid, label in zip(product_id_list, cluster_labels)
            }
            
            logger.info(f"Clustered {len(products)} products into {n_clusters} clusters")
            
            # Step 3: Calculate user interaction statistics with each cluster
            # Get purchase history
            purchase_history = history_df.filter(pl.col('action_type') == "AT_Purchase").select(
                'user_id', 'product_id'
            )
            
            # Add cluster labels to purchase history
            purchase_history_pd = purchase_history.to_pandas()
            purchase_history_pd['cluster'] = purchase_history_pd['product_id'].map(
                lambda pid: product_cluster_dict.get(pid, -1)
            )
            
            # Count purchases per user-cluster
            user_cluster_counts = purchase_history_pd.groupby(['user_id', 'cluster']).size().reset_index(
                name='cluster_purchase_count'
            )
            
            # Calculate total purchases per user
            user_total_purchases = user_cluster_counts.groupby('user_id')['cluster_purchase_count'].sum().reset_index(
                name='total_purchases'
            )
            
            # Calculate cluster purchase ratio
            user_cluster_counts = user_cluster_counts.merge(
                user_total_purchases, on='user_id', how='left'
            )
            user_cluster_counts['cluster_purchase_ratio'] = (
                user_cluster_counts['cluster_purchase_count'] / user_cluster_counts['total_purchases']
            )
            
            # Step 4: Add cluster labels and purchase ratios to target data
            # Add cluster labels to target DataFrame
            target_pd = target_df.to_pandas()
            target_pd['cluster'] = target_pd['product_id'].map(
                lambda pid: product_cluster_dict.get(pid, -1)
            )
            
            # Merge with user-cluster purchase ratios
            result_df = target_pd.merge(
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
    
    @FeatureFactory.register(
        'text_diversity_features',
        num_cols=['distance_from_centroid', 'relative_diversity']
    )
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
        
        # Create embedding lookup dictionary
        product_id_list = products['product_id'].to_list()
        product_embedding_dict = {
            pid: embedding for pid, embedding in zip(product_id_list, product_embeddings)
        }
        
        # Step 2: Calculate user purchase history embedding statistics
        # Get user purchase/view history
        user_history = history_df.filter(
            (pl.col('action_type') == "AT_Purchase") | (pl.col('action_type') == "AT_View")
        ).select('user_id', 'product_id', 'action_type')
        
        # Group by user to get unique products
        user_history_grouped = user_history.group_by('user_id', 'product_id').agg(
            pl.col('action_type').count().alias('interaction_count')
        )
        
        # Convert to pandas for easier processing
        user_history_pd = user_history_grouped.to_pandas()
        
        # Calculate user embedding statistics
        user_stats = {}
        
        for user_id in user_history_pd['user_id'].unique():
            user_products = user_history_pd[user_history_pd['user_id'] == user_id]['product_id'].tolist()
            
            # Get embeddings for this user's products
            user_product_embeddings = [
                product_embedding_dict[pid] for pid in user_products 
                if pid in product_embedding_dict
            ]
            
            if len(user_product_embeddings) == 0:
                continue
                
            # Calculate embedding centroid (mean)
            embedding_centroid = np.mean(user_product_embeddings, axis=0)
            
            # Calculate embedding diversity (average distance from centroid)
            distances = [
                np.linalg.norm(embed - embedding_centroid) 
                for embed in user_product_embeddings
            ]
            embedding_diversity = np.mean(distances) if distances else 0
            
            # Store in dictionary
            user_stats[user_id] = {
                'embedding_centroid': embedding_centroid,
                'embedding_diversity': embedding_diversity
            }
        
        logger.info(f"Calculated embedding statistics for {len(user_stats)} users")
        
        # Step 3: Calculate diversity features for target items
        target_features = []
        
        # Convert to pandas for easier processing
        target_pd = target_df.to_pandas()
        
        for _, row in target_pd.iterrows():
            user_id = row['user_id']
            product_id = row['product_id']
            
            # Initialize features
            distance_from_centroid = 0
            relative_diversity = 0
            
            # Get target product embedding
            if product_id in product_embedding_dict and user_id in user_stats:
                target_embedding = product_embedding_dict[product_id]
                
                # Calculate distance from user's centroid
                centroid = user_stats[user_id]['embedding_centroid']
                distance_from_centroid = np.linalg.norm(target_embedding - centroid)
                
                # Calculate relative diversity (how much more diverse this product is)
                user_diversity = user_stats[user_id]['embedding_diversity']
                if user_diversity > 0:
                    relative_diversity = distance_from_centroid / user_diversity
                else:
                    relative_diversity = distance_from_centroid  # If user has no diversity, use raw distance
            
            # Store features
            target_features.append({
                'user_id': user_id,
                'product_id': product_id,
                'distance_from_centroid': distance_from_centroid,
                'relative_diversity': relative_diversity
            })
        
        # Convert to DataFrame
        import pandas as pd
        features_df = pd.DataFrame(target_features)
        
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
