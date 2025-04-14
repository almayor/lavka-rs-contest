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
            
        # Clean texts (remove empty strings, handle non-string inputs) - vectorized
        import numpy as np
        
        # Vectorized text cleaning
        cleaned_texts = np.array([
            t if isinstance(t, str) and t else " " 
            for t in texts
        ])
        
        if self.model_type == 'sentence-transformers':
            # Get embeddings directly from the model (already batch optimized)
            embeddings = self.model.encode(
                cleaned_texts, 
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=64  # Process in batches for better performance
            )
            return embeddings
            
        elif self.model_type == 'word2vec':
            # For word2vec, optimize with array operations
            embeddings = np.zeros((len(cleaned_texts), self.embedding_size))
            
            # Process in batches for better performance
            batch_size = 100
            num_batches = (len(cleaned_texts) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(cleaned_texts))
                batch_texts = cleaned_texts[start_idx:end_idx]
                
                # Process each text in the batch
                for i, text in enumerate(batch_texts):
                    words = text.lower().split()
                    # Get embeddings for all words in a single list comprehension
                    valid_word_vectors = [self.model[word] for word in words if word in self.model]
                    
                    # Only compute mean if we have valid vectors
                    if valid_word_vectors:
                        # Convert to array and compute mean in one operation
                        word_vectors_array = np.array(valid_word_vectors)
                        embeddings[start_idx + i] = np.mean(word_vectors_array, axis=0)
            
            return embeddings
            
        elif self.model_type == 'fasttext':
            # Optimize fasttext embedding generation
            # Pre-allocate output array
            embeddings = np.zeros((len(cleaned_texts), self.embedding_size))
            
            # Process in batches for better performance
            batch_size = 100
            num_batches = (len(cleaned_texts) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(cleaned_texts))
                batch_texts = cleaned_texts[start_idx:end_idx]
                
                # Create a batch of embeddings
                batch_embeddings = np.array([
                    self.model.get_sentence_vector(text) for text in batch_texts
                ])
                
                # Store in result array
                embeddings[start_idx:end_idx] = batch_embeddings
                
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
                
                # Calculate similarity to purchase history
                if user_id in user_purchase_history:
                    # Calculate weighted similarity
                    purchase_history_embedding = user_purchase_history[user_id]['weighted_embedding']
                    purchase_weighted_similarity = np.dot(target_embedding, purchase_history_embedding) / (
                        np.linalg.norm(target_embedding) * np.linalg.norm(purchase_history_embedding) + 1e-10
                    )
                    
                    # Calculate minimum distance to any purchased product using vectorized operations
                    user_product_matrix = user_purchase_history[user_id]['product_embeddings']
                    target_embedding_reshaped = target_embedding.reshape(1, -1)
                    
                    # Calculate all similarities at once
                    similarities = np.dot(user_product_matrix, target_embedding) / (
                        np.linalg.norm(user_product_matrix, axis=1) * np.linalg.norm(target_embedding) + 1e-10
                    )
                    min_purchase_distance = similarities.max() if len(similarities) > 0 else 0
                
                # Calculate similarity to cart history
                if user_id in user_cart_history:
                    # Calculate weighted similarity
                    cart_history_embedding = user_cart_history[user_id]['weighted_embedding']
                    cart_weighted_similarity = np.dot(target_embedding, cart_history_embedding) / (
                        np.linalg.norm(target_embedding) * np.linalg.norm(cart_history_embedding) + 1e-10
                    )
                    
                    # Calculate minimum distance to any carted product using vectorized operations
                    user_product_matrix = user_cart_history[user_id]['product_embeddings']
                    target_embedding_reshaped = target_embedding.reshape(1, -1)
                    
                    # Calculate all similarities at once
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
        
        # Step 2: Cluster products using k-means (vectorized)
        try:
            from sklearn.cluster import KMeans
            import numpy as np
            import pandas as pd
            
            # Choose number of clusters
            n_clusters = config.get('text_processing.n_clusters', 15)
            
            # Create and fit k-means model
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(product_embeddings)
            
            # Create product_id to cluster mapping as a vectorized operation
            product_id_list = products['product_id'].to_list()
            
            # Create a DataFrame with product_id and cluster_label
            product_clusters_df = pd.DataFrame({
                'product_id': product_id_list,
                'cluster': cluster_labels
            })
            
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
