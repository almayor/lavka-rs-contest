from functools import wraps
from custom_logging import get_logger

class FeatureFactory:
    """Feature generation with selective feature creation"""
    
    # Class-level registry of feature generators
    _feature_registry = {}
    
    @classmethod
    def register(cls, feature_name, depends_on=None, category=None):
        """Decorator to register a method as a feature generator"""
        depends_on = depends_on or []
        
        def decorator(func):
            cls._feature_registry[feature_name] = {
                'func': func,
                'depends_on': depends_on,
                'category': category
            }
            
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                return func(self, *args, **kwargs)
            
            return wrapper
        
        return decorator

    def __init__(self):
        """Initialize feature factory"""
        self.features = {}  # Cache for generated features
        self.config = None
        self.logger = get_logger(self.__class__.__name__)
    
    def set_config(self, config):
        """Set configuration"""
        self.config = config
    
    def generate_features(self, history_df, target_df, requested_features):
        """Generate only the requested features and their dependencies"""
        self.logger.info(f"Generating features: {', '.join(requested_features)}")
        
        # Reset cache for new request
        self.features = {}
        
        # Expand feature groups if needed
        expanded_features = self._expand_feature_groups(requested_features)
        
        # Generate each requested feature (and dependencies)
        for feature_name in expanded_features:
            self._generate_feature(feature_name, history_df, target_df)
        
        # Return only the requested features
        return {f: self.features[f] for f in expanded_features if f in self.features}
    
    def _expand_feature_groups(self, requested_features):
        """Expand feature group names into individual features"""
        if not self.config:
            return requested_features
            
        expanded = []
        feature_groups = self.config.get('features')
        
        for feature in requested_features:
            if feature in feature_groups:
                # This is a feature group
                expanded.extend(feature_groups[feature])
            else:
                # This is an individual feature
                expanded.append(feature)
                
        return list(set(expanded))  # Remove duplicates
    
    def _generate_feature(self, feature_name, history_df, target_df):
        """Generate a single feature, handling dependencies"""
        # Return from cache if already generated
        if feature_name in self.features:
            return self.features[feature_name]
        
        # Check if feature exists
        if feature_name not in self.__class__._feature_registry:
            msg = f"Feature '{feature_name}' is not registered"
            self.logger.error(msg)
            raise ValueError(msg)
        
        # Get feature info
        feature_info = self.__class__._feature_registry[feature_name]
        generator_func = feature_info['func']
        dependencies = feature_info['depends_on']
        
        # Generate dependencies first
        for dep in dependencies:
            self._generate_feature(dep, history_df, target_df)
        
        # Generate this feature
        self.logger.debug(f"Generating feature: {feature_name}")
        feature_df = generator_func(self, history_df, target_df)
        
        # Cache and return
        self.features[feature_name] = feature_df
        return feature_df
    
    def join_features(self, base_df=None, common_keys=None):
        """Join all generated features into a single dataframe"""
        if not self.features:
            logger.warning("No features to join")
            return None
        
        if common_keys is None:
            common_keys = ['user_id', 'product_id']
            
        # Start with base_df or the first feature
        if base_df is not None:
            result = base_df
        else:
            result = list(self.features.values())[0]
        
        # Join the rest
        for feature_name, feature_df in self.features.items():
            if feature_df is not result:  # Don't join with itself
                try:
                    result = result.join(
                        feature_df, on=common_keys, how='left'
                    )
                except Exception as e:
                    self.logger.error(f"Error joining feature {feature_name}: {e}")
                    
        return result
    