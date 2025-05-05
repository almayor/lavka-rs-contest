import hashlib
import pickle
import polars as pl

from functools import wraps
from pathlib import Path
from typing import Any, Optional

from .config import Config
from .custom_logging import get_logger


class FeatureFactory:
    """Feature generation with selective feature creation"""
    
    # Class-level registry of feature generators
    _fgen_registry = {}
    _possible_targets = set()
    
    @classmethod
    def register(cls,
                 fgen_name: str,
                 num_cols: list[str] | None = None,
                 cat_cols: list[str] | None = None,
                 depends_on: str | list[str] | None = None,):
        """
        Decorator to register a method as a feature generator and to specify its dependencies. The method
        must accept two arguments: history_df and target_df, and return a target_df with new columns.
        Args:
            fgen_name (str): Name of the feature generator
            num_cols (list[str] | None): List of numerical columns this generator produces.
            cat_cols (list[str] | None): List of categorical columns this generator produces.
            depends_on (str | list[str] | None): List of feature generators that this feature generator depends on.
        """
        depends_on = depends_on or []
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            cls._fgen_registry[fgen_name] = {
                'func': wrapper,
                'depends_on': depends_on,
                'num_cols': num_cols or [],
                'cat_cols': cat_cols or [],
            }
            return wrapper
        
        return decorator

    @classmethod
    def register_target(cls,
                        target_name: str,
                        depends_on: str | list[str] | None = None):
        """
        Decorator to register a method as a target generator and to specify its dependencies. The method
        must accept two arguments: history_df and target_df, and return a pl.Series of the target. Rows where
        the target is null will be filtered out.
        Args:
            target_name (str): Name of the target to be generated.
            depends_on (str | list[str] | None): List of feature generators that this target depends on.
        """
        depends_on = depends_on or []
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return result, []
        
            cls._fgen_registry[target_name] = {
                'func': wrapper,
                'depends_on': depends_on,
                'num_cols': [],
                'cat_cols': [],
            }
            cls._possible_targets.add(target_name)

            return wrapper
        
        return decorator

    def __init__(self, config: Config):
        """Initialize feature factory"""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    def generate_batch(
            self, history_df, target_df, requested_fgens=None, requested_target=None
        ) -> tuple[pl.DataFrame, pl.Series, list[str], pl.Series]:
        """
        Generate features and target for a batch of requests.
        Args:
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            requested_fgens (str | List[str] | None): Feature generators to invoke (if None, config is used).
            requested_target (str | None): Target to generate (if None, config is used).
        Returns:
            Tuple[pl.DataFrame, pl.Series, List[str], pl.Series]: Tuple containing:
                - Generated features (pl.DataFrame)
                - Target (pl.Series)
                - Categorical column names in the generated features (List[str])
                - Request IDs per row (pl.Series)
        """
        request_ids = target_df['request_id']
        features, cat_columns = self.generate_features(history_df, target_df, requested_fgens)
        target = self.generate_target(history_df, target_df, requested_target)
        mask = ~target.is_null()
        return (
            features.filter(mask),
            target.filter(mask),
            cat_columns,
            request_ids.filter(mask),
        )
        
    def generate_features_only(
            self, history_df: pl.DataFrame, target_df: pl.DataFrame, requested_fgens: list[str] | None = None
        ) -> tuple[pl.DataFrame, list[str], pl.Series]:
        """
        Generate only features without target (for prediction/inference).
        Args:
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            requested_fgens (List[str] | None): Feature generators to generate (if None, config is used).
        Returns:
            Tuple[pl.DataFrame, List[str], pl.Series]: Tuple containing:
                - Generated features (pl.DataFrame)
                - Categorical column names in the generated features (List[str])
                - Request IDs per row (pl.Series)
        """
        request_ids = target_df['request_id']
        features, cat_columns = self.generate_features(history_df, target_df, requested_fgens)
        return features, cat_columns, request_ids

    def generate_features(
            self, history_df: pl.DataFrame, target_df: pl.DataFrame, requested_fgens: list[str] | None = None
        ) -> tuple[pl.DataFrame, list[str]]:
        """
        Generate only the requested features and their dependencies
        Args:
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            requested_fgens (List[str] | None): Feature generators to invoke (if None, config is used).
        Returns:
            Tuple[pl.DataFrame, List[str]]: Tuple containing:
                - Generated features (pl.DataFrame)
                - Categorical column names in the generated features (List[str])
        """
        if requested_fgens is None:
            requested_fgens = self.config.get("features")
        if len(requested_fgens) != len(set(requested_fgens)):
            self.logger.error("Duplicate feature names in requested_fgens")
            raise ValueError("Duplicate feature names in requested_fgens")
        self.logger.info(f"Invoking feature generators: {', '.join(requested_fgens)}")
                
        # Generate each requested feature (and dependencies)
        all_columns, all_cat_columns = set(), set()
        for fgen in requested_fgens:
            target_df = self._generate_feature(
                fgen, history_df, target_df
            )
            cat_columns = self.__class__._fgen_registry[fgen]['cat_cols']
            num_columns = self.__class__._fgen_registry[fgen]['num_cols']
            all_cat_columns.update(cat_columns)
            all_columns.update(cat_columns)
            all_columns.update(num_columns)
        
        all_cat_columns = list(all_cat_columns) if len(all_cat_columns) else None
        self.logger.info("Joined features")
        self.logger.info(f"All column names: {all_columns}")
        self.logger.info(f"All categorical column names: {all_cat_columns}")
        
        target_df = target_df.select(all_columns)
        return target_df, all_cat_columns
    
    def generate_target(self, history_df, target_df, requested_target: str | None = None) -> pl.Series:
        """
        Generate the target variable.
        Args:
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            requested_target (str | None): Target to generate (if None, config is used).
        Returns:
            pl.Series: Generated target (pl.Series)
        """
        if requested_target is None:
            requested_target = self.config.get("target")
        if requested_target not in self.__class__._possible_targets:
            raise ValueError(f"Unknown target {requested_target}")

        feature, _ = self._generate_feature(requested_target, history_df, target_df)
        return feature
    
    def _generate_feature(
            self, fgen_name: str, history_df: pl.DataFrame, target_df: pl.DataFrame,
            invoked_fgens=None
        ) -> pl.DataFrame:
        """
        Generate a single feature, handling dependencies
        Args:
            fgen_name (str): Name of the feature generator to invoke.
            history_df (pl.DataFrame): Historical data.
            target_df (pl.DataFrame): Target data.
            invoked_fgens (set | None): Set of already invoked feature generators (for caching).
        Returns:
            Tuple[pl.DataFrame, List[str], List[str]]: Tuple containing:
                - Generated features (pl.DataFrame)
        """
        # Return from cache if already generated
        if invoked_fgens and fgen_name in invoked_fgens:
            return target_df
        
        # Check if feature exists
        if fgen_name not in self.__class__._fgen_registry:
            self.logger.error(f"Feature '{fgen_name}' is not registered")
            self.logger.error(f"Available features: {list(self.__class__._fgen_registry.keys())}")
            raise ValueError(f"Feature '{fgen_name}' is not registered")
        
        # Get feature info
        feature_info = self.__class__._fgen_registry[fgen_name]
        generator_func = feature_info['func']
        dependencies = feature_info['depends_on']
        
        # Generate dependencies first
        invoked_fgens = invoked_fgens or set()
        for dep_fgen in dependencies:
            self._generate_feature(dep_fgen, history_df, target_df, invoked_fgens)
        
        # Generate this feature
        self.logger.debug(f"Generating feature: {fgen_name}")
        features = generator_func(history_df, target_df)
    
        invoked_fgens.add(fgen_name)
        return features



class CachedFeatureFactory:
    """
    Wraps FeatureFactory.generate_batch and generate_features_only with disk caching.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.factory = FeatureFactory(config)

        # Setup cache directory
        cache_dir = Path(config.get('output.feature_cache_dir', 'feature_cache'))
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        self.enabled = config.get('feature_caching.enabled', True)

    def _cache_file(self, key: str) -> Path:
        return self.cache_dir / f"cache_{key}.pkl"

    def _load(self, key: str):
        if not self.enabled:
            return None
        path = self._cache_file(key)
        if path.exists():
            try:
                self.logger.debug(f"Loading from cache {path}")
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                self.logger.warning(f"Corrupted cache, removing {path}")
                path.unlink(missing_ok=True)
        return None

    def _save(self, key: str, data: Any) -> None:
        if not self.enabled:
            return
        path = self._cache_file(key)
        tmp = path.with_suffix('.tmp')
        try:
            with open(tmp, 'wb') as f:
                pickle.dump(data, f)
            tmp.replace(path)
            self.logger.debug(f"Saved cache {path}")
        except Exception as e:
            self.logger.warning(f"Could not write cache {path}: {e}")
            tmp.unlink(missing_ok=True)

    def generate_batch(
        self,
        history: pl.DataFrame,
        target: pl.DataFrame,
        feature_names: Optional[list[str]] = None,
        target_name: Optional[str] = None
    ) -> tuple[pl.DataFrame, Any, list[str], Any]:
        """
        Generate or load cached (features, target, cat_cols, request_ids).
        """
        feats = feature_names or self.config.get('features', [])  # type: ignore
        key = self._default_key(history, target, feats, target_name)

        cached = self._load(key)
        if cached is not None:
            self.logger.info("Using cached feature batch")
            return cached

        # No cache: generate
        self.logger.info("Generating feature batch")
        batch = self.factory.generate_batch(history, target, feats, target_name)  # type: ignore
        self._save(key, batch)
        return batch

    def generate_features_only(
        self,
        history: pl.DataFrame,
        target: pl.DataFrame,
        feature_names: Optional[list[str]] = None
    ) -> tuple[pl.DataFrame, list[str], pl.Series]:
        """
        Generate or load cached (features, cat_cols, request_ids).
        """
        batch = self.generate_batch(history, target, feature_names, None)
        feat_df, _, cat_cols, req_ids = batch
        return feat_df, cat_cols, req_ids

    def __getattr__(self, name: str) -> Any:
        return getattr(self.factory, name)
    
    @staticmethod
    def _default_key(
        history: pl.DataFrame,
        target: pl.DataFrame,
        features: list[str],
        target_name: Optional[str]
    ) -> str:
        """
        Create a simple cache key based on time ranges and feature list.
        """
        t0 = int(history['timestamp'].min().timestamp()) if not history.is_empty() else 0
        t1 = int(target['timestamp'].max().timestamp()) if not target.is_empty() else 0
        feats = ','.join(sorted(features))
        name = target_name or ''
        raw = f"{t0}-{t1}-{feats}-{name}"
        return hashlib.md5(raw.encode()).hexdigest()
