import hashlib
import pickle
import polars as pl

from functools import wraps
from pathlib import Path
from typing import Any, Optional
from collections import Counter

from .utils.config import Config
from .utils.custom_logging import get_logger
from .feature_selector import FeatureSelector


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
            
        if fgen_name in cls._fgen_registry:
            raise ValueError(f'Feature generator {fgen_name} already exists')
        if num_cols:
            for known_fgen_name, known_fgen_data in cls._fgen_registry.items():
                intersect_cols = set(num_cols) & set(known_fgen_data['num_cols'])
                if intersect_cols:
                    raise ValueError(f'Both {known_fgen_name} and {fgen_name} have num columns {intersect_cols}')
        if cat_cols:
            for known_fgen_name, known_fgen_data in cls._fgen_registry.items():
                intersect_cols = set(cat_cols) & set(known_fgen_data['num_cols'])
                if intersect_cols:
                    raise ValueError(f'Both {known_fgen_name} and {fgen_name} have cat columns {intersect_cols}')
        
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
        self.feature_selector = FeatureSelector(config)
    
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
        requested_fgens = requested_fgens or self.config.get('feature_generators')
        features, cat_columns = self.generate_features(history_df, target_df, requested_fgens)
        target = self.generate_target(history_df, target_df, requested_target)
        mask = ~target.is_null() & ~target.is_nan()
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
        Invoke only the request feature generators and their dependencies
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
            requested_fgens = self.config.get("feature_generators")
        if len(requested_fgens) != len(set(requested_fgens)):
            duplicates = [item for item, count in Counter(requested_fgens).items() if count > 1]
            self.logger.error("Duplicate feature names in requested_fgens: " + ', '.join(duplicates))
            raise ValueError("Duplicate feature names in requested_fgens")
        self.logger.info(f"Invoking feature generators: {', '.join(requested_fgens)}")
                
        # Generate each requested feature (and dependencies)
        all_columns, all_cat_columns = set(), set()
        for fgen in requested_fgens:
            target_df = self._invoke_fgen(
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
        target_df, all_cat_columns = self.feature_selector(
            target_df, all_cat_columns
        )
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

        features, _ = self._invoke_fgen(requested_target, history_df, target_df)
        return features
    
    def _invoke_fgen(
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
            self._invoke_fgen(dep_fgen, history_df, target_df, invoked_fgens)
        
        # Generate this feature
        self.logger.info(f"Invoking feature generator: {fgen_name}")
        features = generator_func(history_df, target_df, self.config)
    
        invoked_fgens.add(fgen_name)
        return features

class CachedFeatureFactory(FeatureFactory):
    """
    A FeatureFactory that caches – and re-uses – the *result of every single feature
    generator* on disk.  The rest of FeatureFactory (dependency handling, typing, etc.)
    stays intact.
    """

    def __init__(self, config: Config):
        # call the parent so we inherit its logger, config, etc.
        super().__init__(config)

        self.enabled: bool = config.get("feature_config.caching.enabled", True)

        cache_root = config.get("output.feature_cache_dir", "results/feature_cache")
        self.cache_dir = Path(cache_root).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # NB: we *do not* look at unrelated parts of the config – only the
        # ``feature_config`` and (for convenience below) ``data`` section.
        self.data_cfg = config.get("data", {})

    def _cache_file(self, key: str) -> Path:
        """Return path <cache_dir>/<hash>.pkl"""
        return self.cache_dir / f"{key}.pkl"

    def _load(self, key: str) -> Optional[pl.DataFrame]:
        if not self.enabled:
            return None
        path = self._cache_file(key)
        if path.exists():
            try:
                with open(path, "rb") as fh:
                    self.logger.debug(f"Loaded cached f-gen ⇒ {path}")
                    return pickle.load(fh)
            except Exception:
                self.logger.warning(f"Corrupted cache file removed: {path}")
                path.unlink(missing_ok=True)
        return None

    def _save(self, key: str, data: pl.DataFrame) -> None:
        if not self.enabled:
            return
        path = self._cache_file(key)
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as fh:
                pickle.dump(data, fh)
            tmp.replace(path)
            self.logger.debug(f"Saved f-gen cache ⇒ {path}")
        except Exception as exc:
            self.logger.warning(f"Could not write cache: {path} ({exc})")
            tmp.unlink(missing_ok=True)

    def _feature_key(
        self,
        fgen_name: str,
        history_df: pl.DataFrame,
        target_df: pl.DataFrame,
    ) -> str:
        """
        Build a *stable* hash for the ‹fgen_name› **and** the pieces of data that
        influence its output.
        """
        # coarsely summarise the time span + row count for both inputs
        def _span(df: pl.DataFrame) -> tuple[int, int, int]:
            if df.is_empty():
                return (0, 0, 0)
            return (
                int(df["timestamp"].min().timestamp()),
                int(df["timestamp"].max().timestamp()),
                len(df),
            )

        h0, h1, hn = _span(history_df)
        t0, t1, tn = _span(target_df)

        raw = f"{fgen_name}|{h0}-{h1}-{hn}|{t0}-{t1}-{tn}"
        return hashlib.md5(raw.encode()).hexdigest()


    def _invoke_fgen( 
        self,
        fgen_name: str,
        history_df: pl.DataFrame,
        target_df: pl.DataFrame,
        invoked_fgens: set[str] | None = None,
    ) -> pl.DataFrame:
        """
        Intercepts FeatureFactory._invoke_fgen, adding:
        • Load-from-cache *before* we compute the generator
        • Save-to-cache *after* it runs
        The rest of the behaviour (dependency recursion, cycle detection,
        logging, etc.) is unchanged.
        """
        invoked_fgens = invoked_fgens or set()

        # short-circuit: already done in this call-stack
        if fgen_name in invoked_fgens:
            return target_df

        # 1) ––––– check cache FIRST
        key = self._feature_key(fgen_name, history_df, target_df)
        cached = self._load(key)
        if cached is not None:
            self.logger.info(f"Using cached '{fgen_name}' (key={key})")
            # Only add columns that are not already present
            for col in cached.columns:
                if col not in target_df.columns:
                    target_df = target_df.with_columns(cached[col])
            invoked_fgens.add(fgen_name)
            return target_df

        # 2) ––––– existing dependency logic (unchanged)
        if fgen_name not in self.__class__._fgen_registry:
            avail = ", ".join(self.__class__._fgen_registry.keys())
            raise ValueError(f"Feature '{fgen_name}' is not registered.  Available: {avail}")

        info = self.__class__._fgen_registry[fgen_name]
        dep_names = info["depends_on"]
        for dep in dep_names:
            target_df = self._invoke_fgen(dep, history_df, target_df, invoked_fgens)

        # 3) ––––– run the generator
        self.logger.info(f"Invoking feature generator: {fgen_name}")
        target_df = info["func"](history_df, target_df, self.config)  # type: ignore[arg-type]

        # 4) ––––– store result columns to cache
        cols = info["num_cols"] + info["cat_cols"]
        if cols:  # a target generator may return an empty list
            self._save(key, target_df.select(cols))

        invoked_fgens.add(fgen_name)
        return target_df

