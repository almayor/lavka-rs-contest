# Lavka Recommender System

A flexible and maintainable recommender system framework with configurable experiments and feature generation capabilities.

## Features

- Streamlined experiment framework with a clean interface
- Time-based data splitting for realistic evaluation
- Feature caching with transparent integration
- Consistent directory structure for all outputs
- Advanced text processing features:
  - Weighted user-product similarity based on purchase history
  - Semantic product clustering and user preferences tracking
  - Text diversity and novelty metrics for recommendations
- Rich set of behavioral and contextual features
- Support for ranking models with CatBoostRanker
- Efficient feature generation with dependency tracking

## Project Structure

```
lavka_recsys/
├── __init__.py
├── config.py                  # Configuration management 
├── custom_logging.py          # Logging utilities
├── data_loader.py             # Data loading and splitting
├── experiment.py              # Unified experiment framework
├── feature_factory.py         # Feature generation with caching
├── feature_generators/        # Feature generator implementations
│   ├── __init__.py
│   ├── collaborative_filtering.py  # Collaborative filtering features
│   ├── text_processor.py      # Text embedding features 
│   └── common.py              # Common feature generators
├── metrics.py                 # Evaluation metrics
├── model_factory.py           # Model creation and training
└── visualizer.py              # Results visualization

results/                       # Centralized output directory
├── feature_cache/             # Cached computed features
├── metrics/                   # Experiment metrics and results
├── model_cache/               # Trained model cache
└── submissions/               # Generated submission files
```

## Output Organization

All outputs are stored in a consistent directory structure:

- `results/` - Root directory for all outputs
- `results/model_cache/` - Trained models
- `results/feature_cache/` - Cached computed features
- `results/metrics/` - Experiment metrics and results
- `results/submissions/` - Submission files for evaluation

## Configuration

The system uses YAML configuration files with the following key sections:

```yaml
# Data configuration
data:
  train_path: "data/train.parquet"
  test_path: "data/test.parquet"
  target_days: 30
  holdout:
    enabled: true
    holdout_days: 30

# Model configuration
model:
  type: "catboost_classifier"  # Model type: catboost_classifier or catboost_ranker
  use_gpu: false
  gpu_devices: "0"
  thread_count: -1
  config:
    catboost_classifier:
      iterations: 500
      learning_rate: 0.1
      depth: 6
      loss_function: "Logloss"
      
# Features to use
features:
  - "count_purchase_user_product"
  - "user_stats"
  - "product_stats"
  - "cart_to_purchase_rate"
  - "purchase_view_ratio"
  # ... other features

# Target definition
target: "CartUpdate_Purchase_vs_View"

# Output settings
output:
  results_dir: "results"
  model_cache_dir: "results/model_cache"
  feature_cache_dir: "results/feature_cache"
  submissions_dir: "results/submissions"
```

## Usage Examples

### Basic Usage

```python
from lavka_recsys.config import Config
from lavka_recsys.experiment import Experiment

# Load configuration from YAML file
config = Config.load("my_config.yaml")

# Update an existing configuration (immutable)
config = (config
  .set('output.results_dir', "new_results")
  .set('model.type', 'catboost_ranker')
)

# Or create configuration programmatically
config = Config({
    "data": {"train_path": "data/train.parquet", "test_path": "data/test.parquet"},
    "features": ["count_purchase_user_product", "user_stats"],
    "target": "CartUpdate_Purchase_vs_View",
    ...
    # model training parameters
    # time split parameters
    # data cleaning parameters
})

# Create experiment
experiment = Experiment("basic_experiment", config)

# Setup (load data, initialize components)
experiment.setup()

# Run experiment (trains model)
results = experiment.run()

# Create submission for evaluation
submission = experiment.create_submission()

print(f"Metrics: {results['metrics']}")
```

## Feature Generation

To add custom features, register a feature generator in the feature factory:

```python
from lavka_recsys import FeatureFactory

@FeatureFactory.register('my_custom_feature_generator', 
                         num_cols=['feature1', 'feature2'], 
                         cat_cols=['cat_feature'], 
                         depends_on=["another_feature"])
def generate_my_feature_generator(
    history_df: pl.DataFrame,
    feature_df: pl.DataFrame
) -> pl.DataFrame:
      # Add new columns to feature_df using history_df
      feature_df = ...
      return feature_df
```

Then you can switch on this feature generator by adding its name to `config.get('feature_generators')`.
Similarly, to add a new target alternative, register them as:

```python
from lavka_recsys import FeatureFactory

@FeatureFactory.register_target('Custom_Target')
def target_cart_update_purchase(
  history_df: pl.DataFrame,
  target_df: pl.DataFrame
) -> pl.Series:
    # The new target must contain a float for each row in `target_df`
    my_target = ...
    return my_target  
```

And then use it by setting `config.set('target', "Custom_Target")`.


## Installation

1. Install dependencies:

```bash
pip install polars pandas numpy catboost scikit-learn matplotlib tqdm
```

2. For NLP features:

```bash
pip install sentence-transformers
```

## Architectural Design

### Key Components

1. **Experiment** (`experiment.py`)
   - Coordinates the overall experiment workflow
   - Handles data loading, feature generation, and model training
   - Provides evaluation and submission generation

2. **CachedFeatureFactory** (`feature_factory.py`)
   - Wraps FeatureFactory to provide transparent caching
   - Uses composition pattern rather than inheritance
   - Avoids redundant feature generation
   - Delegates to underlying FeatureFactory for actual feature generation

3. **FeatureFactory** (`feature_factory.py`)
   - Registry-based feature generation system
   - Handles feature dependencies automatically
   - Supports both numerical and categorical features
   - Provides extensible architecture for adding new features

4. **DataLoader** (`data_loader.py`)
   - Loads and preprocesses train and test data
   - Provides time-based splitting functionality
   - Supports holdout datasets for validation

5. **ModelFactory** (`models/model_factory.py`)
   - Creates model instances based on configuration
   - Supports multiple model types
   - Handles all model-specific configurations and parameters

## Available Features and Generated Columns

The system includes a rich set of feature generators. Here's a description of a few available features and the columns they generate:

### Basic Features

| Feature Name | Description | Generated Columns |
|--------------|-------------|-------------------|
| `count_purchase_user_product` | Counts purchases by user-product pairs | `count_purchase_u_p` |
| `count_purchase_user_store` | Counts purchases by user-store pairs | `count_purchase_u_s` |
| `ctr_product` | Click-through rate for products | `ctr_product` |
| `cart_to_purchase_rate` | Cart-to-Purchase conversion rate | `cart_to_purchase_rate` |
| `purchase_view_ratio` | Purchase-to-View ratio for products | `purchase_view_ratio` |
| `recency_user_product` | Days since last user-product interaction | `days_since_interaction_u_p` |
| `recency_user_store` | Days since last user-store interaction | `days_since_interaction_u_s` |

### Entity Statistics

| Feature Name | Description | Generated Columns |
|--------------|-------------|-------------------|
| `user_stats` | User-level statistics | `user_total_interactions`, `user_total_purchases`, `user_total_views`, `user_unique_products` |
| `product_stats` | Product-level statistics | `product_total_interactions`, `product_total_purchases`, `product_total_views`, `product_unique_users` |
| `store_stats` | Store-level statistics | `store_total_interactions`, `store_total_purchases`, `store_total_views`, `store_unique_products` |
| `city_stats` | City-level statistics | `city_total_interactions`, `city_total_purchases`, `city_total_views`, `city_unique_stores` |

### Temporal Features

| Feature Name | Description | Generated Columns |
|--------------|-------------|-------------------|
| `time_features` | Basic time-related features | `hour_of_day`, `day_of_week`, `month`, `is_weekend` |
| `product_temporal_patterns` | When products are typically purchased | `avg_purchase_hour`, `most_common_purchase_day` |

### Collaborative Filtering Features

| Feature Name | Description | Generated Columns |
|--------------|-------------|-------------------|
| `memory-based-cf` | Memory-based collaborative filtering using Jaccard similarity | `cf_score` |
| `npmi-cf` | Normalized pointwise mutual information collaborative filtering | `npmi_cf_score` |
| `svd-cf` | SVD matrix factorization for collaborative filtering | `svd_cf_score` |
| `puresvd-cf` | PureSVD matrix factorization with binary interaction matrix | `puresvd_cf_score` |

### Text Embedding Features

| Feature Name | Description | Generated Columns |
|--------------|-------------|-------------------|
| `product_embeddings` | Text embeddings for product names | `product_embed_0` through `product_embed_N` |
| `category_embeddings` | Text embeddings for product categories | `category_embed_0` through `category_embed_N` |
| `user_product_distance` | Weighted similarity between target products and user's purchase/cart history | `purchase_weighted_similarity`, `cart_weighted_similarity`, `min_purchase_similarity`, `min_cart_similarity` |
| `text_similarity_cluster` | Clusters products by text similarity and tracks user preferences | `cluster`, `cluster_purchase_ratio`, `cluster_cart_ratio` |
| `text_diversity_features` | Measures novelty of products relative to user's typical purchases | `distance_from_centroid`, `relative_diversity` |

## Troubleshooting

- **Memory issues**: Use feature caching to reduce memory usage
- **Speed issues**: Try fewer features or simplify model configurations
- **Debugging**: Use the `logging` section in config to control logging levels