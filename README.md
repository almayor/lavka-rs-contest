# Lavka Recommender System

A flexible and maintainable recommender system framework with time-aware training, feature caching, and configurable experiments.

## Features

- Unified Trainer for all model training strategies
- Configurable experiment types through a clean interface
- Time-based data splitting with multiple strategies
- Feature caching with transparent integration
- Consistent directory structure for all outputs
- Hyperparameter tuning with Optuna integration
- Text processing for NLP features
- Rich set of behavioral and contextual features
- Support for ranking models with CatBoostRanker
- GPU acceleration for model training
- Efficient feature selection with caching
- Conversion rate modeling with specialized features

## Project Structure

```
lavka_recsys/
├── __init__.py
├── cached_feature_factory.py  # Caching wrapper for FeatureFactory
├── config.py                  # Configuration management 
├── custom_logging.py          # Logging utilities
├── data_loader.py             # Data loading and splitting
├── trainer.py                 # Unified trainer for all training strategies
├── enhanced_training.py       # Legacy compatibility module
├── experiment.py              # Unified experiment framework
├── feature_factory.py         # Feature generation interface
├── feature_selector.py        # Feature selection
├── feature_generators/        # Feature generator implementations
│   ├── __init__.py
│   ├── collaborative_filtering.py  # Collaborative filtering features
│   ├── text_processor.py      # Text embedding features 
│   └── common.py              # Common feature generators
├── hyperparameter_tuner.py    # Hyperparameter optimization
├── metrics.py                 # Evaluation metrics
├── model_factory.py           # Model creation and training
├── time_splitter.py           # Time-based data splitting
└── visualizer.py              # Results visualization

results/                       # Centralized output directory
├── feature_cache/             # Cached computed features
├── metrics/                   # Experiment metrics and results
├── model_cache/               # Trained model cache
└── visualizations/            # Plots and visualizations
```

## Output Organization

All outputs are stored in a consistent directory structure:
- `results/` - Root directory for all outputs
- `results/model_cache/` - Trained models
- `results/feature_cache/` - Cached computed features
- `results/metrics/` - Experiment metrics and results
- `results/visualizations/` - Plots and visualizations

## Configuration

The system uses YAML configuration files with the following key sections:

```yaml
# Experiment configuration
experiment:
  type: "single_run"    # "single_run" or "tuning"
  use_hyperparameter_tuning: false
  evaluation:
    perform_kaggle_simulation: true
    create_submission: true

# Feature selection configuration
feature_selection:
  enabled: false        # Enable/disable feature selection
  method: "importance"  # Feature selection method
  n_features: 10        # Number of top features to select

# Training configuration
training:
  split_type: "expanding_window"  # "standard", "fixed_window", or "expanding_window"
  history_days: 30     # Only used for fixed_window splits
  target_days: 7       # Target window is 7 days
  step_days: 7         # Move window 7 days at a time
  max_splits: 10       # Maximum number of splits
  validation_days: 7   # Days for validation after target

# Model configuration
model:
  type: "catboost"       # Model type: catboost, catboost_ranker, lightgbm, etc.
  use_gpu: false         # Enable/disable GPU acceleration
  gpu_devices: "0"       # GPU device IDs (comma-separated string for multi-GPU)
  thread_count: -1       # Number of CPU threads (-1 means auto)
  config:
    catboost:            # Model-specific parameters
      iterations: 500
      learning_rate: 0.1
      depth: 6
      loss_function: "Logloss"
      
# Features to use
features:
  - "count_purchase_user_product"
  - "user_stats"
  - "product_stats"
  - "cart_to_purchase_rate"    # Cart-to-Purchase conversion rate
  - "purchase_view_ratio"      # Purchase-to-View ratio 
  # ... other features

# Target definition
target: "CartUpdate_Purchase_vs_View"
```

## Usage Examples

### Basic Usage

```python
from lavka_recsys.config import Config
from lavka_recsys.experiment import Experiment

# Load configuration from YAML file
config = Config()
config.load("my_config.yaml")

# Or create configuration programmatically
config = Config({
    "experiment": {"type": "single_run"},
    "training": {"split_type": "standard"},  # Single split
    "features": ["count_purchase_user_product", "user_stats"],
    "target": "CartUpdate_Purchase_vs_View"
})

# Create experiment
experiment = Experiment("basic_experiment", config)

# Setup (load data, initialize components)
experiment.setup()

# Run experiment (trains model based on experiment type in config)
results = experiment.run()

# Evaluate on test data and/or create submission
evaluation = experiment.evaluate()

print(f"Metrics: {results['metrics']}")
```

### Different Experiment Types

You can specify the experiment type in the configuration file:

```yaml
# For standard single run experiment
experiment:
  type: "single_run"
  
# For experiment with hyperparameter tuning
experiment:
  type: "tuning"
```

Or override it in code:

```python
from lavka_recsys.experiment import Experiment

# Create tuning experiment
config.set('experiment.type', 'tuning')
experiment = Experiment("my_experiment", config)
results = experiment.run()
```

### Time Splitting Strategies

The framework supports different time splitting strategies that can be used with any experiment type.

#### Strategy Comparison

1. **Standard Split**
   - Creates a single time-based train/validation split
   - Uses the most recent data for validation
   - Fastest training approach

   ```
   Timeline: [Early] -------------------- [Recent]
   
   Single Split: [History...............][Target]
   ```

2. **Fixed Window**
   - Uses a constant-size history window (e.g., always 30 days)
   - Window moves with the target period
   - Each split has the same amount of training data
   
   ```
   Timeline: [Early] -------------------- [Recent]
   
   Fixed Window (e.g., 30 days history):
   Split 1:            [History-30d][Target 1]
   Split 2:       [History-30d][Target 2]
   Split 3:  [History-30d][Target 3]
   ```

3. **Expanding Window**
   - Always starts from the earliest data point
   - History window grows larger as you move forward in time
   - Later splits include all earlier data plus more recent data
   
   ```
   Timeline: [Early] -------------------- [Recent]
   
   Expanding Window:
   Split 1:  [History................][Target 1]
   Split 2:  [History.........][Target 2]
   Split 3:  [History..][Target 3]
   ```

#### Configuration

```yaml
# Choose your time splitting strategy:

# Standard: Single train/validation split (fastest option)
training:
  split_type: "standard"
  validation_days: 7  # Optional validation period
  
# Fixed window: Use a fixed history window size
training:
  split_type: "fixed_window"
  history_days: 30  # Required for fixed window
  target_days: 7
  step_days: 7
  max_splits: 10
  
# Expanding window: Start from the earliest data, grow history window
training:
  split_type: "expanding_window"
  target_days: 7
  step_days: 7
  max_splits: 10
```

### Hyperparameter Tuning

Enable hyperparameter tuning:

```yaml
experiment:
  type: "tuning"

# Optuna hyperparameter tuning settings
hyperparameter_tuning:
  n_trials: 10
  n_jobs: 1
  timeout: 1800  # 30 minutes
  param_grid:
    catboost:
      # Explicit parameter definition format with type and properties
      
      # Float parameter with log scale (good for learning rates)
      learning_rate:
        type: "float"         # Parameter type: float, int, categorical
        range: [0.01, 0.3]    # Min and max values
        log_scale: true       # Use logarithmic scale for sampling
      
      # Integer parameter (linear scale)
      depth:
        type: "int"
        range: [4, 10]
      
      # Categorical parameter with specific values
      iterations:
        type: "categorical"   # Choose from specific values
        values: [300, 500, 800]
      
      # Float parameter with log scale
      l2_leaf_reg:
        type: "float"
        range: [1.0, 10.0]
        log_scale: true
```

The tuning process will use the same split type specified in the `training` section.

### Direct API Example

You can also use the main components directly:

```python
from lavka_recsys.config import Config
from lavka_recsys.data_loader import DataLoader
from lavka_recsys.cached_feature_factory import CachedFeatureFactory
from lavka_recsys.model_factory import ModelFactory
from lavka_recsys.trainer import Trainer
from lavka_recsys.time_splitter import SplitType

# Create components
config = Config("my_config.yaml")
data_loader = DataLoader(config)
feature_factory = CachedFeatureFactory(config=config)
model_factory = ModelFactory(config)

# Create trainer and train a model
trainer = Trainer(config, data_loader, feature_factory, model_factory)
model = trainer.train(split_type=SplitType.EXPANDING_WINDOW)

# Evaluate the model
score, metrics = trainer.evaluate_model(model)
print(f"Model score: {score}, Metrics: {metrics}")
```

### Kaggle Evaluation and Submission

```python
# Run the experiment
results = experiment.run()

# Evaluate on test data and create submission
evaluation = experiment.evaluate()

# Or just create submission without evaluation
submission_df = experiment.create_kaggle_submission()
```

### Using Ranking Models

The system supports learning-to-rank models through CatBoostRanker integration:

```python
from lavka_recsys.config import Config
from lavka_recsys.experiment import Experiment

# Create configuration
config = Config()

# Set model type to catboost_ranker
config.set('model.type', 'catboost_ranker')

# Create and run experiment
experiment = Experiment("ranking_experiment", config)
experiment.setup()
results = experiment.run()
```

#### Key Features of Ranking Models

1. **Group-based Evaluation**: Ranking models automatically use request_id or group_id columns for grouping examples during training and evaluation.

2. **Ranking-specific Loss Functions**: 
   - YetiRank: Optimizes NDCG directly
   - YetiRankPairwise: Pairwise version of YetiRank
   - PairLogit: Pairwise logistic loss
   - QueryRMSE: Pointwise ranking with RMSE loss
   - QuerySoftMax: Listwise ranking loss

3. **Ranking Metrics**:
   - NDCG (Normalized Discounted Cumulative Gain)
   - PrecisionAt:top=N
   - RecallAt:top=N

4. **When to Use Ranking Models**:
   - When predicting the relative ordering of items is more important than their absolute scores
   - For recommendation tasks where you need to present a ranked list to users
   - When you have groups of items (e.g., products for each user request)

## Feature Generation

To add custom features, register them in the feature factory:

```python
from lavka_recsys.feature_factory import FeatureFactory

@FeatureFactory.register('my_custom_feature')
def generate_my_feature(history_df, target_df):
    # Generate your feature
    feature_df = ...
    return feature_df
```

## Installation

1. Install dependencies:

```bash
pip install polars pandas numpy catboost scikit-learn optuna matplotlib tqdm
```

2. For NLP features:

```bash
pip install sentence-transformers
```

## Architectural Design

### Key Components

1. **Trainer** (`trainer.py`)
   - Central component for all model training strategies
   - Handles data splitting through TimeSplitter
   - Manages feature generation through CachedFeatureFactory
   - Provides model evaluation
   - Supports model caching

2. **CachedFeatureFactory** (`cached_feature_factory.py`)
   - Wraps FeatureFactory to provide transparent caching
   - Uses composition pattern rather than inheritance
   - Avoids redundant feature generation
   - Delegates to underlying FeatureFactory for actual feature generation

3. **Experiment** (`experiment.py`)
   - Coordinates the overall experiment workflow
   - Uses simplified ExperimentTypes (single_run, tuning)
   - Delegates to Trainer for model training and evaluation

4. **HyperparameterTuner** (`hyperparameter_tuner.py`)
   - Uses Optuna for hyperparameter optimization
   - Gets parameter ranges from config
   - Creates temporary Trainer instances for each trial
   - Supports all time splitting strategies

5. **TimeSplitter** (`time_splitter.py`)
   - Provides three splitting strategies: standard, fixed_window, expanding_window
   - Used by Trainer to create appropriate data splits
   - Separates time splitting logic from training logic

6. **ModelFactory** (`model_factory.py`)
   - Creates model instances based on configuration
   - Supports multiple model types:
     - CatBoostModel: Standard classification model
     - CatBoostRanker: Learning-to-rank model for ordered recommendations
     - LightGBMModel: LightGBM implementation
   - Handles all model-specific configurations and parameters
   - Supports GPU acceleration with configurable parameters:
     - Automatic GPU device selection
     - Multi-GPU training for supported models
     - Optimized thread count for CPU operations

## Troubleshooting

- **Memory issues**: Use feature caching to reduce memory usage
- **Speed issues**: Try smaller models, fewer features, or reduce splits 
- **Debugging**: Use the `logging` section in config to control logging levels

## Available Features and Generated Columns

The system includes a rich set of feature generators. Here's a description of the available features and the columns they generate:

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

### Advanced Features

| Feature Name | Description | Generated Columns |
|--------------|-------------|-------------------|
| `cross_features` | Interaction between entity statistics | `user_product_purchase_cross` |

For the full list of features, see the feature generators in the `feature_generators/` directory.

## Advanced Topics

- **Feature Caching**: Features are automatically cached to speed up development
- **Model Caching**: Trained models are cached to avoid retraining
- **Visualization**: Use the `visualizer.py` module to create plots of results
- **Feature Selection**: Enable automatic feature selection to improve model performance 
- **Custom Features**: Extend the system by adding your own feature generators