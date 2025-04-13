# Simplified Recommender System

A streamlined recommender system framework with NLP capabilities for easy experimentation.

## Features

- Simple, focused implementation for quick iteration
- Pretrained NLP models for text features
- Time-based data splitting for realistic evaluation
- Optional hyperparameter tuning
- Minimal dependencies

## Installation

1. Install core dependencies:

```bash
pip install polars catboost optuna scikit-learn pandas numpy
```

2. Install NLP dependencies:

```bash
pip install sentence-transformers
```

## Project Structure

The simplified framework consists of these key components:

- `simplified_data_loader.py` - Handles data loading and time-based splitting
- `simplified_experiment.py` - Manages the experiment lifecycle
- `simplified_hyperparameter_tuner.py` - Provides optional parameter tuning
- `text_processor.py` - Processes text features using pretrained models
- `feature_factory.py` - Your existing feature generation code (unmodified)
- `model_factory.py` - Your existing model factory (unmodified)

## Basic Usage

```python
from lavka_recsys.config import Config
from lavka_recsys.simplified_experiment import SimpleExperiment
from lavka_recsys.text_processor import register_text_embedding_features

# Register text embedding features
register_text_embedding_features()

# Create configuration
config = Config({
    "experiment_name": "simple_recommender",
    "model": {
        "type": "catboost",
        "config": {
            "catboost": {
                "iterations": 500,
                "learning_rate": 0.1,
                "depth": 6
            }
        }
    },
    "features": [
        "count_purchase_user_product",
        "ctr_product",
        "user_stats", 
        "product_stats",
        "store_stats",
        "product_embeddings",
        "category_embeddings"
    ],
    "target": "CartUpdate_Purchase_vs_View",
    "data": {
        "train_path": "data/train.parquet",
        "test_path": "data/test.parquet"
    },
    "text_processing": {
        "model_type": "sentence-transformers",
        "model_name": "all-MiniLM-L6-v2",
        "embedding_dimensions": 20
    }
})

# Create and run experiment
experiment = SimpleExperiment("my_experiment", config)
results = experiment.run()
print(f"Metrics: {results['metrics']}")

# Generate predictions
submission = experiment.predict()
```

## Text Feature Options

The system supports multiple pretrained NLP models:

1. **Sentence Transformers** (recommended):
   - Modern, transformer-based embeddings
   - Configuration: `"model_type": "sentence-transformers"`
   - Good models: `"all-MiniLM-L6-v2"` (small), `"all-mpnet-base-v2"` (larger but better)

2. **Word2Vec** (optional):
   - Classic word embeddings
   - Configuration: `"model_type": "word2vec"`
   - Good models: `"glove-wiki-gigaword-100"`, `"word2vec-google-news-300"`

## Hyperparameter Tuning

To run with hyperparameter tuning:

```python
results = experiment.run_with_tuning()
best_params = results['best_params']
submission = experiment.predict(best_params)
```

## Customization

For custom feature engineering, simply add your feature generators to the `feature_factory.py` file as you've been doing.

## Memory Requirements

- **Base System**: ~2GB RAM
- **With Sentence Transformers**: ~3GB RAM (depends on model size)
- **With Word2Vec**: ~2.5GB RAM

## Troubleshooting

- If you encounter memory issues, try a smaller model or reduce embedding dimensions
- For faster debugging, set `"sample_size": 10000` in your configuration