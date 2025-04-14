#!/usr/bin/env python3
"""
Example script demonstrating the refactored experiment code.

This script shows how to use the new architecture with:
1. Single experiment with model-specific feature selection
2. Static hyperparameter tuning with visualizations
3. ExperimentManager for coordinated workflows
"""

import os
import sys
from lavka_recsys.config import Config
from lavka_recsys.experiment import Experiment
from lavka_recsys.experiment_manager import ExperimentManager

def run_single_experiment():
    """Run a single experiment with the refactored code"""
    print("\n=== Running Single Experiment ===\n")
    
    # Load configuration
    config = Config.from_file('default_config.yaml')
    
    # Create and run experiment
    experiment = Experiment("single_experiment", config)
    experiment.setup()
    results = experiment.run()
    
    print(f"Experiment results: {results.get('metrics', {})}")
    
    return results

def run_experiment_with_feature_selection():
    """Run an experiment with feature selection using ExperimentManager"""
    print("\n=== Running Experiment with Feature Selection ===\n")
    
    # Load configuration and enable feature selection
    config = Config.from_file('default_config.yaml')
    config.set('feature_selection.enabled', True)
    config.set('feature_selection.method', 'importance')
    config.set('feature_selection.n_features', 20)  # Select top 20 features
    
    # Use ExperimentManager to coordinate feature selection and experiment
    manager = ExperimentManager(config)
    results = manager.run_with_feature_selection("feature_selection_experiment")
    
    # Get experiment results
    experiment_results = results.get('experiment_results', {})
    print(f"Experiment metrics: {experiment_results.get('metrics', {})}")
    
    # Get feature selection results
    selection_results = results.get('feature_selection', {})
    for model_type, model_selection in selection_results.items():
        selected_features = model_selection.get('selected_features', [])
        print(f"\nSelected features for {model_type}: {len(selected_features)}")
        for i, feature in enumerate(selected_features[:10]):
            print(f"  {i+1}. {feature}")
    
    return results

def run_classifier_ranker_experiment():
    """Run experiment with separate feature selection for classifier and ranker"""
    print("\n=== Running Classifier-Ranker Experiment ===\n")
    
    # Load configuration and enable feature selection
    config = Config.from_file('default_config.yaml')
    config.set('feature_selection.enabled', True)
    config.set('feature_selection.method', 'importance')
    config.set('feature_selection.n_features', 20)  # Select top 20 features
    
    # Use ExperimentManager to run the dual model experiment
    manager = ExperimentManager(config)
    results = manager.run_classifier_ranker_experiment("dual_model_experiment")
    
    # Print classifier results
    classifier_results = results.get('classifier', {}).get('experiment_results', {})
    print(f"Classifier metrics: {classifier_results.get('metrics', {})}")
    
    # Print ranker results
    ranker_results = results.get('ranker', {}).get('experiment_results', {})
    print(f"Ranker metrics: {ranker_results.get('metrics', {})}")
    
    return results

def run_hyperparameter_tuning():
    """Run hyperparameter tuning with the static method"""
    print("\n=== Running Hyperparameter Tuning ===\n")
    
    # Load configuration
    config = Config.from_file('default_config.yaml')
    
    # Define parameter spaces for tuning
    parameter_spaces = {
        'model.config.catboost.learning_rate': {
            'type': 'float',
            'range': [0.01, 0.3],
            'log_scale': True
        },
        'model.config.catboost.depth': {
            'type': 'int',
            'range': [4, 8]
        },
        'model.config.catboost.l2_leaf_reg': {
            'type': 'float',
            'range': [1.0, 10.0],
            'log_scale': True
        }
    }
    
    # Run tuning with visualization generation
    best_experiment, best_config, study, visualizations = Experiment.tune_configurations(
        config, parameter_spaces, n_trials=5,  # Use fewer trials for demo
        create_visualizations=True
    )
    
    # Print best parameters
    print(f"Best parameters: {study.best_params}")
    print(f"Best value: {study.best_value:.4f}")
    
    # Print paths to visualizations
    print("\nVisualization paths:")
    for name, path in visualizations.items():
        print(f"  {name}: {path}")
    
    return best_experiment, best_config, study, visualizations

def main():
    """Main function to run the examples"""
    if len(sys.argv) > 1:
        example = sys.argv[1]
    else:
        print("Available examples:")
        print("  1. single - Run a single experiment")
        print("  2. feature_selection - Run experiment with feature selection")
        print("  3. dual_model - Run classifier and ranker with separate feature selection")
        print("  4. tuning - Run hyperparameter tuning with visualizations")
        print("  all - Run all examples")
        
        example = input("\nEnter example number or 'all': ")
    
    if example in ['1', 'single']:
        run_single_experiment()
    elif example in ['2', 'feature_selection']:
        run_experiment_with_feature_selection()
    elif example in ['3', 'dual_model']:
        run_classifier_ranker_experiment()
    elif example in ['4', 'tuning']:
        run_hyperparameter_tuning()
    elif example in ['all']:
        run_single_experiment()
        run_experiment_with_feature_selection()
        run_classifier_ranker_experiment()
        run_hyperparameter_tuning()
    else:
        print(f"Unknown example: {example}")

if __name__ == "__main__":
    main()