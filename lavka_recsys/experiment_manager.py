import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any

from .config import Config
from .custom_logging import get_logger
from .experiment import Experiment
from .feature_selector import FeatureSelector

class ExperimentManager:
    """
    Manages the workflow between feature selection and experiment running.
    
    This class provides high-level workflows that combine feature selection
    and experiment running with appropriate coordination.
    """
    
    def __init__(self, config: Config):
        """
        Initialize with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.feature_selection_results = {}
        
    def run_with_feature_selection(self, experiment_name: str) -> Dict:
        """
        Run a complete workflow with feature selection followed by experiment.
        
        Args:
            experiment_name: Name for the experiment
            
        Returns:
            Dict: Combined results from feature selection and experiment
        """
        # Create results directory
        results_dir = self.config.get('output.results_dir', 'results')
        experiment_dir = os.path.join(results_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Run feature selection if enabled
        if self.config.get('feature_selection.enabled', False):
            model_type = self.config.get('model.type')
            self.logger.info(f"Running feature selection for model type: {model_type}")
            
            # Create and run feature selector
            selector = FeatureSelector(f"{experiment_name}_selector", self.config, model_type)
            selection_results = selector.run()
            self.feature_selection_results[model_type] = selection_results
            
            # Update config with selected features
            selected_features = selection_results.get('selected_features', [])
            if selected_features:
                self.logger.info(f"Selected {len(selected_features)} features for {model_type}")
                
                # Update config with selected features
                # This is a bit of a hack since we're not really supposed to modify 
                # the original config, but it's the cleanest way to pass the selection
                original_features = self.config.get('features', [])
                pruned_features = [f for f in original_features if f in selected_features]
                self.config.set('features', pruned_features)
        
        # Run main experiment
        self.logger.info(f"Running main experiment: {experiment_name}")
        experiment = Experiment(experiment_name, self.config)
        experiment.setup()
        experiment_results = experiment.run()
        
        # Run evaluation if enabled
        evaluation_results = {}
        if self.config.get('experiment.evaluation.perform_kaggle_simulation', False) or \
           self.config.get('experiment.evaluation.create_submission', False):
            evaluation_results = experiment.evaluate()
        
        # Combine results
        combined_results = {
            'experiment_results': experiment_results,
            'feature_selection': self.feature_selection_results,
            'evaluation': evaluation_results
        }
        
        # Save combined results
        results_path = os.path.join(experiment_dir, "combined_results.json")
        try:
            with open(results_path, 'w') as f:
                # Convert to JSON-serializable format
                serializable_results = self._convert_to_serializable(combined_results)
                json.dump(serializable_results, f, indent=4)
            self.logger.info(f"Saved combined results to: {results_path}")
        except Exception as e:
            self.logger.error(f"Error saving combined results: {str(e)}")
        
        return combined_results
    
    def run_classifier_ranker_experiment(self, experiment_name: str) -> Dict:
        """
        Run a full experiment with separate feature selection for classifier and ranker.
        
        This method runs two separate feature selection processes and experiments:
        1. A classifier experiment with features selected for classification
        2. A ranker experiment with features selected for ranking
        
        Args:
            experiment_name: Base name for the experiments
            
        Returns:
            Dict: Combined results from both experiments
        """
        # Save original model type
        original_model_type = self.config.get('model.type')
        
        # Create results directory
        results_dir = self.config.get('output.results_dir', 'results')
        experiment_dir = os.path.join(results_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 1. Run classifier experiment
        classifier_config = self.config.copy()
        classifier_config.set('model.type', 'catboost')  # Ensure classifier model type
        
        classifier_manager = ExperimentManager(classifier_config)
        classifier_results = classifier_manager.run_with_feature_selection(f"{experiment_name}_classifier")
        
        # 2. Run ranker experiment
        ranker_config = self.config.copy()
        ranker_config.set('model.type', 'catboost_ranker')  # Set ranker model type
        
        ranker_manager = ExperimentManager(ranker_config)
        ranker_results = ranker_manager.run_with_feature_selection(f"{experiment_name}_ranker")
        
        # Restore original model type
        self.config.set('model.type', original_model_type)
        
        # Combine results
        combined_results = {
            'classifier': classifier_results,
            'ranker': ranker_results,
            'experiment_name': experiment_name,
            'timestamp': self._get_timestamp()
        }
        
        # Save combined results
        results_path = os.path.join(experiment_dir, "dual_model_results.json")
        try:
            with open(results_path, 'w') as f:
                # Convert to JSON-serializable format
                serializable_results = self._convert_to_serializable(combined_results)
                json.dump(serializable_results, f, indent=4)
            self.logger.info(f"Saved dual model results to: {results_path}")
        except Exception as e:
            self.logger.error(f"Error saving dual model results: {str(e)}")
        
        return combined_results
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert an object to a JSON-serializable format"""
        import numpy as np
        import polars as pl
        from datetime import datetime
        
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pl.DataFrame):
            return "DataFrame (not serialized)"
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()