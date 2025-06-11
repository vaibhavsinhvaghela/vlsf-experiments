"""
Common Pipeline Utilities
------------------------
Shared utilities for running evaluation pipelines across different datasets.
"""

import os
import sys
import datetime
from pathlib import Path

def generate_run_id(dataset_name, model_name, dataset_type=None):
    """Generate a unique run ID based on timestamp, dataset, model name and dataset type"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_name.replace("/", "_").replace("-", "_").lower()
    
    # Simple concatenation with underscore separators
    components = [dataset_name, timestamp]
    
    # Add dataset_type if provided
    if dataset_type:
        components.append(dataset_type)
    
    # Add model slug
    components.append(model_slug)
    
    # Join all components with underscores
    return "_".join(components)

def setup_directories(base_dir, dataset_name, run_id):
    """Create and return paths to the organized directory structure"""
    run_dir = Path(base_dir) / dataset_name / run_id
    
    # Create directory structure
    dataset_dir = run_dir / "dataset"
    predictions_dir = run_dir / "predictions"
    analysis_dir = run_dir / "analysis"
    
    for directory in [dataset_dir, predictions_dir, analysis_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    paths = {
        "run_dir": run_dir,
        "dataset_path": dataset_dir / f"{dataset_name}_samples.csv",
        "predictions_path": predictions_dir / "model_predictions.csv",
        "analysis_dir": analysis_dir
    }
    
    return paths

def write_summary_file(summary_path, title, run_id, timestamp, config, paths):
    """Write a standardized summary file for a pipeline run"""
    with open(summary_path, "w") as f:
        f.write(f"{title} Summary\n")
        f.write(f"{'=' * len(title)}{'=' * 8}\n\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Configuration:\n")
        
        # Write all configuration parameters
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        
        f.write(f"\nResults Location:\n")
        f.write(f"  Dataset: {paths['dataset_path']}\n")
        f.write(f"  Predictions: {paths['predictions_path']}\n")
        f.write(f"  Analysis: {paths['analysis_dir']}\n")
