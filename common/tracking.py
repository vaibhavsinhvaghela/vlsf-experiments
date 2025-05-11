#!/usr/bin/env python3
"""
Tracking system for BBQ and StereoSet evaluation runs.

This module provides functions to track and query evaluation runs for
both BBQ and StereoSet datasets, storing run parameters and results in
JSON files for easy searching and analysis.
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union


# Default paths for tracking files
BBQ_TRACKING_FILE = "tracking/bbq_runs.json"
STEREOSET_TRACKING_FILE = "tracking/stereoset_runs.json"


def ensure_tracking_dir():
    """Ensure the tracking directory exists."""
    tracking_dir = Path("tracking")
    tracking_dir.mkdir(exist_ok=True)
    
    # Create empty tracking files if they don't exist
    for file_path in [BBQ_TRACKING_FILE, STEREOSET_TRACKING_FILE]:
        path = Path(file_path)
        if not path.exists():
            with open(path, 'w') as f:
                json.dump({}, f, indent=2)


def add_run(
    dataset_type: str,
    run_id: str,
    model_name: str,
    model_type: str,
    num_examples: int,
    prompt_strategy: Optional[str] = None,
    categories: Optional[str] = None,
    bias_types: Optional[str] = None,
    split: Optional[str] = None,
    results_dir: str = None,
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add a new run to the tracking file.
    
    Args:
        dataset_type: Either 'bbq' or 'stereoset'
        run_id: Unique identifier for the run
        model_name: Name of the model used
        model_type: Type of model (e.g., 'openai', 'anthropic', 'mock')
        num_examples: Number of examples evaluated
        prompt_strategy: Strategy used for prompting (for BBQ)
        categories: Categories included in the evaluation
        bias_types: Bias types included (for StereoSet)
        split: Dataset split used
        results_dir: Directory containing the results
        metrics: Dictionary of metrics from the evaluation
    """
    # Ensure tracking directory exists
    ensure_tracking_dir()
    
    # Determine which tracking file to use
    tracking_file = BBQ_TRACKING_FILE if dataset_type.lower() == 'bbq' else STEREOSET_TRACKING_FILE
    
    # Read existing tracking data
    try:
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        tracking_data = {}
    
    # Create new run entry
    run_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_name": model_name,
        "model_type": model_type,
        "num_examples": num_examples,
        "results_dir": results_dir,
    }
    
    # Add optional fields if provided
    if prompt_strategy:
        run_entry["prompt_strategy"] = prompt_strategy
    if categories:
        run_entry["categories"] = categories
    if bias_types:
        run_entry["bias_types"] = bias_types
    if split:
        run_entry["split"] = split
    if metrics:
        run_entry["metrics"] = metrics
    
    # Add to tracking data
    tracking_data[run_id] = run_entry
    
    # Write back to file
    with open(tracking_file, 'w') as f:
        json.dump(tracking_data, f, indent=2, sort_keys=True)


def update_run_metrics(dataset_type: str, run_id: str, metrics: Dict[str, Any]) -> None:
    """
    Update metrics for an existing run.
    
    Args:
        dataset_type: Either 'bbq' or 'stereoset'
        run_id: Unique identifier for the run
        metrics: Dictionary of metrics to add or update
    """
    # Determine which tracking file to use
    tracking_file = BBQ_TRACKING_FILE if dataset_type.lower() == 'bbq' else STEREOSET_TRACKING_FILE
    
    # Read existing tracking data
    try:
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        tracking_data = {}
    
    # Update metrics if run exists
    if run_id in tracking_data:
        if "metrics" not in tracking_data[run_id]:
            tracking_data[run_id]["metrics"] = {}
        
        # Update metrics
        tracking_data[run_id]["metrics"].update(metrics)
        
        # Write back to file
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2, sort_keys=True)
    else:
        print(f"Run ID {run_id} not found in tracking file.")


def get_runs(dataset_type: str, **filters) -> Dict[str, Dict[str, Any]]:
    """
    Get runs matching the specified filters.
    
    Args:
        dataset_type: Either 'bbq' or 'stereoset'
        **filters: Key-value pairs to filter runs by
        
    Returns:
        Dictionary of runs matching the filters
    """
    # Determine which tracking file to use
    tracking_file = BBQ_TRACKING_FILE if dataset_type.lower() == 'bbq' else STEREOSET_TRACKING_FILE
    
    # Read tracking data
    try:
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    
    # Apply filters
    if not filters:
        return tracking_data
    
    filtered_runs = {}
    for run_id, run_data in tracking_data.items():
        match = True
        for key, value in filters.items():
            # Handle nested keys (e.g., metrics.accuracy)
            if '.' in key:
                parts = key.split('.')
                current = run_data
                for part in parts:
                    if part not in current:
                        match = False
                        break
                    current = current[part]
                
                if match and current != value:
                    match = False
            # Handle direct keys
            elif key not in run_data or run_data[key] != value:
                match = False
        
        if match:
            filtered_runs[run_id] = run_data
    
    return filtered_runs


def get_latest_runs(dataset_type: str, n: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Get the n most recent runs.
    
    Args:
        dataset_type: Either 'bbq' or 'stereoset'
        n: Number of runs to return
        
    Returns:
        Dictionary of the n most recent runs
    """
    # Get all runs
    all_runs = get_runs(dataset_type)
    
    # Sort by timestamp
    sorted_runs = sorted(
        all_runs.items(),
        key=lambda x: x[1].get("timestamp", ""),
        reverse=True
    )
    
    # Return top n
    return {run_id: run_data for run_id, run_data in sorted_runs[:n]}


def print_run_summary(run_id: str, run_data: Dict[str, Any]) -> None:
    """Print a summary of a run."""
    print(f"Run ID: {run_id}")
    print(f"Timestamp: {run_data.get('timestamp', 'N/A')}")
    print(f"Model: {run_data.get('model_name', 'N/A')} ({run_data.get('model_type', 'N/A')})")
    print(f"Examples: {run_data.get('num_examples', 'N/A')}")
    
    if "prompt_strategy" in run_data:
        print(f"Prompt Strategy: {run_data['prompt_strategy']}")
    
    if "categories" in run_data:
        print(f"Categories: {run_data['categories']}")
    
    if "bias_types" in run_data:
        print(f"Bias Types: {run_data['bias_types']}")
    
    if "results_dir" in run_data:
        print(f"Results Directory: {run_data['results_dir']}")
    
    if "metrics" in run_data:
        print("\nMetrics:")
        for metric, value in run_data["metrics"].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    print()


def main():
    """Command-line interface for querying tracking data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Query BBQ and StereoSet evaluation runs")
    parser.add_argument("dataset", choices=["bbq", "stereoset"], help="Dataset to query")
    parser.add_argument("--latest", type=int, default=0, help="Show N latest runs")
    parser.add_argument("--model", type=str, help="Filter by model name")
    parser.add_argument("--model-type", type=str, help="Filter by model type")
    parser.add_argument("--run-id", type=str, help="Show details for specific run ID")
    
    args = parser.parse_args()
    
    # Show specific run
    if args.run_id:
        runs = get_runs(args.dataset)
        if args.run_id in runs:
            print_run_summary(args.run_id, runs[args.run_id])
        else:
            print(f"Run ID {args.run_id} not found.")
        return
    
    # Show latest runs
    if args.latest > 0:
        runs = get_latest_runs(args.dataset, args.latest)
        for run_id, run_data in runs.items():
            print_run_summary(run_id, run_data)
        return
    
    # Apply filters
    filters = {}
    if args.model:
        filters["model_name"] = args.model
    if args.model_type:
        filters["model_type"] = args.model_type
    
    runs = get_runs(args.dataset, **filters)
    
    # Print summary
    print(f"Found {len(runs)} {args.dataset.upper()} runs matching filters")
    
    if runs:
        print("\nRun IDs:")
        for run_id in runs.keys():
            print(f"  {run_id}")
    

if __name__ == "__main__":
    main()
