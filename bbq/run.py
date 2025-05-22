#!/usr/bin/env python3
"""
BBQ Pipeline Runner
------------------
This script provides a unified interface to run the complete BBQ evaluation pipeline:
1. Prepare the BBQ dataset
2. Evaluate models on the dataset
3. Analyze the results

Usage:
    python -m bbq.run --model_name mistralai/Mistral-7B-Instruct-v0.1 --model_type together
"""

import os
import sys
import argparse
import datetime
import json
import backoff
from pathlib import Path

# Import BBQ modules
from bbq.prepare_bbq_dataset import prepare_bbq_dataset
from bbq.evaluate_model_on_bbq import evaluate_bbq_dataset
from bbq.analyze_bbq_results import analyze_bbq_results

# Import common utilities
from common.pipeline_utils import generate_run_id, setup_directories, write_summary_file

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the complete BBQ evaluation pipeline")
    
    # Dataset preparation arguments
    parser.add_argument("--num_examples", type=int, default=100,
                       help="Number of examples to sample from the dataset")
    parser.add_argument("--categories", type=str, default=None,
                       help="Comma-separated list of categories to include (default: all)")
    parser.add_argument("--split", type=str, default="all",
                       help="Dataset split to use: 'train', 'validation', 'test', or 'all'")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    
    # Model evaluation arguments
    parser.add_argument("--model_type", type=str, default="together",
                       help="Model type: 'together', 'openai', 'anthropic', 'gemini', or 'mock'")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1",
                       help="Name of the model to evaluate")
    parser.add_argument("--prompt_strategy", type=str, default="baseline",
                       help="Prompt strategy to use: 'baseline', 'cot', etc.")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between API calls in seconds")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to evaluate (default: all)")
    parser.add_argument("--max_retries", type=int, default=5,
                       help="Maximum number of retries for API calls")
    parser.add_argument("--base_delay", type=float, default=2.0,
                       help="Base delay for exponential backoff in seconds")
    
    # Output organization
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Base directory for all results")
    parser.add_argument("--run_id", type=str, default=None,
                       help="Custom run ID (default: auto-generated based on timestamp and model)")
    parser.add_argument("--skip_prepare", action="store_true",
                       help="Skip dataset preparation step")
    parser.add_argument("--skip_evaluate", action="store_true",
                       help="Skip model evaluation step")
    parser.add_argument("--skip_analyze", action="store_true",
                       help="Skip results analysis step")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint file to resume from")
    parser.add_argument("--save_checkpoint", action="store_true",
                       help="Save checkpoint after each step")
    
    return parser.parse_args()



def save_checkpoint(checkpoint_path, completed_steps, paths, args):
    """
    Save checkpoint information to a JSON file.
    """
    checkpoint_data = {
        "run_id": args.run_id,
        "completed_steps": completed_steps,
        "paths": {k: str(v) for k, v in paths.items()},
        "args": vars(args)
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"Checkpoint saved to: {checkpoint_path}")

def load_checkpoint(checkpoint_path):
    """
    Load checkpoint information from a JSON file.
    """
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        print(f"Loaded checkpoint from: {checkpoint_path}")
        return checkpoint_data
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    on_backoff=lambda details: print(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries")
)
def run_prepare_step(paths, args):
    """Run the preparation step with backoff for retries"""
    categories = args.categories.split(",") if args.categories else None
    prepare_bbq_dataset(
        output_path=str(paths["dataset_path"]),
        num_examples=args.num_examples,
        categories=categories,
        split=args.split,
        seed=args.seed
    )
    return True

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    on_backoff=lambda details: print(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries")
)
def run_evaluate_step(paths, args):
    """Run the evaluation step with backoff for retries"""
    evaluate_bbq_dataset(
        str(paths["dataset_path"]),
        str(paths["predictions_path"]),
        args.model_type,
        args.model_name,
        args.prompt_strategy,
        args.delay,
        args.max_examples,
        args.max_retries if hasattr(args, 'max_retries') else 5,
        args.base_delay if hasattr(args, 'base_delay') else 2.0
    )
    return True

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    on_backoff=lambda details: print(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries")
)
def run_analyze_step(paths, args):
    """Run the analysis step with backoff for retries"""
    analyze_bbq_results(
        str(paths["predictions_path"]),
        str(paths["analysis_dir"]),
        args.model_name
    )
    return True

def main():
    args = parse_arguments()
    completed_steps = []
    
    # Load from checkpoint if specified
    if args.checkpoint:
        checkpoint_data = load_checkpoint(args.checkpoint)
        if checkpoint_data:
            args.run_id = checkpoint_data["run_id"]
            completed_steps = checkpoint_data["completed_steps"]
            # Convert paths back to Path objects
            paths = {k: Path(v) for k, v in checkpoint_data["paths"].items()}
            print(f"Resuming from checkpoint with run ID: {args.run_id}")
            print(f"Completed steps: {', '.join(completed_steps)}")
        else:
            # Generate run ID if not provided and no valid checkpoint
            if not args.run_id:
                args.run_id = generate_run_id("bbq", args.model_name)
            # Setup directory structure
            paths = setup_directories(args.results_dir, "bbq", args.run_id)
    else:
        # Generate run ID if not provided
        if not args.run_id:
            args.run_id = generate_run_id("bbq", args.model_name)
        # Setup directory structure
        paths = setup_directories(args.results_dir, "bbq", args.run_id)
    
    print(f"Starting BBQ pipeline with run ID: {args.run_id}")
    print(f"Results will be saved to: {paths['run_dir']}")
    
    # Define checkpoint path
    checkpoint_path = paths["run_dir"] / "checkpoint.json"
    
    # Step 1: Prepare dataset
    if not args.skip_prepare and "prepare" not in completed_steps:
        print("\n=== Step 1: Preparing BBQ dataset ===")
        success = run_prepare_step(paths, args)
        if success:
            completed_steps.append("prepare")
            if args.save_checkpoint:
                save_checkpoint(checkpoint_path, completed_steps, paths, args)
    else:
        print("\n=== Skipping dataset preparation ===")
    
    # Step 2: Evaluate model
    if not args.skip_evaluate and "evaluate" not in completed_steps:
        print("\n=== Step 2: Evaluating model on BBQ dataset ===")
        success = run_evaluate_step(paths, args)
        if success:
            completed_steps.append("evaluate")
            if args.save_checkpoint:
                save_checkpoint(checkpoint_path, completed_steps, paths, args)
    else:
        print("\n=== Skipping model evaluation ===")
    
    # Step 3: Analyze results
    if not args.skip_analyze and "analyze" not in completed_steps:
        print("\n=== Step 3: Analyzing results ===")
        success = run_analyze_step(paths, args)
        if success:
            completed_steps.append("analyze")
            if args.save_checkpoint:
                save_checkpoint(checkpoint_path, completed_steps, paths, args)
    else:
        print("\n=== Skipping results analysis ===")
    
    print(f"\nBBQ pipeline completed successfully!")
    print(f"Results saved to: {paths['run_dir']}")
    
    # Create a summary file with the run configuration
    summary_path = paths["run_dir"] / "run_summary.txt"
    config = {
        "Model": args.model_name,
        "Model Type": args.model_type,
        "Prompt Strategy": args.prompt_strategy,
        "Dataset": "BBQ",
        "Examples": args.num_examples,
        "Categories": args.categories if args.categories else "all",
        "Split": args.split if args.split else "all",
        "Completed Steps": completed_steps
    }
    
    write_summary_file(
        summary_path=summary_path,
        title="BBQ Evaluation Pipeline",
        run_id=args.run_id,
        timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        config=config,
        paths=paths
    )
    
    # Add run to tracking system
    try:
        from common.tracking import add_run
        
        # Extract metrics from analysis if available
        metrics = {}
        metrics_file = paths["analysis_dir"] / "metrics.json"
        if metrics_file.exists() and (not args.skip_analyze or "analyze" in completed_steps):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Add run to tracking
        add_run(
            dataset_type="bbq",
            run_id=args.run_id,
            model_name=args.model_name,
            model_type=args.model_type,
            num_examples=args.num_examples,
            prompt_strategy=args.prompt_strategy,
            categories=args.categories,
            split=args.split,
            results_dir=str(paths["run_dir"]),
            metrics=metrics
        )
        print("Run added to tracking system.")
    except Exception as e:
        print(f"Warning: Could not add run to tracking system: {e}")
        # Continue execution even if tracking fails

if __name__ == "__main__":
    main()
