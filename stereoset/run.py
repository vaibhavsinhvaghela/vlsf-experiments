#!/usr/bin/env python3
"""
StereoSet Pipeline Runner
------------------------
This script provides a unified interface to run the complete StereoSet evaluation pipeline:
1. Prepare the StereoSet dataset
2. Evaluate models on the dataset
3. Analyze the results

Usage:
    python -m stereoset.run --model_name mistralai/Mistral-7B-Instruct-v0.1 --model_type together
"""

import os
import sys
import re
import argparse
import datetime
import json
import backoff
from pathlib import Path

# Import StereoSet modules
from stereoset.prepare_stereoset_dataset import create_results_file
from stereoset.evaluate_model_on_stereoset import evaluate_stereoset_dataset
from stereoset.analyze_stereoset_results import analyze_stereoset_results

# Import common utilities
from common.pipeline_utils import generate_run_id, setup_directories, write_summary_file

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the complete StereoSet evaluation pipeline")
    
    # Dataset preparation arguments
    parser.add_argument("--num_examples", type=int, default=100,
                       help="Number of examples to sample from the dataset")
    parser.add_argument("--bias_types", type=str, default=None,
                       help="Comma-separated list of bias types to include (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    
    # Model evaluation arguments
    parser.add_argument("--model_type", type=str, default="together",
                       choices=["together", "gemini", "openai", "anthropic", "mock"],
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
    bias_types = args.bias_types.split(",") if args.bias_types else None
    create_results_file(
        output_path=str(paths["dataset_path"]),
        num_examples=args.num_examples,
        bias_type=bias_types[0] if bias_types else "all",
        categories=None,  # Use all categories
        split="validation",  # Use validation split
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
    evaluate_stereoset_dataset(
        str(paths["dataset_path"]),
        str(paths["predictions_path"]),
        args.model_type,
        args.model_name,
        args.delay,
        args.max_examples,
        args.max_retries,
        args.base_delay
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
    analyze_stereoset_results(
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
                args.run_id = generate_run_id("stereoset", args.model_name)
            # Setup directory structure
            paths = setup_directories(args.results_dir, "stereoset", args.run_id)
    else:
        # Generate run ID if not provided
        if not args.run_id:
            args.run_id = generate_run_id("stereoset", args.model_name)
        # Setup directory structure
        paths = setup_directories(args.results_dir, "stereoset", args.run_id)
    
    print(f"Starting StereoSet pipeline with run ID: {args.run_id}")
    print(f"Results will be saved to: {paths['run_dir']}")
    
    # Define checkpoint path
    checkpoint_path = paths["run_dir"] / "checkpoint.json"
    
    # Step 1: Prepare dataset
    if not args.skip_prepare and "prepare" not in completed_steps:
        print("\n=== Step 1: Preparing StereoSet dataset ===")
        success = run_prepare_step(paths, args)
        if success:
            completed_steps.append("prepare")
            if args.save_checkpoint:
                save_checkpoint(checkpoint_path, completed_steps, paths, args)
    else:
        print("\n=== Skipping dataset preparation ===")
    
    # Step 2: Evaluate model
    if not args.skip_evaluate and "evaluate" not in completed_steps:
        print("\n=== Step 2: Evaluating model on StereoSet dataset ===")
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
    
    print(f"\nStereoSet pipeline completed successfully!")
    print(f"Results saved to: {paths['run_dir']}")
    
    # Create a summary file with the run configuration
    summary_path = paths["run_dir"] / "run_summary.txt"
    config = {
        "Model": args.model_name,
        "Model Type": args.model_type,
        "Prompt Strategy": args.prompt_strategy,
        "Dataset": "StereoSet",
        "Examples": args.num_examples,
        "Bias Types": args.bias_types if args.bias_types else "all",
        "Completed Steps": completed_steps
    }
    
    write_summary_file(
        summary_path=summary_path,
        title="StereoSet Evaluation Pipeline",
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
        metrics_file = paths["analysis_dir"] / "metrics_summary.txt"
        if metrics_file.exists() and (not args.skip_analyze or "analyze" in completed_steps):
            # For StereoSet, we'll extract key metrics from the summary file
            # since they're not stored in JSON format
            try:
                metrics_text = metrics_file.read_text()
                # Extract stereotype score
                ss_match = re.search(r'Stereotype Score \(SS\): ([0-9.]+)', metrics_text)
                if ss_match:
                    metrics["stereotype_score"] = float(ss_match.group(1))
                # Extract ICAT score
                icat_match = re.search(r'ICAT Score \(LMS × \(1 - SS\)\): ([0-9.]+)', metrics_text)
                if icat_match:
                    metrics["icat_score"] = float(icat_match.group(1))
                # Extract language modeling score
                lms_match = re.search(r'Language Modeling Score \(LMS\): ([0-9.]+)', metrics_text)
                if lms_match:
                    metrics["language_modeling_score"] = float(lms_match.group(1))
            except Exception:
                pass
        
        # Add run to tracking
        add_run(
            dataset_type="stereoset",
            run_id=args.run_id,
            model_name=args.model_name,
            model_type=args.model_type,
            num_examples=args.num_examples,
            prompt_strategy=args.prompt_strategy,
            bias_types=args.bias_types,
            results_dir=str(paths["run_dir"]),
            metrics=metrics
        )
        print("Run added to tracking system.")
    except Exception as e:
        print(f"Warning: Could not add run to tracking system: {e}")
        # Continue execution even if tracking fails

if __name__ == "__main__":
    main()
