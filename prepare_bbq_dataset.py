#!/usr/bin/env python3
"""
Create BBQ results file for metrics analysis.

This script loads the BBQ dataset, iterates through examples, and creates a CSV file
compatible with bbq_metrics.py without running model evaluation.
"""

import os
import argparse
import pandas as pd
import re
import random
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Prepare BBQ dataset samples with stratified sampling")
    parser.add_argument("--output", type=str, default="results/bbq_dataset_samples.csv",
                         help="Path to output CSV file")
    parser.add_argument("--num_examples", type=int, default=100,
                         help="Number of examples to include in the output file")
    parser.add_argument("--categories", type=str, default=None,
                         help="Comma-separated list of categories to include (default: all)")
    parser.add_argument("--split", type=str, default="all",
                         choices=["train", "validation", "test", "all"],
                         help="Dataset split to use")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for sampling")
    return parser.parse_args()


def parse_bbq_choices(example):
    """
    Parse choices from BBQ example.
    
    In the BBQ dataset, choices are embedded in the user message as (A), (B), (C) format.
    This function extracts these choices.
    """
    # Get the user message that contains the question and choices
    user_message = [msg for msg in example["messages"] if msg["role"] == "user"][0]["content"]
    
    # Extract the choices using regex
    choices_pattern = r"\(([A-C])\) ([^(]+)"
    choices_matches = re.findall(choices_pattern, user_message)
    
    # Create a dictionary mapping option letters to their text
    choices_dict = {letter: text.strip() for letter, text in choices_matches}
    
    return choices_dict


def create_results_file(output_path, num_examples=None, categories=None, split="test", seed=42):
    """
    Create a results file for BBQ metrics analysis.
    
    Args:
        output_path: Path to save the output CSV
        num_examples: Number of examples to process (None for all)
        categories: List of categories to include (None for all)
        split: Dataset split to use
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    print(f"Loading BBQ dataset (split: {split})...")
    bbq = load_dataset("jbreuch/bbq-all")
    
    # Handle the "all" split option
    if split == "all":
        print("Combining all available splits...")
        data = []
        for available_split in bbq.keys():
            split_data = list(bbq[available_split])
            data.extend(split_data)
            print(f"Added {len(split_data)} examples from {available_split} split")
    else:
        # Get the specified split
        if split not in bbq:
            print(f"Split '{split}' not found in dataset. Available splits: {list(bbq.keys())}")
            return
        data = list(bbq[split])
    
    print(f"Loaded {len(data)} examples")
    
    # Group examples by category
    examples_by_category = defaultdict(list)
    for i, example in enumerate(data):
        category = example["category"]
        examples_by_category[category].append((i, example))
    
    print(f"Found {len(examples_by_category)} categories")
    
    # Filter categories if specified
    if categories:
        examples_by_category = {k: v for k, v in examples_by_category.items() if k in categories}
        print(f"Filtered to {len(examples_by_category)} categories")
    
    available_categories = list(examples_by_category.keys())
    
    # If no number specified, use all examples
    if num_examples is None:
        selected_data = data
    else:
        # Calculate examples per category for stratified sampling
        examples_per_category = num_examples // len(examples_by_category)
        remaining = num_examples % len(examples_by_category)
        
        # Distribute remaining examples
        extra_examples = {cat: 1 if i < remaining else 0 
                         for i, cat in enumerate(available_categories)}
        
        # Select examples from each category
        selected_indices = []
        selected_data = []
        
        print("\nStratified sampling:")
        for category in available_categories:
            category_examples = examples_by_category[category]
            num_to_select = examples_per_category + extra_examples[category]
            
            # If we need more than available, take all
            if num_to_select >= len(category_examples):
                selected = category_examples
                print(f"  {category}: taking all {len(selected)} examples")
            else:
                # Random sample without replacement
                selected = random.sample(category_examples, num_to_select)
                print(f"  {category}: selected {len(selected)} of {len(category_examples)} examples")
            
            # Add selected examples
            for idx, example in selected:
                selected_indices.append(idx)
                selected_data.append(example)
        
        print(f"Selected {len(selected_data)} examples in total")
    
    # Create results data
    results = []
    for example in tqdm(selected_data, desc="Processing examples"):
        # Get basic example data
        category = example["category"]
        context = example["context"]
        question = example["question"]
        correct_answer = example["answer"]  # The letter of the correct answer (A, B, or C)
        
        # Parse choices from the user message
        choices_dict = parse_bbq_choices(example)
        
        # Get ambiguity condition
        context_condition = "ambig"
        if "tags" in example and isinstance(example["tags"], dict) and "ambiguity" in example["tags"]:
            context_condition = example["tags"]["ambiguity"]
        
        # Determine if the example is ambiguous
        is_ambiguous = context_condition in ["ambig", "ambiguous"]
        
        # Get target answer if available (for stereotype analysis)
        target_answer = None
        if "tags" in example and isinstance(example["tags"], dict) and "target_answer" in example["tags"]:
            target_answer = example["tags"]["target_answer"]
        
        # Add to results
        results.append({
            "Category": category,
            "Context": context,
            "Question": question,
            "Ground Truth": correct_answer,
            "Predicted Output": correct_answer,  # Default to correct for dataset exploration
            "Ambiguity": "ambiguous" if is_ambiguous else "disambiguated",
            "Context Condition": context_condition,
            "Target Answer": target_answer,
            "Correct": True,  # Default to True for dataset exploration
            "Choices": ", ".join([f"{k}: {v}" for k, v in choices_dict.items()]),
        })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(results)} examples to {output_path}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    
    # Category distribution
    cat_counts = df["Category"].value_counts()
    print("\nCategory Distribution:")
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count} examples")
    
    # Ambiguity distribution
    ambig_counts = df["Ambiguity"].value_counts()
    print("\nAmbiguity Distribution:")
    for ambig, count in ambig_counts.items():
        print(f"  {ambig}: {count} examples")
    
    # Context condition distribution
    condition_counts = df["Context Condition"].value_counts()
    print("\nContext Condition Distribution:")
    for cond, count in condition_counts.items():
        print(f"  {cond}: {count} examples")
    
    return df


def main():
    """Main function."""
    args = parse_arguments()
    create_results_file(
        args.output,
        num_examples=args.num_examples,
        categories=args.categories.split(",") if args.categories else None,
        split=args.split,
        seed=args.seed
    )


if __name__ == "__main__":
    main() 