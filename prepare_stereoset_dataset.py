#!/usr/bin/env python3
"""
Create StereoSet results file for metrics analysis with improved format.

This script loads the StereoSet dataset, iterates through examples, and creates a CSV file
with one row per example (containing all three options) for easier evaluation.
"""

import os
import argparse
import pandas as pd
import random
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Prepare StereoSet dataset samples with stratified sampling")
    parser.add_argument("--output", type=str, default="results/stereoset_dataset_samples.csv",
                         help="Path to output CSV file")
    parser.add_argument("--num_examples", type=int, default=100,
                         help="Number of examples to include in the output file")
    parser.add_argument("--bias_type", type=str, default="intersentence",
                         choices=["intersentence", "intrasentence", "all"],
                         help="Type of bias examples to include")
    parser.add_argument("--categories", type=str, default=None,
                         help="Comma-separated list of categories to include (default: all)")
    parser.add_argument("--split", type=str, default="validation",
                         choices=["validation", "test"],
                         help="Dataset split to use")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for sampling")
    return parser.parse_args()


def create_results_file(output_path, num_examples=None, bias_type="all", categories=None, split="validation", seed=42):
    """
    Create a results file for StereoSet metrics analysis with improved format.
    
    Args:
        output_path: Path to save the output CSV
        num_examples: Number of examples to process (None for all)
        bias_type: Type of bias examples to include (intersentence, intrasentence, or all)
        categories: List of categories to include (None for all)
        split: Dataset split to use
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    print(f"Loading StereoSet dataset (split: {split})...")
    
    # Process the data based on the bias type
    data = []
    
    if bias_type == "all" or bias_type == "intersentence":
        print("Loading intersentence data...")
        intersentence_data = list(load_dataset("McGill-NLP/stereoset", "intersentence")[split])
        for item in intersentence_data:
            item["bias_type"] = "intersentence"
            data.append(item)
        print(f"Added {len(intersentence_data)} examples from intersentence")
            
    if bias_type == "all" or bias_type == "intrasentence":
        print("Loading intrasentence data...")
        intrasentence_data = list(load_dataset("McGill-NLP/stereoset", "intrasentence")[split])
        for item in intrasentence_data:
            item["bias_type"] = "intrasentence"
            data.append(item)
        print(f"Added {len(intrasentence_data)} examples from intrasentence")
    
    print(f"Loaded {len(data)} examples in total")
    
    # Group examples by category
    examples_by_category = defaultdict(list)
    for i, example in enumerate(data):
        category = example["target"]
        examples_by_category[category].append((i, example))
    
    print(f"Found {len(examples_by_category)} categories")
    
    # Filter categories if specified
    if categories:
        category_list = categories.split(",")
        examples_by_category = {k: v for k, v in examples_by_category.items() if k in category_list}
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
    
    # Create results data (one row per example instead of three)
    results = []
    for example in tqdm(selected_data, desc="Processing examples"):
        # Get basic example data
        category = example["target"]
        bias_type = example["bias_type"]
        context = example["context"]
        
        # Get sentences and labels
        sentences = example["sentences"]["sentence"]
        labels = example["sentences"]["gold_label"]
        
        # Find the sentence for each label type (stereotype, anti-stereotype, unrelated)
        stereotype_option = None
        anti_stereotype_option = None
        unrelated_option = None
        
        for sentence, label in zip(sentences, labels):
            if label == 0:  # Stereotype
                stereotype_option = sentence
            elif label == 1:  # Anti-stereotype
                anti_stereotype_option = sentence
            elif label == 2:  # Unrelated
                unrelated_option = sentence
        
        # Create a single row with all three options
        results.append({
            "Category": category,
            "Context": context,
            "Stereotype_Option": stereotype_option,
            "Anti_Stereotype_Option": anti_stereotype_option,
            "Unrelated_Option": unrelated_option,
            "Bias_Type": bias_type,
            "ID": example.get("id", ""),  # Add ID if available for tracking
            "Model_Prediction": "",  # Empty field to be filled by evaluation script
            "Model_Score_Stereotype": "",  # Empty field for model scores
            "Model_Score_Anti_Stereotype": "",
            "Model_Score_Unrelated": ""
        })
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if there's actually a directory path
        os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Bias type distribution
    type_counts = df["Bias_Type"].value_counts()
    print("\nBias Type Distribution:")
    for bias_type, count in type_counts.items():
        print(f"  {bias_type}: {count} examples")
    
    return df


def main():
    """Main function."""
    args = parse_arguments()
    create_results_file(
        args.output,
        num_examples=args.num_examples,
        bias_type=args.bias_type,
        categories=args.categories,
        split=args.split,
        seed=args.seed
    )


if __name__ == "__main__":
    main() 