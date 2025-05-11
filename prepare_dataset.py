#!/usr/bin/env python3
"""
Unified dataset preparation script for bias evaluation datasets.

This script loads a specified dataset (BBQ or StereoSet), processes it,
and creates a standardized CSV file for model evaluation.
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
    parser = argparse.ArgumentParser(description="Prepare bias evaluation dataset samples")
    parser.add_argument("--dataset", type=str, required=True, choices=["bbq", "stereoset"],
                        help="Dataset to prepare (bbq or stereoset)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output CSV file (defaults to results/[dataset]_samples.csv)")
    parser.add_argument("--num_examples", type=int, default=100,
                        help="Number of examples to include in the output file")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated list of categories to include (default: all)")
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split to use (default depends on dataset)")
    parser.add_argument("--bias_type", type=str, default="intersentence",
                        choices=["intersentence", "intrasentence", "all"],
                        help="Type of bias examples to include (StereoSet only)")
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


def prepare_bbq_dataset(output_path, num_examples=None, categories=None, split="test", seed=42):
    """
    Prepare BBQ dataset for evaluation.
    
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
            "Dataset": "bbq",
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
    
    return df


def prepare_stereoset_dataset(output_path, num_examples=None, bias_type="all", categories=None, split="validation", seed=42):
    """
    Prepare StereoSet dataset for evaluation.
    
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
            "Dataset": "stereoset",
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
    
    # Set default output path if not specified
    if args.output is None:
        args.output = f"results/{args.dataset}_dataset_samples.csv"
    
    # Set default split based on dataset
    if args.split is None:
        args.split = "test" if args.dataset == "bbq" else "validation"
    
    # Parse categories if provided
    categories = args.categories.split(",") if args.categories else None
    
    if args.dataset == "bbq":
        prepare_bbq_dataset(
            args.output,
            num_examples=args.num_examples,
            categories=categories,
            split=args.split,
            seed=args.seed
        )
    elif args.dataset == "stereoset":
        prepare_stereoset_dataset(
            args.output,
            num_examples=args.num_examples,
            bias_type=args.bias_type,
            categories=categories,
            split=args.split,
            seed=args.seed
        )
    else:
        print(f"Error: Unknown dataset {args.dataset}")


if __name__ == "__main__":
    main()
