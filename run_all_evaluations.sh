#!/bin/bash

# Activate virtual environment
source venv/bin/activate

MODEL="Qwen/Qwen2.5-72B-Instruct-Turbo"
PROMPTS="baseline cot few_shot bias_aware structured contrastive"

# BBQ PCA dataset - commented out as already run
for P in $PROMPTS; do
  echo "Running BBQ PCA with $P strategy..."
  python -m bbq.run --model_type together --model_name "$MODEL" \
    --prompt_strategy "$P" --input results/bbq_pca_lhs_100.csv \
    --delay 0.1 --base_delay 0.01
done

# BBQ Semantic dataset
for P in $PROMPTS; do
  echo "Running BBQ Semantic with $P strategy..."
  python -m bbq.run --model_type together --model_name "$MODEL" \
    --prompt_strategy "$P" --input results/bbq_semantic_100.csv \
    --delay 0.1 --base_delay 0.01
done

# StereoSet PCA dataset
for P in $PROMPTS; do
  echo "Running StereoSet PCA with $P strategy..."
  python -m stereoset.run --model_type together --model_name "$MODEL" \
    --prompt_strategy "$P" --input results/stereoset_pca_lhs_100.csv \
    --delay 0.1 --base_delay 0.01 --dataset_type pca
done

# StereoSet Semantic dataset
for P in $PROMPTS; do
  echo "Running StereoSet Semantic with $P strategy..."
  python -m stereoset.run --model_type together --model_name "$MODEL" \
    --prompt_strategy "$P" --input results/stereoset_semantic_100.csv \
    --delay 0.1 --base_delay 0.01 --dataset_type semantic
done

echo "All evaluations complete!"
