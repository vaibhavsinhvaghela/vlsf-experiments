# BBQ Dataset Bias Evaluation Toolkit

This toolkit provides a streamlined workflow for evaluating language model bias using the Bias Benchmark for QA (BBQ) dataset.

## About the BBQ Dataset

The BBQ (Bias Benchmark for QA) dataset is designed to measure social biases in language models through a multiple-choice question-answering format. Key characteristics:

- **Format**: Multiple-choice questions with 3 options (A, B, C)
- **Categories**: Tests for bias across 9 social dimensions:
  - Age
  - Disability status
  - Gender identity
  - Nationality
  - Physical appearance
  - Race/ethnicity
  - Religion
  - Sexual orientation
  - Socioeconomic status (SES)
  
- **Question Types**:
  - **Ambiguous**: Questions where the context provides insufficient information to determine the answer, requiring models to avoid defaulting to stereotypes
  - **Disambiguated**: Questions with clear context that provides a definitive answer

- **Bias Detection**: The dataset tracks "target answers" (stereotypical choices) to measure if models show bias by selecting these options in ambiguous contexts

## Toolkit Components

This toolkit consists of three main scripts:

1. **prepare_bbq_dataset.py**: Creates stratified samples from the BBQ dataset
2. **evaluate_model_on_bbq.py**: Evaluates language models on the BBQ samples
3. **analyze_bbq_results.py**: Generates metrics and visualizations from evaluation results

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd bias_detection_datasets

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install datasets pandas numpy matplotlib tqdm requests python-dotenv
```

Create a `.env` file in the root directory with your API keys:

```
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Usage

### 1. Prepare the Dataset

Create a stratified sample from the BBQ dataset:

```bash
python3 prepare_bbq_dataset.py --num_examples 100 --seed 42
```

Options:
- `--output`: Path to save the output CSV (default: results/bbq_dataset_samples.csv)
- `--num_examples`: Number of examples to include (default: 100)
- `--categories`: Comma-separated list of categories to include (default: all)
- `--split`: Dataset split to use (default: all)
- `--seed`: Random seed for sampling (default: 42)

### 2. Evaluate a Model

Run a model evaluation on the prepared dataset:

```bash
python3 evaluate_model_on_bbq.py --model_type gemini --model_name "gemini-2.0-flash" --max_examples 50
```

Options:
- `--input`: Path to input CSV file (default: results/bbq_dataset_samples.csv)
- `--output`: Path to output CSV file (default: results/model_evaluation_results.csv)
- `--model_type`: Type of model to use ["gemini", "openai", "anthropic", "mock"]
- `--model_name`: Specific model name to use
- `--delay`: Delay between API calls in seconds (default: 1.0)
- `--max_examples`: Maximum number of examples to process

### 3. Analyze Results

Generate metrics and visualizations from the evaluation results:

```bash
python3 analyze_bbq_results.py --model_name "Gemini 2.0 Flash"
```

Options:
- `--input`: Path to input CSV file (default: results/model_evaluation_results.csv)
- `--output_dir`: Directory to save analysis results and plots (default: results/analysis_metrics)
- `--model_name`: Name of the model to include in output files and plot titles

## Metrics Explained

The analysis generates several key metrics:

- **Overall Accuracy**: How often the model answers correctly across all questions
- **Accuracy by Category**: Performance broken down by social dimension
- **Accuracy by Ambiguity**: Performance on ambiguous vs. unambiguous questions
- **Stereotype Score**: Rate at which the model selects the stereotypical answer in ambiguous contexts
- **Bias Score**: The difference in accuracy between disambiguated and ambiguous contexts

## Example Workflow

```bash
# 1. Create a dataset with 200 examples, evenly distributed across categories
python prepare_bbq_dataset.py --num_examples 200

# 2. Evaluate using OpenAI's GPT-4
python evaluate_model_on_bbq.py --model_type openai --model_name "gpt-4" --delay 3

# 3. Analyze the results with custom output directory
python analyze_bbq_results.py --model_name "GPT-4" --output_dir "results/gpt4_analysis"
```

## License

This project makes use of the BBQ dataset created by Parrish et al. If you use this toolkit in your research, please cite the original BBQ paper:

```
@article{parrish2022bbq,
  title={BBQ: A hand-built bias benchmark for question answering},
  author={Parrish, Alicia and Chen, Angelica and Nangia, Nikita and Padmakumar, Vishakh and Phang, Jason and Thompson, Jana and Htut, Phu Mon and Bowman, Samuel},
  journal={Findings of the Association for Computational Linguistics: ACL 2022},
  year={2022}
}
``` 