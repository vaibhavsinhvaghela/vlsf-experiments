import os
import csv
import time
import requests
import argparse
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate language models on BBQ dataset")
    parser.add_argument("--input", type=str, default="results/bbq_dataset_samples.csv",
                        help="Path to input CSV file with BBQ dataset")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output CSV file with model predictions (defaults to model-specific name)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between API calls in seconds to avoid rate limiting")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process (for testing)")
    parser.add_argument("--model_type", type=str, default="gemini",
                        choices=["gemini", "openai", "anthropic", "mock"],
                        help="Type of model to use for evaluation")
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash",
                        help="Specific model name to use (e.g., gemini-2.0-flash, gpt-4, claude-3-opus)")
    return parser.parse_args()

def extract_answer_choice(response_text, choices):
    choice_letters = list(choices.keys())
    for letter in choice_letters:
        patterns = [
            rf"\b{letter}\b",
            rf"option\s+{letter}",
            rf"choice\s+{letter}",
            rf"answer\s+{letter}",
            rf"answer is\s+{letter}",
            rf"\({letter}\)",
        ]
        for pattern in patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                return letter
    for letter, text in choices.items():
        clean_text = text.lower().strip()
        if clean_text in response_text.lower():
            return letter
    neither_patterns = [
        r"\bneither\b",
        r"\bboth\s+equally\b",
        r"\bequally\s+likely\b",
        r"\bcannot\s+determine\b",
        r"\bcannot\s+be\s+determined\b",
        r"\bimpossible\s+to\s+determine\b",
        r"\binsufficient\s+information\b",
        r"\bnot\s+enough\s+information\b",
    ]
    for pattern in neither_patterns:
        if re.search(pattern, response_text, re.IGNORECASE):
            for letter, text in choices.items():
                if re.search(r"neither|both|equal|cannot|determine", text, re.IGNORECASE):
                    return letter
    print(f"Warning: Could not determine answer choice from: {response_text}")
    return choice_letters[0] if choice_letters else "A"

def call_gemini_api(prompt, model_name="gemini-2.0-flash", max_retries=5, base_delay=2.0):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    for retry in range(max_retries):
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 429:
                wait_time = base_delay * (2 ** retry)
                print(f"Rate limited. Waiting {wait_time:.1f} seconds before retry {retry+1}/{max_retries}")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                content = data["candidates"][0].get("content", {})
                parts = content.get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
            print(f"Warning: Unexpected API response format: {data}")
            return ""
        except requests.exceptions.RequestException as e:
            wait_time = base_delay * (2 ** retry)
            print(f"API request error: {e}. Waiting {wait_time:.1f} seconds before retry {retry+1}/{max_retries}")
            time.sleep(wait_time)
    print(f"Error: Failed to call Gemini API after {max_retries} retries")
    return ""

def call_openai_api(prompt, model_name="gpt-3.5-turbo", max_retries=5, base_delay=2.0):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 500
    }
    for retry in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 429:
                wait_time = base_delay * (2 ** retry)
                print(f"Rate limited. Waiting {wait_time:.1f} seconds before retry {retry+1}/{max_retries}")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            print(f"Warning: Unexpected API response format: {data}")
            return ""
        except requests.exceptions.RequestException as e:
            wait_time = base_delay * (2 ** retry)
            print(f"API request error: {e}. Waiting {wait_time:.1f} seconds before retry {retry+1}/{max_retries}")
            time.sleep(wait_time)
    print(f"Error: Failed to call OpenAI API after {max_retries} retries")
    return ""

def call_anthropic_api(prompt, model_name="claude-3-opus-20240229", max_retries=5, base_delay=2.0):
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file")
    url = "https://api.anthropic.com/v1/messages"
    headers = {"anthropic-version": "2023-06-01", "x-api-key": ANTHROPIC_API_KEY, "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.1
    }
    for retry in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 429:
                wait_time = base_delay * (2 ** retry)
                print(f"Rate limited. Waiting {wait_time:.1f} seconds before retry {retry+1}/{max_retries}")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            data = response.json()
            if "content" in data and len(data["content"]) > 0:
                return data["content"][0]["text"]
            print(f"Warning: Unexpected API response format: {data}")
            return ""
        except requests.exceptions.RequestException as e:
            wait_time = base_delay * (2 ** retry)
            print(f"API request error: {e}. Waiting {wait_time:.1f} seconds before retry {retry+1}/{max_retries}")
            time.sleep(wait_time)
    print(f"Error: Failed to call Anthropic API after {max_retries} retries")
    return ""

def mock_api_call(prompt, model_name="mock", **kwargs):
    choices = {}
    choice_pattern = r'\(([A-C])\)\s+(.*?)(?=\([A-C]\)|$)'
    matches = re.findall(choice_pattern, prompt)
    for letter, text in matches:
        choices[letter] = text.strip()
    question_match = re.search(r'Question:\s*(.*?)\n', prompt)
    question = question_match.group(1) if question_match else ""
    if 'C' in choices and re.search(r'\b(not|cannot|undetermined|unknown)\b', choices['C'], re.IGNORECASE):
        return "The answer is C."
    elif question.lower().startswith("who"):
        if "ambiguous" in prompt.lower() or "unclear" in prompt.lower():
            return "Based on the context, I would choose B as there is insufficient information to determine."
        else:
            return "A"
    else:
        return f"I believe the correct answer is {'A' if hash(prompt) % 3 == 0 else 'B' if hash(prompt) % 3 == 1 else 'C'}"

def evaluate_bbq_dataset(input_file, output_file, model_type, model_name, delay=1.0, max_examples=None):
    with open(input_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    
    if max_examples and max_examples > 0:
        rows = rows[:max_examples]
    
    total_examples = len(rows)
    print(f"Processing {total_examples} examples from {input_file} using {model_type} model: {model_name}")
    
    model_api = {
        "gemini": call_gemini_api,
        "openai": call_openai_api,
        "anthropic": call_anthropic_api,
        "mock": mock_api_call
    }.get(model_type)
    
    if not model_api:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    for i, row in enumerate(rows):
        choices_str = row["Choices"]
        choices = {}
        choices_pattern = r'([A-C]):\s*(.*?)(?:,\s*(?=[A-C]:)|$)'
        matches = re.findall(choices_pattern, choices_str)
        for letter, text in matches:
            choices[letter] = text.strip('"')
        
        context = row["Context"].strip()
        question = row["Question"].strip()
        
        prompt = f"{context}\n\nQuestion: {question}\n\n"
        for letter, text in choices.items():
            prompt += f"({letter}) {text}\n"
        prompt += "\nPlease answer with the letter (A, B, or C) that best answers the question based on the given context."
        
        print(f"[{i+1}/{total_examples}] Calling {model_type} API for example in category: {row['Category']}")
        response_text = model_api(prompt, model_name=model_name)
        
        answer_choice = extract_answer_choice(response_text, choices)
        
        rows[i]["Predicted Output"] = answer_choice
        rows[i]["Model Response"] = response_text
        
        if (i + 1) % 10 == 0 or i == total_examples - 1:
            print(f"Saving intermediate results after processing {i+1} examples...")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, mode='w', encoding='utf-8') as file:
                fieldnames = list(rows[0].keys())
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        if i < total_examples - 1:
            time.sleep(delay)
    
    print(f"Evaluation complete. Results saved to {output_file}")

def main():
    args = parse_arguments()
    output_file = args.output if args.output else f"results/model_evaluation_results_{args.model_name.replace('-', '_')}.csv"
    evaluate_bbq_dataset(
        args.input,
        output_file,
        args.model_type,
        args.model_name,
        args.delay,
        args.max_examples
    )

if __name__ == "__main__":
    main()