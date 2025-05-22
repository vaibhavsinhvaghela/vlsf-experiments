import os
import time
import json
import requests
import re
from dotenv import load_dotenv
from together import Together

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def call_together_api(prompt, model_name, prompt_strategy="baseline", max_retries=5, base_delay=2.0):
    """
    Call the Together API with the given prompt and return the generated text.
    Implements exponential backoff for rate limiting.
    """
    if not TOGETHER_API_KEY:
        raise ValueError("TOGETHER_API_KEY not found in .env file")
    client = Together(api_key=TOGETHER_API_KEY)
    prompt = f"[INST] {prompt} [/INST]" if "Mixtral" in model_name else prompt
    for retry in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = base_delay * (2 ** retry)
                print(f"Rate limited. Waiting {wait_time:.1f} seconds before retry {retry+1}/{max_retries}")
                time.sleep(wait_time)
                continue
            print(f"API request error: {e}")
            return ""
    print(f"Error: Failed to call Together API after {max_retries} retries")
    return ""


def call_openai_api(prompt, model_name="gpt-4", max_retries=5, base_delay=2.0):
    """
    Call the OpenAI API with the given prompt and return the generated text.
    Implements exponential backoff for rate limiting.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 500
    }
    for retry in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
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
    """
    Call the Anthropic API with the given prompt and return the generated text.
    Implements exponential backoff for rate limiting.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "anthropic-version": "2023-06-01",
        "x-api-key": ANTHROPIC_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.1
    }
    for retry in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
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


def call_gemini_api(prompt, model_name="gemini-2.0-flash", max_retries=5, base_delay=2.0):
    """
    Call the Gemini API with the given prompt and return the generated text.
    Implements exponential backoff for rate limiting.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    for retry in range(max_retries):
        try:
            response = requests.post(url, json=payload)
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


def mock_api_call(prompt, model_name="mock", **kwargs):
    """
    Mock API for testing purposes. Returns a simulated response.
    """
    print("Using MOCK API - no actual API call is being made")
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


def get_model_api(model_type):
    """
    Returns the appropriate API function based on model type.
    """
    api_functions = {
        "together": call_together_api,
        "gemini": call_gemini_api,
        "openai": call_openai_api,
        "anthropic": call_anthropic_api,
        "mock": mock_api_call
    }
    if model_type not in api_functions:
        raise ValueError(f"Unsupported model type: {model_type}")
    return api_functions[model_type]
