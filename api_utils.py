import os
import time
import requests
import json
import re

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def call_together_api(prompt, model_name, prompt_strategy="baseline", max_retries=5, base_delay=2.0):
    if not TOGETHER_API_KEY:
        raise ValueError("TOGETHER_API_KEY not found in .env file")
    if not TOGETHER_AVAILABLE:
        raise ImportError("Together API client not available. Install with 'pip install together'")
    client = Together(api_key=TOGETHER_API_KEY)
    prompt = f"[INST] {prompt} [/INST]" if "Mixtral" in model_name else prompt
    for retry in range(max_retries):
        try:
            if prompt_strategy == "structured":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=500
                )
            else:
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

def call_gemini_api(prompt, model_name="gemini-2.0-flash", max_retries=5, base_delay=2.0):
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

def call_openai_api(prompt, model_name="gpt-4-turbo", max_retries=5, base_delay=2.0):
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
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}]
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

def mock_api_call(prompt, model_name="mock", **kwargs):
    is_bbq = "BBQ" in prompt or "bias benchmark" in prompt.lower()
    is_stereoset = "StereoSet" in prompt or "continuation" in prompt or "BLANK" in prompt
    if is_bbq:
        choice_pattern = r"\\(([A-C])\\)"
        choices = re.findall(choice_pattern, prompt)
        if choices:
            import random
            weights = [0.5, 0.3, 0.2][:len(choices)]
            weights = [w / sum(weights) for w in weights]
            selected = random.choices(choices, weights=weights)[0]
            return f"The answer is {selected}."
        else:
            return "I don't see the answer choices. Please provide options."
    elif is_stereoset:
        options = re.findall(r"Option \\d+: (.*?)(?=Option \\d+:|$)", prompt, re.DOTALL)
        options = [opt.strip() for opt in options]
        if options:
            import random
            scores = [random.uniform(0.6, 0.9), random.uniform(0.4, 0.8), random.uniform(0.1, 0.4)]
            return f"I've evaluated the options to fill in the blank:\n" + "\n".join([f"- Option {i+1}: {score:.2f}" for i, score in enumerate(scores[:len(options)])])
        else:
            return "I need specific options to evaluate for the blank."
    return "I've analyzed the prompt but I'm not sure how to respond appropriately."

def get_model_api(model_type):
    if model_type == "together":
        return call_together_api
    elif model_type == "gemini":
        return call_gemini_api
    elif model_type == "openai":
        return call_openai_api
    elif model_type == "anthropic":
        return call_anthropic_api
    elif model_type == "mock":
        return mock_api_call
    else:
        raise ValueError(f"Unknown model type: {model_type}")
