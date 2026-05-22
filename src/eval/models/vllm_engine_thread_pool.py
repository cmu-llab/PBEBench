import os
import sys
import json
import pathlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
sys.path.append(module_path)

VLLM_SERVER_URL = "http://0.0.0.0:8001/v1/chat/completions"
MAX_RETRIES = 5
MAX_SEQ_LEN = 32768
MAX_NEW_TOKENS = 2048

class vLLMEngine:
    def __init__(self, model_ckpt: str, port: int = 8001, seed: int = 42):
        self.port = port
        self.model_ckpt = model_ckpt
        self.seed = seed

    def generate_response(self, user_prompt: str, temperature: float = 0.7, top_p: float = 0.95, max_new_tokens: int = MAX_NEW_TOKENS):
        """Sends a request to vLLM for generating a response"""
        messages = [{"role": "user", "content": user_prompt}]
        payload = {
            "model": self.model_ckpt,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": self.seed,
        }

        for _ in range(MAX_RETRIES):
            try:
                response = requests.post(VLLM_SERVER_URL, json=payload).json()
                model_response = response["choices"][0]["message"]["content"]
                return {"model_response": model_response, "error": False}
            except Exception as e:
                print(f"Error: {e}")
        return {"model_response": "Got Error", "error": True}


def run_in_parallel(engine: vLLMEngine, prompts: list[str], max_workers: int = 8):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_prompt = {
            executor.submit(engine.generate_response, prompt): prompt for prompt in prompts
        }
        for future in as_completed(future_to_prompt):
            result = future.result()
            results.append(result)
    return results


# Example usage
if __name__ == "__main__":
    engine = vLLMEngine(model_ckpt="your-model-name")

    prompts = [
        "What is the capital of France?",
        "Summarize the plot of Inception.",
        "Explain the theory of relativity.",
        "What's the difference between TCP and UDP?",
    ]

    responses = run_in_parallel(engine, prompts, max_workers=4)
    for i, response in enumerate(responses):
        print(f"Prompt {i+1}: {prompts[i]}")
        print(f"Response: {response['model_response']}")
        print()
