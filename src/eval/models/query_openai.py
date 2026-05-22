import os
import sys
import json
import openai
import pathlib
import argparse
from tqdm import tqdm

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
sys.path.append(module_path)

from src.eval.utils import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Query OpenAI models for inference.")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o",
                        help="Model name (e.g., gpt-4o, o3-mini, o4-mini, gpt-5)")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input JSON/JSONL file with prompts")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output JSONL file for predictions")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max completion tokens")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature (uses API default if not specified)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Top-p (uses API default if not specified)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    model_name = args.model
    input_path = args.input
    output_path = args.output
    max_tokens = args.max_tokens
    
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Support both .json and .jsonl
    if input_path.endswith(".jsonl"):
        data = read_jsonl(input_path)
    else:
        data = json.load(open(input_path))
    
    for i, rec in enumerate(data):
        rec["index"] = i
    
    preds = []
    covered_ids = set()
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        open(output_path, "w").close()
    else:
        preds = read_jsonl(output_path)
        covered_ids = set(rec['input']['index'] for rec in preds)
        print(f"Resuming: found {len(covered_ids)} covered IDs")

    print(f"Model: {model_name}")
    print(f"Max tokens: {max_tokens}")
    if args.temperature is not None:
        print(f"Temperature: {args.temperature}")
    if args.top_p is not None:
        print(f"Top-p: {args.top_p}")
    
    for i, rec in tqdm(enumerate(data), total=len(data)):
        if i in covered_ids:
            continue
        
        messages = [{"role": "user", "content": rec['prompt']}]
        
        # Build request params
        request_params = {
            "model": model_name,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        
        # Only add temperature/top_p if explicitly provided
        if args.temperature is not None:
            request_params["temperature"] = args.temperature
        if args.top_p is not None:
            request_params["top_p"] = args.top_p
        
        try:
            response = client.chat.completions.create(**request_params)
            output_text = response.choices[0].message.content
            raw_response = response.to_dict()
        except openai.BadRequestError as e:
            # Handle max_tokens exceeded or other bad request errors
            print(f"\nWarning: BadRequestError for index {i}: {e}")
            output_text = None
            raw_response = {"error": str(e)}
        
        pred = {
            "input": rec,
            "outputs": [output_text],
            "raw_response": raw_response,
        }
        preds.append(pred)
        with open(output_path, "a") as f:
            f.write(json.dumps(pred) + "\n")
