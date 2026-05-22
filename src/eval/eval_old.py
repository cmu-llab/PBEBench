import argparse
import asyncio
import json
import tqdm
import os

from models.loader import load_model, generate
from utils import prompt_construct
from src.eval.utils import read_jsonl
from budget_tracker import estimate_tokens, PRICES

async def run_budget_dry_run(args, dataset):
    total_estimated_cost = 0.0
    model_name = args.model_ckpt
    if model_name not in PRICES:
        print(f"Error: Model '{model_name}' not found in PRICES for budget dry run.")
        return
    
    pricing = PRICES[model_name]
    if pricing["input_per_1k"] is None or pricing["output_per_1k"] is None:
        print(f"Error: Pricing data not available for '{model_name}' for budget dry run.")
        return
        
    print(f"--- Starting Budget Dry Run for model: {model_name} ---")
    for data in dataset:
        input_text = data['prompt']
        input_tokens = estimate_tokens(input_text)
        output_tokens = args.max_new_tokens  # Approximate output tokens as max_new_tokens

        input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
        output_cost = (output_tokens / 1000) * pricing["output_per_1k"]
        total_estimated_cost += (input_cost + output_cost)
    
    print(f"Estimated total cost for {len(dataset)} samples: ${total_estimated_cost:.4f}")
    print("--- Budget Dry Run Complete ---")

async def main(args):
    if args.input_path.endswith('.jsonl'):
        dataset = []
        with open(args.input_path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))
    elif args.input_path.endswith('.json'):
        dataset = json.load(open(args.input_path))
    else:
        raise ValueError('Invalid input file format')

    if args.budget_dry_run:
        await run_budget_dry_run(args, dataset)
        return # Exit without making API calls

    if args.prompt_path == "v1":
        from prompts import prompt_v1
        prompt = prompt_v1
    else:
        with open(args.prompt_path, 'r') as f:
            prompt = f.read()

    engine = load_model(args.model_ckpt, args.engine_type, args.tp)
    sample_parameters = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "prompt": prompt,
        "max_rules_num": args.max_rules_num,
        "program_num": args.max_rules_num,
        "program_length": args.max_tokens_length,
        "max_rule_length": args.max_tokens_length,
        "characters": args.character,
        "model_ckpt": args.model_ckpt,
    }

    with open(args.output_path, 'w') as f: pass

    for i in tqdm.tqdm(range(0, len(dataset), args.batchsize)):
        batch = dataset[i:i+args.batchsize]
        inputs = []
        for data in batch:
            inputs.append(data['prompt'])

        all_sample_outputs = [[] for _ in range(len(inputs))]

        for _ in range(args.num_samples):
            sampled_outputs = generate(engine, inputs, args.engine_type, sample_parameters)
            for j, output in enumerate(sampled_outputs):
                all_sample_outputs[j].append(output)

        with open(args.output_path, 'a') as f:
            for input_item, output_list in zip(batch, all_sample_outputs):
                f.write(json.dumps({"input": input_item, "outputs": output_list}) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--prompt_path', type=str, default="v1")
    parser.add_argument('--max_rules_num', type=int, default=5)
    parser.add_argument('--max_tokens_length', type=int, default=3)
    parser.add_argument('--character', type=str, default='[a, b, c]')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--engine_type', type=str, choices=['huggingface', 'openai', 'claude', 'gemini', 'sglang', 'vllm'], required=True)
    parser.add_argument('--budget_dry_run', action='store_true', help="Approximate cost without making API calls.")
    args = parser.parse_args()

    asyncio.run(main(args))
