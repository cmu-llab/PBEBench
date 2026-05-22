import os
import sys
import copy
import json
import pathlib
import argparse
import requests
from tqdm import tqdm
from typing import Union
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
# print(module_path)
sys.path.append(module_path)

from src.eval.utils import prompt_construct, read_jsonl

def get_args():
    parser = argparse.ArgumentParser(description="Run inference with different settings.")
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, default=None, help="path to the config to be used.")
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--prompt_path', type=str, default="v1")
    parser.add_argument('--max_rules_num', type=int, default=5)
    parser.add_argument('--max_tokens_length', type=int, default=3)
    parser.add_argument('--character', type=str, default='[a, b, c]')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=8192)
    parser.add_argument('--reasoning_effort', type=str, default=None, help='set the reasoning effort')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--start_index', default=-1, type=int, help='ending index for experiments/data (inclusive)')
    parser.add_argument('--end_index', default=-1, type=int, help='starting index for experiments/data (inclusive)')
    parser.add_argument('--num_samples', type=int, default=1, help="how many samples to draw per instance")
    # parser.add_argument("--model_name", type=str, required=True, help="which model is to be queried")
    parser.add_argument("--no_think", action="store_true", help="Disable chain-of-thought reasoning globally.")
    parser.add_argument("--port", type=int, default=8002, help="Port where vLLM server is being served")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of parallel threads/workers to be used for querying vLLM.")

    return parser.parse_args()

args = get_args()
assert args.num_samples >= 1, "num samples can only be 1 or more"
# vLLM server details
PORT = args.port
VLLM_SERVER_URL = f"http://0.0.0.0:{PORT}/v1/chat/completions"
MAX_RETRIES = 5
MAX_NEW_TOKENS = args.max_new_tokens
NUM_WORKERS = args.num_workers

def generate_response(rec, model_ckpt, sample_parameters):
    user_prompt = rec['prompt'] # directly use the pre-generated prompts in promptsfile.

    # if len(user_prompt) > MAX_SEQ_LEN - MAX_NEW_TOKENS:
    #     user_prompt = user_prompt[:15000] + user_prompt[-15000:]

    messages = [{"role": "user", "content": user_prompt}]

    payload = {
        "model": model_ckpt,
        "messages": messages,
        "max_tokens": sample_parameters['max_new_tokens'],
        "temperature": sample_parameters['temperature'],
        "top_p": sample_parameters['top_p'],
        "top_k": sample_parameters['top_k'],
        "seed": 42
    }
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort

    if sample_parameters['no_think']:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    response = requests.post(VLLM_SERVER_URL, json=payload)
    response.raise_for_status()
    response_json = response.json()
    # print(response_json)
    # print(response_json, type(response_json))
    model_response = response_json["choices"][0]["message"]
    # print(model_response)
    # print(model_response)
    ret_response = {
        "input": rec,
        "outputs": [model_response['content']],
        "chains_of_thought": [model_response.get('reasoning_content')]
    }
    # print(ret_response["chains_of_thought"])

    return ret_response

def main(args):
    # model_name = args.model_name
    # write_path = args.write_path]

    model_ckpt = args.model_ckpt
    if args.prompt_path == "v1":
        from src.eval.prompts import prompt_v1
        prompt = prompt_v1
    else:
        with open(args.prompt_path, 'r') as f:
            prompt = f.read()
    sample_parameters = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "no_think": args.no_think,
        "prompt": prompt,
        "max_rules_num": args.max_rules_num,
        "program_num": args.max_rules_num,
        "program_length": args.max_tokens_length,
        "max_rule_length": args.max_tokens_length,
        "characters": args.character,
    }
    if args.input_path.endswith('.jsonl'):
        base_dataset = read_jsonl(args.input_path)
    elif args.input_path.endswith('.json'):
        base_dataset = json.load(open(args.input_path))
    # elif 
    else:
        if args.config_path is not None:
            base_dataset = load_dataset(args.input_path, args.config_path)['test']
        else:
            base_dataset = load_dataset(args.input_path)['test']
        base_dataset = [base_dataset[i] for i in range(len(base_dataset))]
        # raise ValueError('Invalid input file format')
    
    # add indices to all instances.
    dataset = []
    for i in range(len(base_dataset)):
        if args.num_samples == 1:
            base_dataset[i]["index"] = i
            dataset.append(base_dataset[i])
        else: # stack each instance in the data num_samples times.
            for j in range(args.num_samples):
                rec = copy.deepcopy(base_dataset[i])
                rec['index'] = f"{i}_{j}"
                dataset.append(rec)

    if args.start_index != -1 and args.end_index != -1:
        dataset = dataset[args.start_index : args.end_index+1]

    covered_indices = set()
    if not os.path.exists(args.output_path):
        open(args.output_path, "w").close()
    else:
        existing_preds = read_jsonl(args.output_path)
        covered_indices = set(rec['input']['index'] for rec in existing_preds)
        # if skip_index_till > 0:
        print(f"Skipping {len(covered_indices)} covered indices")

    pending_data = [rec for rec in dataset if rec['index'] not in covered_indices]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor, open(args.output_path, "a") as f_out:
        
        futures = {executor.submit(generate_response, rec, model_ckpt, sample_parameters) for rec in pending_data}

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
            except Exception as e:
                print("Future exception:", e)
                # import pdb; pdb.set_trace()
            # print(result)
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

if __name__ == "__main__":
    main(args)