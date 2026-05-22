import os
import sys
import copy
import json
import pathlib
import argparse
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from transformers import AutoTokenizer

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
# print(module_path)
sys.path.append(module_path)

from src.eval.utils import prompt_construct, read_jsonl

def get_args():
    parser = argparse.ArgumentParser(description="Run inference with different settings.")
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--prompt_path', type=str, default="v1")
    parser.add_argument('--max_rules_num', type=int, default=5)
    parser.add_argument('--max_tokens_length', type=int, default=3)
    parser.add_argument('--character', type=str, default='[a, b, c]')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--num_samples', type=int, default=1, help="how many samples to draw per instance")
    # parser.add_argument("--model_name", type=str, required=True, help="which model is to be queried")
    parser.add_argument("--no_think", action="store_true", help="Disable chain-of-thought reasoning globally.")
    parser.add_argument("--port", type=int, default=8002, help="Port where vLLM server is being served")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of parallel threads/workers to be used for querying vLLM.")
    parser.add_argument("--thinking_budget", type=int, default=8192, help="Thinking budget for chain-of-thought reasoning.")

    return parser.parse_args()

args = get_args()
assert args.num_samples >= 1, "num samples can only be 1 or more"
# vLLM server details
PORT = args.port
VLLM_SERVER_URL = f"http://0.0.0.0:{PORT}/v1/chat/completions"
MAX_RETRIES = 5
MAX_SEQ_LEN = 32768
MAX_NEW_TOKENS = args.max_new_tokens
NUM_WORKERS = args.num_workers

def generate_response(rec, model_ckpt, sample_parameters):
    f = open("response.json", "w")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    # END_TOKEN = tokenizer.encode("<|return|>", add_special_tokens=False)[0]
    # THINKING_END_TOKEN = tokenizer.encode("<|end|>", add_special_tokens=False)[0]
    END_TOKEN = "<|return|>"
    THINKING_END_TOKEN = "<|end|>"
    state = "1" # Initialize state
    user_prompt = rec['prompt'] # directly use the pre-generated prompts in promptsfile.

    # if len(user_prompt) > MAX_SEQ_LEN - MAX_NEW_TOKENS:
    #     user_prompt = user_prompt[:15000] + user_prompt[-15000:]

    messages = [{"role": "user", "content": user_prompt}]

    # First generation until thinking budget
    payload_first_gen = {
        "model": model_ckpt,
        "messages": messages,
        "max_tokens": sample_parameters['thinking_budget'],
        "temperature": sample_parameters['temperature'],
        "top_p": sample_parameters['top_p'],
        "top_k": sample_parameters['top_k'],
        "seed": 42,
        # "output_tokens": True,
        "logprobs": 1
    }
    response_first_gen = requests.post(VLLM_SERVER_URL, json=payload_first_gen)
    response_first_gen.raise_for_status()
    response_json_first_gen = response_first_gen.json()
    model_response_first_gen = response_json_first_gen["choices"][0]["message"]
    first_gen_content = model_response_first_gen.get("content") or ""
    first_gen_reasoning = model_response_first_gen.get("reasoning_content") or ""
    first_gen_tokens = [entry["token"] for entry in response_json_first_gen["choices"][0]["logprobs"]["content"]]

    # first_gen_tokens = {}
    # print("first_gen_tokens:")
    # print(first_gen_tokens)

    total_token_count_resp1 = response_json_first_gen["usage"]["completion_tokens"]
    total_token_count_resp2 = 0
    max_tokens_total = sample_parameters['max_new_tokens'] # Total max tokens for both runs


    # Combine all content from first generation
    gen1_all = (first_gen_reasoning or "") + (first_gen_content or "")

    final_output = {}

    # Early return if END_TOKEN is in first generation content
    reasoning_token_count_resp1 = 0
    reasoning_token_count_resp2 = 0
    if END_TOKEN in first_gen_tokens:
        reasoning_token_count_resp1 = first_gen_tokens.index(END_TOKEN)
        state = "1"
        # content_to_return = first_gen_content.replace(END_TOKEN, "").strip("\n")
        content_to_return = first_gen_content
        thinking_content_to_return = first_gen_reasoning # Use first_gen_reasoning as thinking content
        print(f"STATE: {state}, RESP1: {first_gen_content}, RESP2: NULL")
        final_output = {
            "input": rec,
            "chains_of_thought": thinking_content_to_return,
            "outputs": content_to_return
        }
    else:
        reasoning_token_count_resp1 = total_token_count_resp1
        early_stop_instruction = (
            "\nConsider the thinking token budget, I will not generate any more reasoning tokens, and provide the final answer directly.\n" + "<|end|><|start|>assistant<|channel|>final<|message|>"
        )
        
        # Check if </think> is in the first generation's content
        if THINKING_END_TOKEN in first_gen_tokens:
            # print("3")
            messages_for_second_gen = messages + [{"role": "assistant", "content": gen1_all}]
            state = "2"
        else:
            # print("thinking budget is reached, truncating CoT")
            # Truncate CoT and add early stopping text
            truncated_cot_messages = messages + [{"role": "assistant", "content": gen1_all + early_stop_instruction}]
            messages_for_second_gen = truncated_cot_messages
            state = "3"

        # Second generation for final output
        max_tokens_second_gen = max(0, max_tokens_total - total_token_count_resp1)
        payload_second_gen = {
            "model": model_ckpt,
            "messages": messages_for_second_gen,
            "max_tokens": max_tokens_second_gen,
            "temperature": sample_parameters['temperature'],
            "top_p": sample_parameters['top_p'],
            "top_k": sample_parameters['top_k'],
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False}, # Disable thinking for the second gen
            "logprobs": 1
        }
        # reasoning_token_count_resp2
        response_second_gen = requests.post(VLLM_SERVER_URL, json=payload_second_gen)
        # print("Second generation")
        # data_temp = response_second_gen.json()
        # print(json.dump(data_temp, f, indent=2))
        response_second_gen.raise_for_status()
        response_json_second_gen = response_second_gen.json()
        model_response_second_gen = response_json_second_gen["choices"][0]["message"]
        full_response_content = model_response_second_gen.get('content') or ""

        second_gen_reasoning = model_response_second_gen.get("reasoning_content") or ""
        second_gen_content = model_response_second_gen.get("content") or ""
        
        print(f"STATE: {state}, RESP1: {first_gen_content}, RESP2: {full_response_content}")
        if state == "3":
            # print("First generation")
            # data_temp1 = response_first_gen.json()
            # print(json.dumps(data_temp1, indent=2))

            print("Second generation")
            data_temp2 = response_second_gen.json()
            print(json.dumps(data_temp2, indent=2))

        total_token_count_resp2 = response_json_second_gen["usage"]["completion_tokens"]
        
        # Merge outputs
        combined_reasoning = (first_gen_reasoning + "\n" + second_gen_reasoning).strip()
        combined_content   = (first_gen_content + "\n" + second_gen_content).strip()

        final_output = {
            "input": rec,
            "chains_of_thought": combined_reasoning,
            "outputs": combined_content
        }
    if total_token_count_resp1 + total_token_count_resp2 > sample_parameters['max_new_tokens']:
        raise ValueError("More than expected used, exiting")
    return final_output

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
        "thinking_budget": args.thinking_budget,
    }
    if args.input_path.endswith('.jsonl'):
        base_dataset = read_jsonl(args.input_path)
    elif args.input_path.endswith('.json'):
        base_dataset = json.load(open(args.input_path))
    else:
        raise ValueError('Invalid input file format')
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
                # print(result)
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()
            except Exception as e:
                print(f"❌ Error in worker: {e}")
                for f in futures:
                    f.cancel()
                sys.exit(1)   # exit whole script immediately

if __name__ == "__main__":
    main(args)
