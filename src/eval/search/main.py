import os
import sys
import copy
import json
import pathlib
import argparse
import requests
import numpy as np
import editdistance
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
sys.path.append(module_path)

from typing import Union
from src.eval.utils import read_jsonl
from src.eval.prompts import prompt_v1 as STEP_AGNOSTIC_PROMPT
from src.eval.prompts import prompt_step as STEP_AWARE_PROMPT
from src.metrics.functional_correctness import extract_program_str_from_last_python_block, try_ast_literal_eval, ProgramBFCCNode, ProgramVocabulary, identity

# from src.eval.models.run_vllm_inference_v2 import

import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Run inference with different settings."
    )

    # === File paths ===
    paths = parser.add_argument_group("File paths")
    paths.add_argument('--model_ckpt', type=str, required=True, help="Model checkpoint used for inference.")
    paths.add_argument('--input_path', type=str, required=True, help="Path to prompts/input dataset.")
    paths.add_argument('--output_path', type=str, required=True, help="Path to output file.")
    parser.add_argument('--start_index', default=-1, type=int, help='ending index for experiments/data (inclusive)')
    parser.add_argument('--end_index', default=-1, type=int, help='starting index for experiments/data (inclusive)')

    # === Prompting & reasoning ===
    reasoning = parser.add_argument_group("Prompting and reasoning")
    reasoning.add_argument("--prompt_template", choices=["step_aware", "step_agnostic"], default="step_aware", help="Step aware/agnostic inference prompting.")
    reasoning.add_argument('--max_prog_len', type=int, default=3, help="Maximum size of alpha (α) and beta (β) for replace programs.")
    reasoning.add_argument("--vocab", type=str, default="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", help="Vocabulary characters.")
    reasoning.add_argument("--no_think", action="store_true", help="Disable chain-of-thought reasoning globally.")

    # === Inference ===
    inference = parser.add_argument_group("Inference settings")
    inference.add_argument("--port", type=int, default=8002, help="Port where vLLM server is running.")
    inference.add_argument("--num_workers", type=int, default=12, help="Number of parallel workers for querying vLLM.")
    inference.add_argument('--max_new_tokens', type=int, default=8192, help="Maximum number of new tokens to generate.")
    inference.add_argument('--temperature', type=float, default=0.0, help="Sampling temperature.")
    inference.add_argument('--top_k', type=int, default=50, help="Top-k sampling parameter.")
    inference.add_argument('--top_p', type=float, default=0.95, help="Top-p (nucleus) sampling parameter.")

    # === Beam/cascade search ===
    beam = parser.add_argument_group("Beam/cascade search")
    beam.add_argument('--num_samples', type=int, default=1, help="Number of samples per instance (sampling budget).")
    beam.add_argument('--program_num', type=int, default=20, help="Max possible size of the entire ground truth cascade.")
    beam.add_argument('--step_size', type=int, default=10, help="Size of intermediate cascade/search steps.")
    beam.add_argument('--num_steps', type=int, default=1, help="Number of search steps.")
    beam.add_argument('--reward_fn', choices=['edit_sim', 'soft_pass'], default='soft_pass', help='reward function to be used for picking the best cascade after each search step.')

    return parser.parse_args()

args = get_args()
assert args.num_samples >= 1, "num samples can only be 1 or more"
vocab_chars = args.vocab
# vLLM server details
PORT = args.port
VLLM_SERVER_URL = f"http://0.0.0.0:{PORT}/v1/chat/completions"
MAX_NEW_TOKENS = args.max_new_tokens
NUM_WORKERS = args.num_workers

def edit_sim(pred_outputs: list[str], original_inputs: list[str], target_outputs: list[str]) -> float:
    pred_final_output_tok: list[str[str]] = [w.split() for w in pred_outputs] 
    inputs_tok: list[str[str]] = [w.split() for w in original_inputs] 
    targets_tok: list[str[str]] = [w.split() for w in target_outputs] 

    pred_target_dist = sum(editdistance.eval(s, t) for s, t in zip(pred_final_output_tok, targets_tok))
    input_target_dist = sum(editdistance.eval(s, t) for s, t in zip(inputs_tok, targets_tok))

    return (1 - (pred_target_dist / input_target_dist))

def soft_pass(pred_outputs: list[str], original_inputs: list[str], target_outputs: list[str]) -> float:
    assert len(pred_outputs) == len(target_outputs)
    return sum([pred_output == target_output for pred_output, target_output in zip(pred_outputs, target_outputs)])/len(pred_outputs)

PROMPT_TEMPLATE_MAP = {
    "step_aware": STEP_AWARE_PROMPT, 
    "step_agnostic": STEP_AGNOSTIC_PROMPT,
}

REWARD_FN_MAP = {
    'edit_sim': edit_sim, 
    'soft_pass': soft_pass,
}

def generate_response(rec, model_ckpt, sample_parameters):
    user_prompt = rec['prompt'] # directly use the pre-generated prompts in promptsfile.
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

    if sample_parameters['no_think']:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    response = requests.post(VLLM_SERVER_URL, json=payload)
    response.raise_for_status()
    response_json = response.json()
    model_response = response_json["choices"][0]["message"]
    rec["model_response"] = model_response['content']
    rec["model_cot"] = model_response.get('reasoning_content')

    return rec

def extract_programs_from_model_respose(model_response: str, max_programs: int):
    pred_cascade = try_ast_literal_eval(extract_program_str_from_last_python_block(model_response))
    program_seq = []
    for program in pred_cascade[:max_programs]: # pick only as many programs as permitted by the step_size.
        if isinstance(program, str):
            try: 
                program = ProgramBFCCNode.from_string(program, vocabulary=ProgramVocabulary(vocab_chars))
            except IndexError as e:
                # print(f"\x1b[31;1m{e}\x1b[0m")
                program = identity
            except AssertionError as e:
                # print(f"\x1b[31;1m{e}\x1b[0m")
                program = identity # replace invalid programs with identity programs.
        else: 
            # assert isinstance(program, ProgramBFCCNode), f"Invalid program type: {type(program)}"
            # print(f"\x1b[31;1mInvalid program type: {type(program)}. Replacing with identity. Check if this isn't the result of WRONG FORMATTING!!!\x1b[0m")
            program = identity
        program_seq.append(program)

    return program_seq

def generate_queries(
        dataset: list[dict], prior_responses: list[dict], 
        step_id: int=0, prompt_template: Union[str, None]=None, 
        num_samples: int=1, reward_fn=None, 
        program_num: int=-1, program_length: int=-1,
        vocab_chars: Union[str, None]=None,
        prompt_template_name: Union[str, None]=None,
    ) -> list[dict]: 
    """Generate (num_samples) queries for next iteration (step_id) of the search and use
    the (reward_fn) to pick the most promising cascade to build on top of."""
    assert program_num != -1, "set a valid program_num."
    assert program_length != -1, "set a valid program_length."
    assert vocab_chars is not None, "need to pass vocab_chars."
    assert prompt_template_name is not None, "need to pass prompt_template_name."

    dataset_id_to_prior_responses = defaultdict(lambda: [])
    dataset_id_to_best_response = {}
    if step_id != 0:
        for rec in prior_responses:
            dataset_id = rec['index'].split("_")[0].strip()
            rec['pred_cascade'] = extract_programs_from_model_respose(
                model_response=rec['model_response'],
                max_programs=program_num,
            )
            dataset_id_to_prior_responses[dataset_id].append(rec)
        for dataset_id, candidate_responses in dataset_id_to_prior_responses.items():
            reward_values = []
            for response in candidate_responses:
                pred_outputs = [i for i in response['curr_inputs']]
                for program in response['pred_cascade']:
                    pred_outputs = program(pred_outputs)
                response['pred_outputs'] = pred_outputs
                response['reward'] = reward_fn(pred_outputs, response['inputs'], response['outputs'])
                reward_values.append(response['reward'])
            max_reward_index = np.argmax(reward_values)
            best_response = candidate_responses[max_reward_index]
            dataset_id_to_best_response[dataset_id] = best_response       

    # add indices to all instances.
    queries = []
    for i in range(len(dataset)):
        for j in range(num_samples):
            rec = copy.deepcopy(dataset[i])
            rec['index'] = f"{i}_{j}_{step_id}"
            rec['step_index'] = step_id
            if step_id == 0:
                rec['curr_inputs'] = rec['inputs']
                rec['prev_best_index'] = None
                rec['prev_step_best_reward'] = 0
            else:
                best_cascade = dataset_id_to_best_response[f"{i}"]
                rec['curr_inputs'] = best_cascade['pred_outputs']
                rec['prev_best_index'] = best_cascade['index']
                rec['prev_step_best_reward'] = best_cascade['reward']
            if prompt_template_name == "step_aware":
                rec['prompt'] = prompt_template.format(
                    inputs_list=rec['curr_inputs'], # intermediate outputs from prev stage. 
                    outputs_list=rec['outputs'], 
                    program_num=args.program_num, 
                    program_length=program_length,
                    step_size=program_num,
                )
            elif prompt_template_name == "step_agnostic":
                rec['prompt'] = prompt_template.format(
                    inputs_list=rec['curr_inputs'], # intermediate outputs from prev stage. 
                    outputs_list=rec['outputs'], 
                    program_num=program_num, 
                    program_length=program_length,
                )
            queries.append(rec)

    return queries

def main(args):
    model_ckpt = args.model_ckpt
    sample_parameters = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "no_think": args.no_think,
    }
    if args.input_path.endswith('.jsonl'):
        dataset = read_jsonl(args.input_path)
    elif args.input_path.endswith('.json'):
        dataset = json.load(open(args.input_path))
    else:
        raise ValueError('Invalid input file format')

    if args.start_index != -1 and args.end_index != -1:
        dataset = dataset[args.start_index : args.end_index+1]

    covered_indices = set()
    if not os.path.exists(args.output_path):
        open(args.output_path, "w").close()
    else:
        existing_preds = read_jsonl(args.output_path)
        index_to_existing_preds = {rec['index']: rec for rec in existing_preds}
        covered_indices = set(rec['index'] for rec in existing_preds)
        # if skip_index_till > 0:
        print(f"Skipping {len(covered_indices)} covered indices")
    
    responses = []
    for step_index in range(args.num_steps):
        queries = generate_queries(
            dataset=dataset, prior_responses=responses, 
            step_id=step_index, prompt_template=PROMPT_TEMPLATE_MAP[args.prompt_template],
            num_samples=args.num_samples, reward_fn=REWARD_FN_MAP[args.reward_fn], 
            program_num=args.step_size, program_length=args.max_prog_len,
            vocab_chars=vocab_chars, prompt_template_name=args.prompt_template,
        )
        pending_queries, responses = [], []
        for rec in queries:
            if rec['index'] not in covered_indices:
                pending_queries.append(rec)
            else: responses.append(index_to_existing_preds[rec['index']])

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor, open(args.output_path, "a") as f_out:
            futures = {executor.submit(generate_response, rec, model_ckpt, sample_parameters) for rec in pending_queries}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"step {step_index+1}/{args.num_steps}"):
                result = future.result()
                responses.append(result)
                # print(result)
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()

# main
if __name__ == "__main__":
    main(args)
    