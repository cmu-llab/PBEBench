import os
import re
import ast
import sys
import json
import string
import pathlib
import argparse
import numpy as np
import editdistance
from typing import Union
from collections import defaultdict

# script to evaluate functional correctness for real SLI tasks where we do not have the ground truth cascade at all.

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.data_generation.primitives import ProgramBFCCNode, ProgramVocabulary

def sort_dict_by_key(d: dict, reverse=False):
    return dict(sorted(d.items(), key=lambda x: x[0], reverse=reverse))

def safe_mean(x):
    if len(x) == 0: return 0
    return np.mean(x)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate functional correctness of model generated program sequence.")
    parser.add_argument(
        "--model_predictions_path", type=str,
        help="Path to the model predictions file."
    )
    parser.add_argument(
        "--max_programs", type=int, default=50,
        help="Maximum programs the model is allowed to use.",
    )
    return parser.parse_args()

def classify_iod(iod: float):
    if iod < 0.2: return "IOD < 0.2"
    elif iod < 0.4: return "0.2 <= IOD < 0.4"
    elif iod < 0.6: return "0.4 <= IOD < 0.6"
    elif iod < 0.8: return "0.6 <= IOD < 0.8"
    else: return "0.8 <= IOD"

def reward(pred_final_output: list[str], inputs: list[str], targets: list[str]) -> float:
    pred_final_output_tok: list[str[str]] = [w.split() for w in pred_final_output] 
    inputs_tok: list[str[str]] = [w.split() for w in inputs] 
    targets_tok: list[str[str]] = [w.split() for w in targets] 

    pred_target_dist = sum(editdistance.eval(s, t) for s, t in zip(pred_final_output_tok, targets_tok))
    input_target_dist = sum(editdistance.eval(s, t) for s, t in zip(inputs_tok, targets_tok))

    return (1 - (pred_target_dist / input_target_dist))

def identity(x: list[str]) -> list[str]: return x

def find_degenerate_program_ids(input_dict: dict):
    prev_outputs = input_dict["inputs"]
    skipped_programs = [] # skip degenerate programs that don't alter any inputs.
    for program_ind,program in enumerate(input_dict['programs']):
        program = program.replace("\\","")
        program_node = ProgramBFCCNode.from_string(program, vocabulary=ProgramVocabulary(vocab_chars), max_window_size=5)
        next_inputs = program_node(prev_outputs)
        # print(prev_outputs, program, next_inputs)
        if next_inputs == prev_outputs:
            skipped_programs.append(program_ind)
        prev_outputs = next_inputs
        input_dict["programs"][program_ind] = program # just to fix the issue with exccess back slashes

    return skipped_programs

def extract_program(program: Union[str, ProgramBFCCNode], vocab_chars) -> ProgramBFCCNode:
    if isinstance(program, str):
        try: 
            program = ProgramBFCCNode.from_string(program, vocabulary=ProgramVocabulary(vocab_chars), max_window_size=5)
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

    return program

def eval_outputs(
        inputs: list[str], outputs: list[str], 
        pred_cascade: list[Union[str, ProgramBFCCNode]], 
        vocab_chars: str="abcdefghijkuvwxyz", max_programs: int=5):
    pred_outputs: list[str] = [i for i in inputs]
    valid_programs = 0
    invalid_programs = 0
    for program in pred_cascade[:max_programs]: # we should only pick the first max_programs (m in the paper) programs predicted by the model in the cascade (in case the model disobeys the instruction and uses more programs than max_programs)
        if isinstance(program, str):
            try: 
                program = ProgramBFCCNode.from_string(program, vocabulary=ProgramVocabulary(vocab_chars), max_window_size=5)
                valid_programs += 1
            except IndexError as e:
                # print(f"\x1b[31;1m{e}\x1b[0m")
                program = identity
                invalid_programs += 1
            except AssertionError as e:
                # print(f"\x1b[31;1m{e}\x1b[0m")
                program = identity # replace invalid programs with identity programs.
                invalid_programs += 1
        else: 
            # assert isinstance(program, ProgramBFCCNode), f"Invalid program type: {type(program)}"
            # print(f"\x1b[31;1mInvalid program type: {type(program)}. Replacing with identity. Check if this isn't the result of WRONG FORMATTING!!!\x1b[0m")
            program = identity
            invalid_programs += 1
        pred_outputs: list[str] = program(pred_outputs)
    
    return pred_outputs, valid_programs, invalid_programs

def eval_mean_pass_at_1(dataset, predictions: list[str], vocab_chars: str="abcdefghijkuvwxyz", max_programs: int=5):
    # passing_programs = 0
    passing_programs = []
    for rec, pred_cascade in zip(dataset, predictions):
        inputs = rec["inputs"]
        outputs = rec["outputs"]
        pred_outputs, valid_programs, invalid_programs = eval_outputs(inputs=inputs, outputs=outputs, pred_cascade=pred_cascade, vocab_chars=vocab_chars, max_programs=max_programs)
        test_pass_ratio = sum([int(o == po) for o,po in zip(outputs, pred_outputs)])/len(outputs)
        if test_pass_ratio == 1: passing_programs.append(1)
        else: passing_programs.append(0)

    return np.mean(passing_programs), passing_programs

def eval_rewards(dataset, predictions: list[str], vocab_chars: str="abcdefghijkuvwxyz", max_programs: int=5):
    rewards = []
    total_valid_programs = 0
    total_invalid_programs = 0
    for rec, pred_cascade in zip(dataset, predictions):
        inputs = rec["inputs"]
        outputs = rec["outputs"]
        pred_outputs, valid_programs, invalid_programs = eval_outputs(inputs=inputs, outputs=outputs, pred_cascade=pred_cascade, vocab_chars=vocab_chars, max_programs=max_programs)
        total_valid_programs += valid_programs
        total_invalid_programs += invalid_programs
        rewards.append(reward(pred_outputs, inputs, outputs))


    return np.mean(rewards), rewards, total_valid_programs/(total_invalid_programs+total_valid_programs)

def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

# def eval_mean_pass_at_k(dataset, predictions: list[list[str]], k: int=5, vocab_chars: str="abcdefghijkuvwxyz", max_programs: int=5):
#     n = len(predictions[0])
#     assert k <= n, f"k should be less than {n}"
#     instance_pass_at_ks = []
#     for rec, sampled_cascades in zip(dataset, predictions):
#         inputs = rec["inputs"]
#         outputs = rec["outputs"]
#         c = 0
#         for pred_cascade in sampled_cascades:
#             test_pass_ratio = eval_passing_test_cases(inputs=inputs, outputs=outputs, pred_cascade=pred_cascade, vocab_chars=vocab_chars, max_programs=max_programs)
#             if test_pass_ratio > 0.999: c += 1
#         instance_pass_at_k = pass_at_k(n, c, k)
#         instance_pass_at_ks.append(instance_pass_at_k)

#     return np.mean(instance_pass_at_ks)

def extract_first_python_block(markdown_text: str) -> str:
    """
    Extracts first Python code block from a markdown string.

    Args:
        markdown_text (str): The markdown text containing code blocks.

    Returns:
        List[str]: A list of Python code blocks.
    """
    code_block_pattern = r"```python\s(.*?)```"
    matches = re.findall(code_block_pattern, markdown_text, re.DOTALL)
    try: return [block.strip() for block in matches][0]
    except IndexError: return "[]"

def extract_last_python_block(markdown_text: str) -> str:
    """
    Extracts first Python code block from a markdown string.

    Args:
        markdown_text (str): The markdown text containing code blocks.

    Returns:
        List[str]: A list of Python code blocks.
    """
    code_block_pattern = r"```python\s(.*?)```"
    matches = re.findall(code_block_pattern, markdown_text, re.DOTALL)
    try: return [block.strip() for block in matches][-1]
    except IndexError: return "[]"

def extract_program_str_from_last_python_block(model_response: str) -> str:
    try: pred = extract_last_python_block(model_response) 
    except TypeError as e:
        if model_response is None: pass
        else: print(e)
        pred = "replace('x','x')"

    # wrap any replace(...) that isn’t already quoted in single quotes
    # e.g.  replace('kd','ka')  →  'replace('kd','ka')'
    pred = re.sub(
        r"(?<!['\"])(replace\([^)]*\))(?!['\"])",
        r'"\1"',
        pred
    )

    return pred

# def load_model_predictions(path: str):
#     raw_data = read_jsonl(path)
#     loaded_data = []
#     for rec in raw_data:
#         loaded_data.append(rec['input'])
#         loaded_data[-1]["prediction"] = extract_first_python_block(rec['output'])
#         print(loaded_data[-1]['prediction'])

#     return loaded_data

def load_model_predictions(path: str, extract_method) -> list[dict]: 
    raw_data: list[dict] = read_jsonl(path) 
    loaded_data = []
    for rec in raw_data: 
        # extract the code‐block text
        if isinstance(rec['output'], dict):
            rec['output'] = rec["output"]["text"]  # fix specific for QwQ file (should correct the prediction format for future)
        pred = extract_method(rec['output']) 

        # wrap any replace(...) that isn’t already quoted in single quotes
        # e.g.  replace('kd','ka')  →  'replace('kd','ka')'
        pred = re.sub(
            r"(?<!['\"])(replace\([^)]*\))(?!['\"])",
            r'"\1"',
            pred
        )

        # assign and print
        rec['input']["predictions"] = [pred]
        # print(pred)

        loaded_data.append(rec['input'])

    return loaded_data

def load_model_predictions_multi(path: str, extract_method) -> list[dict]: 
    raw_data: list[dict] = read_jsonl(path) 
    loaded_data = []
    null_ctr = 0
    for rec in raw_data: 
        # extract the code‐block text
        rec['input']["predictions"] = []
        for output in rec['outputs']: 
            try: pred = extract_method(output) 
            except TypeError as e:
                if output is None:
                    null_ctr += 1
                else: print(e)
                pred = "replace('x','x')"

            # wrap any replace(...) that isn’t already quoted in single quotes
            # e.g.  replace('kd','ka')  →  'replace('kd','ka')'
            pred = re.sub(
                r"(?<!['\"])(replace\([^)]*\))(?!['\"])",
                r'"\1"',
                pred
            )

            # assign and print
            rec['input']["predictions"].append(pred) 
            # print(pred)

        loaded_data.append(rec['input']) 
    print(f"found {null_ctr} nulls")

    return loaded_data 

def process_model_predictions_multi(raw_data: list[dict], extract_method) -> list[dict]: 
    loaded_data = []
    null_ctr = 0
    for rec in raw_data: 
        # extract the code‐block text
        rec['input']["predictions"] = []
        for output in rec['outputs']: 
            try: pred = extract_method(output) 
            except TypeError as e:
                if output is None:
                    null_ctr += 1
                else: print(e)
                pred = "replace('x','x')"

            # wrap any replace(...) that isn’t already quoted in single quotes
            # e.g.  replace('kd','ka')  →  'replace('kd','ka')'
            pred = re.sub(
                r"(?<!['\"])(replace\([^)]*\))(?!['\"])",
                r'"\1"',
                pred
            )

            # assign and print
            rec['input']["predictions"].append(pred) 
            # print(pred)

        loaded_data.append(rec['input']) 
    print(f"found {null_ctr} nulls")

    return loaded_data 

def normalized_levenshtein(s1: str, s2: str) -> float:
    distance = editdistance.eval(s1, s2)
    max_len = max(len(s1), len(s2))
    return distance / max_len if max_len > 0 else 0.0

def try_ast_literal_eval(string): # do some minor fixes.
    """Takes program sequence string and returns the extracted sequence of actual replace programs."""
    try:
        program_seq = ast.literal_eval(string)
        if isinstance(program_seq, str):
            program_seq = [program_seq] 
        # print(program_seq, type(program_seq))
        try: 
            assert isinstance(program_seq, list), f"program sequence of list form couldn't be extracted or derived: {program_seq}"
            return program_seq
        except AssertionError: return []
    except SyntaxError: return [] # empty prediction if no code blocks found.

def pick_highest_reward_prog_seq(programs: list[str], inputs: list[str], outputs: list[str], max_programs: int=5, vocab_chars=None):
    progs_and_rewards = []
    for prog_seq_string in programs:
        pred_cascade = try_ast_literal_eval(prog_seq_string)
        pred_outputs, valid_programs, invalid_programs = eval_outputs(inputs=inputs, outputs=outputs, pred_cascade=pred_cascade, vocab_chars=vocab_chars, max_programs=max_programs)
        progs_and_rewards.append((pred_cascade[:max_programs], reward(pred_outputs, inputs, outputs)))
    progs_and_rewards = sorted(progs_and_rewards, reverse=True, key=lambda x: x[1]) # sort by reward.
    # print(progs_and_rewards[0][0])

    return progs_and_rewards[0][0]    

def pick_highest_reward_outputs(programs: list[str], inputs: list[str], outputs: list[str], max_programs: int=5, vocab_chars=None):
    pred_outputs_and_rewards = []
    for prog_seq_string in programs:
        pred_cascade = try_ast_literal_eval(prog_seq_string)
        pred_outputs, valid_programs, invalid_programs = eval_outputs(inputs=inputs, outputs=outputs, pred_cascade=pred_cascade, vocab_chars=vocab_chars, max_programs=max_programs)
        pred_outputs_and_rewards.append((pred_outputs, reward(pred_outputs, inputs, outputs)))
    pred_outputs_and_rewards = sorted(pred_outputs_and_rewards, reverse=True, key=lambda x: x[1]) # sort by reward.
    # print(progs_and_rewards[0][0])

    return pred_outputs_and_rewards[0][0]    

# main
if __name__ == "__main__":
    from src.data_generation.utils import read_jsonl

    args = parse_args()
    
    vocab_chars = "".join(json.load(open("data/real_sli/vocab.json")))+string.ascii_lowercase+string.ascii_uppercase
    model_predictions_path = args.model_predictions_path
    metric_values_path = os.path.join("metric_values", pathlib.Path(model_predictions_path).stem+".json")

    for extraction_type,extract_method in {
        "extract first code block": extract_first_python_block,
        "extract last code block": extract_last_python_block,
    }.items():
        # try:
        data = load_model_predictions_multi(model_predictions_path, extract_method=extract_method)
        best_predictions = [pick_highest_reward_prog_seq(programs=rec['predictions'], inputs=rec['inputs'], outputs=rec['outputs'], max_programs=args.max_programs, vocab_chars=vocab_chars) for rec in data]
        # print([len(p) for p in best_predictions])
        # except KeyError:
        #     data = load_model_predictions(model_predictions_path, extract_method=extract_method)
        #     best_predictions = [try_ast_literal_eval(rec['predictions'][0]) for rec in data]
        # [try_ast_literal_eval(rec['prediction']) for rec in data]
        # exit() 
        # print(instance_complexities[0])

        mean_pass_at_1, inst_level_passing_progs = eval_mean_pass_at_1(data, best_predictions, vocab_chars=vocab_chars, max_programs=args.max_programs)
        reward_at_1, inst_level_rewards, valid_program_rate = eval_rewards(data, best_predictions, vocab_chars=vocab_chars, max_programs=args.max_programs)

        metric_values = []
        for passing, reward_val in zip(inst_level_passing_progs, inst_level_rewards):
            if passing == 1: assert reward_val == 1, f"reward: {reward_val}, passing: {passing}"
            if reward_val < 1: assert passing == 0
            metric_values.append({"passing": passing, "reward": reward_val})
        with open(metric_values_path, "w") as f:
            json.dump(metric_values, f, indent=4)
        
        print(extraction_type)
        print(f"pass@1: {mean_pass_at_1:.4f}")
        print(f"reward@1: {reward_at_1:.4f}")
        print(f"valid program rate: {valid_program_rate:.4f}")
        # print(cascade_length_dist)
        # print(num_relns_dist)
        # print(io_dist)

    # print("pass@3:", pass_at_3) 
    # print("pass@5:", pass_at_5)
    # print("pass@10:", pass_at_10)