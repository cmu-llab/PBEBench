from collections import defaultdict
from statistics import mean

import os
import re
import ast
import sys
import pathlib
import numpy as np
import editdistance
from typing import Union

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.data_generation.utils import read_jsonl
from src.data_generation.primitives import ProgramBFCCNode, ProgramVocabulary

# def reward(pred_final_output: list[str], inputs: list[str], targets: list[str]) -> float:
#     pred_final_output_tok: list[str[str]] = [w.split() for w in pred_final_output] # type: ignore
#     inputs_tok: list[str[str]] = [w.split() for w in inputs] # type: ignore
#     targets_tok: list[str[str]] = [w.split() for w in targets] # type: ignore

#     pred_target_dist = sum(editdistance.eval(s, t) for s, t in zip(pred_final_output_tok, targets_tok))
#     input_target_dist = sum(editdistance.eval(s, t) for s, t in zip(inputs_tok, targets_tok))

#     return (1 - (pred_target_dist / input_target_dist))

def reward(pred_final_output: list[str], inputs: list[str], targets: list[str]) -> float:
    return float(pred_final_output == targets)


def identity(x: list[str]) -> list[str]: return x

def eval_passing_test_cases(inputs: list[str], outputs: list[str], pred_cascade: list[Union[str, ProgramBFCCNode]], vocab_chars: str="abcdef"):
    pred_outputs: list[str] = [i for i in inputs]
    for program in pred_cascade:
        if isinstance(program, str):
            try: program = ProgramBFCCNode.from_string(program, vocabulary=ProgramVocabulary(vocab_chars))
            except IndexError as e:
                print(f"\x1b[31;1m{e}\x1b[0m")
                program = identity
            except AssertionError as e:
                print(f"\x1b[31;1m{e}\x1b[0m")
                program = identity # replace invalid programs with identity programs.
        else: # instead of halting eval, maybe keep iterating with default identity program
            # assert isinstance(program, ProgramBFCCNode), f"Invalid program type: {type(program)}"
            print(f"\x1b[31;1mInvalid program type: {type(program)}. Replacing with identity. Check if this isn't the result of WRONG FORMATTING!!!\x1b[0m")
            program = identity
        pred_outputs: list[str] = program(pred_outputs)
    
    return sum([o == po for o,po in zip(outputs, pred_outputs)])/len(outputs)

def eval_outputs(inputs: list[str], outputs: list[str], pred_cascade: list[Union[str, ProgramBFCCNode]], vocab_chars: str="abcdef"):
    pred_outputs: list[str] = [i for i in inputs]
    for program in pred_cascade:
        if isinstance(program, str):
            try: program = ProgramBFCCNode.from_string(program, vocabulary=ProgramVocabulary(vocab_chars))
            except IndexError as e:
                print(f"\x1b[31;1m{e}\x1b[0m")
                program = identity
            except AssertionError as e:
                print(f"\x1b[31;1m{e}\x1b[0m")
                program = identity # replace invalid programs with identity programs.
        else: 
            # assert isinstance(program, ProgramBFCCNode), f"Invalid program type: {type(program)}"
            print(f"\x1b[31;1mInvalid program type: {type(program)}. Replacing with identity. Check if this isn't the result of WRONG FORMATTING!!!\x1b[0m")
            program = identity
        pred_outputs: list[str] = program(pred_outputs)
    
    return pred_outputs

def eval_mean_pass_at_1(dataset, predictions: list[str], vocab_chars: str="abcdef"):
    passing_programs = 0
    for rec, pred_cascade in zip(dataset, predictions):
        inputs = rec["inputs"]
        outputs = rec["outputs"]
        test_pass_ratio = eval_passing_test_cases(inputs=inputs, outputs=outputs, pred_cascade=pred_cascade, vocab_chars=vocab_chars)
        if test_pass_ratio > 0.999: passing_programs += 1

    return passing_programs/len(dataset)

def eval_rewards(dataset, predictions: list[str], vocab_chars: str="abcdef"):
    rewards = []
    for rec, pred_cascade in zip(dataset, predictions):
        inputs = rec["inputs"]
        outputs = rec["outputs"]
        pred_outputs = eval_outputs(inputs=inputs, outputs=outputs, pred_cascade=pred_cascade, vocab_chars=vocab_chars)
        rewards.append(reward(pred_outputs, inputs, outputs))

    return np.mean(rewards)

def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def eval_mean_pass_at_k(dataset, predictions: list[list[str]], k: int=5, vocab_chars: str="abcdef"):
    n = len(predictions[0])
    assert k <= n, f"k should be less than {n}"
    instance_pass_at_ks = []
    for rec, sampled_cascades in zip(dataset, predictions):
        inputs = rec["inputs"]
        outputs = rec["outputs"]
        c = 0
        for pred_cascade in sampled_cascades:
            test_pass_ratio = eval_passing_test_cases(inputs=inputs, outputs=outputs, pred_cascade=pred_cascade, vocab_chars=vocab_chars)
            if test_pass_ratio > 0.999: c += 1
        instance_pass_at_k = pass_at_k(n, c, k)
        instance_pass_at_ks.append(instance_pass_at_k)

    return np.mean(instance_pass_at_ks)

def extract_first_python_block(markdown_text):
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

# def load_model_predictions(path: str):
#     raw_data = read_jsonl(path)
#     loaded_data = []
#     for rec in raw_data:
#         loaded_data.append(rec['input'])
#         loaded_data[-1]["prediction"] = extract_first_python_block(rec['output'])
#         print(loaded_data[-1]['prediction'])

#     return loaded_data

def load_model_predictions(path: str) -> list[dict]: # type: ignore
    raw_data: list[dict] = read_jsonl(path) # type: ignore
    loaded_data = []
    for i, rec in enumerate(raw_data): # type: ignore
        # extract the code‐block text
        if not isinstance(rec['output'], str):
            continue
        pred = extract_first_python_block(rec['output']) # type: ignore

        # wrap any replace(...) that isn’t already quoted in single quotes
        # e.g.  replace('kd','ka')  →  'replace('kd','ka')'
        pred = re.sub(
            r"(?<!['\"])(replace\([^)]*\))(?!['\"])",
            r'"\1"',
            pred
        )

        # assign and print
        rec['input']["prediction"] = pred
        # print(pred)

        loaded_data.append(rec['input']) # type: ignore

    return loaded_data # type: ignore

def load_model_predictions_multi(path: str) -> list[dict]: # type: ignore
    raw_data: list[dict] = read_jsonl(path) # type: ignore
    loaded_data = []
    for rec in raw_data: # type: ignore
        # extract the code‐block text
        rec['input']["predictions"] = []
        for output in rec['outputs']: # type: ignore
            pred = extract_first_python_block(output) # type: ignore

            # wrap any replace(...) that isn’t already quoted in single quotes
            # e.g.  replace('kd','ka')  →  'replace('kd','ka')'
            pred = re.sub(
                r"(?<!['\"])(replace\([^)]*\))(?!['\"])",
                r'"\1"',
                pred
            )

            # assign and print
            rec['input']["predictions"].append(pred) # type: ignore
            # print(pred)

        loaded_data.append(rec['input']) # type: ignore

    return loaded_data # type: ignore

def try_ast_literal_eval(string): # do some minor fixes.
    try:
        program_seq = ast.literal_eval(string)
        if isinstance(program_seq, str):
            program_seq = [program_seq] 
        # print(program_seq, type(program_seq))
        assert isinstance(program_seq, list), f"program sequence of list form couldn't be extracted or derived: {program_seq}"
        return program_seq
    except SyntaxError:
        return [] # empty prediction if no code blocks found.

def avg_reward_per_cascade_length(path: str, vocab_chars: str = "abcdefghijkuvwxyz"):
    data = load_model_predictions(path)
    rewards_by_length = defaultdict(list)

    for rec in data:
        cascade_length = rec.get("cascade_length")
        if cascade_length is None:
            cascade_length = len(rec.get("programs"))
        
        if cascade_length is None:
            continue
        print("Darsh 2 2 2 2")
        inputs = rec["inputs"]
        targets = rec["outputs"]
        pred_cascade = try_ast_literal_eval(rec["prediction"])
        pred_outputs = eval_outputs(inputs, targets, pred_cascade, vocab_chars)
        r = reward(pred_outputs, inputs, targets)
        rewards_by_length[cascade_length].append(r)

    avg_rewards = {
        length: mean(rewards) for length, rewards in sorted(rewards_by_length.items())
    }
    return avg_rewards

if __name__ == "__main__":
    path = "/home/darsha/Testing/pbe-reasoning/outputs/qwen3_32b_preds_cascaded_checked_balanced_program_transformations_dataset_2048_max_token_length.jsonl"
    result = avg_reward_per_cascade_length(path)
    print("Darsh")
    print(result)
    print("Average Reward per Cascade Length:")
    for length, avg in result.items():
        print(f"Cascade length {length}: {avg:.4f}")
