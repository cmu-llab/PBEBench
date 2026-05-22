import os
import re
import ast
import sys
import copy
import json
import pathlib
import numpy as np
from tqdm import tqdm
from collections import defaultdict

module_path = str(pathlib.Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from typing import Union
from src.data_generation.utils import read_jsonl
from src.data_generation.generate import write_jsonl
from src.eval.prompts import prompt_v1 as PROMPT_TEMPLATE
from src.data_generation.primitives import ProgramBFCCNode, ProgramVocabulary
from src.metrics.functional_correctness import extract_last_python_block, eval_outputs, reward

def evaluate_word_mismatches(inputs, outputs, pred_outputs):
    unchanged_words_changed = 0
    unchanged_words_unchanged = 0
    changed_words_correctly = 0
    changed_words_incorrectly = 0
    for i,o,po in zip(inputs, outputs, pred_outputs):
        if i == o:
            if o == po: unchanged_words_unchanged += 1
            else: unchanged_words_changed += 1
        else:
            if o != po: changed_words_incorrectly += 1
                # print(i, o, po)
            else: changed_words_correctly += 1
    assert unchanged_words_unchanged + unchanged_words_changed + changed_words_correctly + changed_words_incorrectly == len(inputs), "Mismatch in total counts of word evaluations."
    # print(f"Unchanged words: {unchanged_words_unchanged} unchanged, {unchanged_words_changed} changed")
    # print(f"Changed words: {changed_words_correctly} correct, {changed_words_incorrectly} incorrect")

    return {
        "unchanged": {
            "correct": unchanged_words_unchanged,
            "incorrect": unchanged_words_changed,
        },
        "changed": {
            "correct": changed_words_correctly, 
            "incorrect": changed_words_incorrectly,
        },
        "correct": unchanged_words_unchanged + changed_words_correctly,
        "incorrect": changed_words_incorrectly + unchanged_words_changed,
    }

def try_ast_literal_eval_verbose(string): # do some minor fixes.
    """Takes program sequence string and returns the extracted sequence of actual replace programs."""
    program_seq = ast.literal_eval(string)
    if isinstance(program_seq, str):
        program_seq = [program_seq] 
    
    return program_seq
    # try:
    #     program_seq = ast.literal_eval(string)
    #     if isinstance(program_seq, str):
    #         program_seq = [program_seq] 
    #     # print(program_seq, type(program_seq))
    #     try: 
    #         assert isinstance(program_seq, list), f"program sequence of list form couldn't be extracted or derived: {program_seq}"
    #         return program_seq
    #     except AssertionError: return []
    # except SyntaxError: return [] # empty prediction if no code blocks found.
def pick_highest_reward_prog_seq(programs: list[str], inputs: list[str], outputs: list[str], max_programs: int=5, vocab_chars=None):
    progs_and_metrics = []
    for index, prog_seq_string in enumerate(programs):
        pred_cascade = try_ast_literal_eval_verbose(prog_seq_string)
        pred_outputs, valid_programs, invalid_programs = eval_outputs(inputs=inputs, outputs=outputs, pred_cascade=pred_cascade, vocab_chars=vocab_chars, max_programs=max_programs)
        # progs_and_rewards.append((pred_cascade[:max_programs], pred_outputs, reward(pred_outputs, inputs, outputs), index))
        progs_and_metrics.append({
            "trunc_programs": pred_cascade[:max_programs], 
            "pred_outputs": pred_outputs, 
            "reward": reward(pred_outputs, inputs, outputs), 
            "word_errors": evaluate_word_mismatches(inputs, outputs, pred_outputs),
            "index": index,
        })
    progs_and_metrics = sorted(progs_and_metrics, reverse=True, key=lambda x: x['reward']) # sort by reward.
    # print(progs_and_rewards)
    # exit()
    return progs_and_metrics[0]

def pick_best_prog_seq_by(
        programs: list[str], inputs: list[str], 
        outputs: list[str], max_programs: int=5, vocab_chars=None, 
        metric1: Union[str, None]="reward", metric2: Union[str, None]=None,
    ) -> list[dict]:
    progs_and_metrics = []
    for index, prog_seq_string in enumerate(programs):
        pred_cascade = try_ast_literal_eval_verbose(prog_seq_string)
        pred_outputs, valid_programs, invalid_programs = eval_outputs(inputs=inputs, outputs=outputs, pred_cascade=pred_cascade, vocab_chars=vocab_chars, max_programs=max_programs)
        word_errors = evaluate_word_mismatches(inputs, outputs, pred_outputs)
        progs_and_metrics.append({
            "trunc_programs": pred_cascade[:max_programs], 
            "pred_outputs": pred_outputs, 
            "reward": reward(pred_outputs, inputs, outputs), 
            "word_errors": word_errors,
            "word_errors_correct": word_errors['correct'],
            "word_errors_incorrect": word_errors['incorrect'],
            "index": index,
        })
    # print(f"Metrics for all program sequences: {[{metric: pm[metric], 'reward': pm['reward']} for pm in progs_and_metrics]}")
    # exit()
    # sort by a key/metric.
    if metric2 is None:
        progs_and_metrics = sorted(progs_and_metrics, reverse=True, key=lambda x: x[metric1])
    else:
        progs_and_metrics = sorted(progs_and_metrics, reverse=True, key=lambda x: (x[metric1], x[metric2]))

    return progs_and_metrics[0]

def extract_model_predictions_multi(raw_data: list[dict], extract_method) -> list[dict]: 
    loaded_data = []
    null_ctr = 0
    for rec in raw_data: 
        # extract the code‐block text
        rec["predictions"] = []
        response = rec['response']
        if isinstance(response, str): response = [response]
        elif response is None: response = ["replace('x','x')"]
        # elif isinstance(response, list): 
        #     print("LIST RESPONSE")
        #     exit()
        for resp in response:
            try: pred = extract_method(resp) 
            except TypeError as e:
                if resp is None:
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
            rec["predictions"].append(pred) 
            # print(pred)

        loaded_data.append(rec) 
    print(f"found {null_ctr} nulls")

    return loaded_data

def compose_response(resp: str, cot: str) -> str:
    return f"""<think>
{cot.strip()}
</think>

{resp}"""

# main
if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = os.path.splitext(input_path)[0] + "_filtered.jsonl"

    program_num = 30
    # vocab_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    vocab_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    expected_inputs = 50
    vocab = ProgramVocabulary(list(vocab_chars))
    data = read_jsonl(input_path)
    program_length = 3

    extracted_data = extract_model_predictions_multi(data, extract_method=extract_last_python_block)
    
    best_predictions = [pick_best_prog_seq_by(programs=rec['predictions'], inputs=rec['inputs'], outputs=rec['outputs'], max_programs=program_num, vocab_chars=vocab_chars, metric1="word_errors_correct", metric2="reward") for rec in extracted_data]
    print(best_predictions[0])
    print(best_predictions[1])
    # extracted_progs_and_metrics = [extract_all_programs(programs=rec['predictions'], inputs=rec['inputs'], outputs=rec['outputs'], max_programs=program_num, vocab_chars=vocab_chars) for rec in extracted_data]

    # print(best_predictions[0])
    # exit()
    successful_data = []
    for best_pred, rec in zip(best_predictions, data):
        reward_value = best_pred['reward']
        word_errors_correct = best_pred['word_errors_correct']
        pred_outputs = best_pred['pred_outputs']
        prog_seq = best_pred['trunc_programs']
        response_index = best_pred['index']
        # print(reward_value, len(rec['programs']))
        if reward_value == 1 or (len(rec['programs']) >= 15 and reward_value >= 0.9): # if the best prediction is correct, or if there are many programs and the best prediction is mostly correct, keep the record.
            # print(rec['outputs'] == best_pred[1]) 
            assert len(rec['chains_of_thought']) == len(rec['response']), "Corrupted aggregate response."
            chosen_response = rec['response'][response_index]
            cot_for_response = rec['chains_of_thought'][response_index]
            rec['response'] = compose_response(chosen_response, cot_for_response)
            # print(rec['response'])e
            # print(prog_seq)
            del rec['chains_of_thought']
            successful_data.append(rec)
    # with open(output_path, 'w') as f:
    print(len(successful_data), len(data))
    write_jsonl(successful_data, output_path)
    # print(inst_level_passing_progs)