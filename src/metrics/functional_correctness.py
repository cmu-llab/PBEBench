import os
import re
import ast
import sys
import json
import pathlib
import argparse
import numpy as np
import editdistance
from typing import Union
from collections import defaultdict

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
    # parser.add_argument(
    #     "--search_mode", action="store_true",
    #     help="flag to indicate model preds are a folder of search steps instead of single pred file."
    # )
    parser.add_argument(
        "--vocab", type=str,
        default="abcdefghijkuvwxyz",
        help="vocabulary to be used for inputs/programs",
    )
    parser.add_argument(
        "--max_programs", type=int, default=5,
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
        program_node = ProgramBFCCNode.from_string(program, vocabulary=ProgramVocabulary(vocab_chars))
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
                program = ProgramBFCCNode.from_string(program, vocabulary=ProgramVocabulary(vocab_chars))
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
        try: 
            program_seq = ast.literal_eval(string)
            if isinstance(program_seq, str):
                program_seq = [program_seq] 
            # print(program_seq, type(program_seq))
            try: 
                assert isinstance(program_seq, list), f"program sequence of list form couldn't be extracted or derived: {program_seq}"
                return program_seq
            except AssertionError: return []
        except ValueError: return []
    except SyntaxError: return [] # empty prediction if no code blocks found.

def get_instance_complexities(model_preds, max_programs: int, vocab_chars):
    instance_complexity = [[0 for _ in range(7)] for _ in range(len(model_preds))] # index 0-3: B,F,CB,CF counts, index 4: effective 
    pred_cascade_complexity_scores = [[] for _ in range(len(model_preds))] # score based on summing up the lengths of the alphas and betas.
    gt_cascade_complexity_scores = [0 for _ in range(len(model_preds))] # score based on summing up the lengths of the alphas and betas.

    for ind,rec in enumerate(model_preds):
        prev_outputs = rec['inputs']
        rec["skipped_programs"] = [] # skip degenerate programs that don't alter any inputs.

        for program_ind,program in enumerate(rec['programs']):
            program = program.replace("\\","")
            program_node = ProgramBFCCNode.from_string(program, vocabulary=ProgramVocabulary(vocab_chars))
            cascade_complexity_score = (len(program_node.predicate) + len(program_node.transform))
            gt_cascade_complexity_scores[ind] += cascade_complexity_score
            next_inputs = program_node(prev_outputs)
            # print(prev_outputs, program, next_inputs)
            if next_inputs == prev_outputs:
                rec["skipped_programs"].append(program_ind)
            prev_outputs = next_inputs
            rec['programs'][program_ind] = program

        for pred_ind,prediction in enumerate(rec['predictions']):
            pred_cascade_complexity_scores[ind] = [0 for _ in range(len(rec['predictions']))]
            pred_cascade = try_ast_literal_eval(prediction)
            for program_ind, program in enumerate(pred_cascade):
                try: program = program.replace("\\","")
                except AttributeError: program = "" 
                try: 
                    program_node = ProgramBFCCNode.from_string(program, vocabulary=ProgramVocabulary('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-=1234567890 §{}[]|\?"/?><,.'), max_window_size=20) # deliberately add a very huge vocab so we can count the true complexity.
                    cascade_complexity_score = (len(program_node.predicate) + len(program_node.transform))
                    pred_cascade_complexity_scores[ind][pred_ind] += cascade_complexity_score
                except AssertionError:
                    pred_cascade_complexity_scores[ind][pred_ind] += 0 # invalid programs get a zero score.

        # edit distance complexity feature not affected by the skipped_programs.
        instance_complexity[ind][5] = np.mean([normalized_levenshtein(i,o) for i,o in zip(rec["inputs"], rec["outputs"])])
        # effective cascade length after skipping degenerate programs.
        instance_complexity[ind][4] = len(rec['programs'])-len(rec["skipped_programs"]) # effective cascade length
        # no. of words that have changed.
        instance_complexity[ind][6] = sum((i != o) for i,o in zip(rec['inputs'], rec['outputs']))/5

        # effective relation counts after removing degenerate programs.
        for link in rec["bfcc_dag"]:
            if link[0] in rec["skipped_programs"] or link[-1] in rec["skipped_programs"]: continue
            if link[1][0] == "N": pass
            elif link[1][0] == "B" and link[0] < link[-1]: # bleeding.
                instance_complexity[ind][0] += 1
            elif link[1][0] == "F" and link[0] < link[-1]: # feeding.
                instance_complexity[ind][1] += 1
            elif link[1][0] == "B" and link[0] > link[-1]: # counter-bleeding.
                instance_complexity[ind][2] += 1
            elif link[1][0] == "F" and link[0] > link[-1]: # counter-feeding.
                instance_complexity[ind][3] += 1

    return instance_complexity, gt_cascade_complexity_scores, pred_cascade_complexity_scores

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
    
    vocab_chars = args.vocab
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
        instance_complexities, gt_cascade_complexity_scores, pred_cascade_complexity_scores = get_instance_complexities(data, max_programs=args.max_programs, vocab_chars=vocab_chars)
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

        # sampled_predictions = [[ast.literal_eval(pred) for pred in rec['prediction']] for rec in data] 
        # pass_at_1 = eval_mean_pass_at_k(data, sampled_predictions, k=1) 
        # pass_at_3 = eval_mean_pass_at_k(data, sampled_predictions, k=3)
        # pass_at_5 = eval_mean_pass_at_k(data, sampled_predictions, k=5)
        # pass_at_10 = eval_mean_pass_at_k(data, sampled_predictions, k=10)
        
        print(extraction_type)
        print(f"pass@1: {mean_pass_at_1:.4f}")
        print(f"reward@1: {reward_at_1:.4f}")
        print(f"valid program rate: {valid_program_rate:.4f}")
        print(f"gt complexity: {np.mean(gt_cascade_complexity_scores):.2f}")
        print(f"pred complexity: {np.mean([safe_mean(p) for p in pred_cascade_complexity_scores]):.2f}")
        
        perf_by_cascade_length = defaultdict(lambda: [])
        perf_by_num_relns = defaultdict(lambda: [])
        perf_by_num_uniq_relns = defaultdict(lambda: [])
        perf_by_io_dist = defaultdict(lambda: [])

        cascade_length_dist = defaultdict(lambda: 0)
        num_relns_dist = defaultdict(lambda: 0)
        io_dist = defaultdict(lambda: 0)
        
        for i in range(len(data)):
            effective_cascade_length = instance_complexities[i][4]
            num_relns = sum(instance_complexities[i][:4])
            num_uniq_relns = sum([int(rel_count > 0) for rel_count in instance_complexities[i][:4]])
            io_distance = instance_complexities[i][5]
            
            perf_by_cascade_length[effective_cascade_length].append(inst_level_passing_progs[i])
            perf_by_num_relns[num_relns].append(inst_level_passing_progs[i])
            perf_by_num_uniq_relns[num_uniq_relns].append(inst_level_passing_progs[i])
            perf_by_io_dist[classify_iod(io_distance)].append(inst_level_passing_progs[i])
            # assert effective_cascade_length == instance_complexities[i][4]
            cascade_length_dist[effective_cascade_length] += 1
            num_relns_dist[num_relns] += 1
            io_dist[classify_iod(io_distance)] += 1

        perf_by_cascade_length = dict(perf_by_cascade_length)
        for k in perf_by_cascade_length:
            perf_by_cascade_length[k] = (round(sum(perf_by_cascade_length[k])/len(perf_by_cascade_length[k]),4),len(perf_by_cascade_length[k]))
        for k in perf_by_num_relns:
            perf_by_num_relns[k] = (round(sum(perf_by_num_relns[k])/len(perf_by_num_relns[k]),4),len(perf_by_num_relns[k]))
        for k in perf_by_io_dist:
            perf_by_io_dist[k] = (round(sum(perf_by_io_dist[k])/len(perf_by_io_dist[k]),4),len(perf_by_io_dist[k]))
        for k in perf_by_num_uniq_relns:
            perf_by_num_uniq_relns[k] = (round(sum(perf_by_num_uniq_relns[k])/len(perf_by_num_uniq_relns[k]),4),len(perf_by_num_uniq_relns[k]))

        cascade_length_dist = dict(cascade_length_dist)
        num_relns_dist = dict(num_relns_dist)
        io_dist = dict(io_dist)
    
        perf_by_cascade_length = dict(perf_by_cascade_length)
        perf_by_num_relns = dict(perf_by_num_relns)
        perf_by_io_dist = dict(perf_by_io_dist)
        
        print("Perf. by Cascade Length:", sort_dict_by_key(perf_by_cascade_length))
        print("Perf. by # Relations:", sort_dict_by_key(perf_by_num_relns))
        print("Perf. by # Unique Relations (e.g 1101 category = 3):", sort_dict_by_key(perf_by_num_uniq_relns))
        print("Perf. by IOD:", {k: perf_by_io_dist.get(k) for k in ['IOD < 0.2', '0.2 <= IOD < 0.4', '0.4 <= IOD < 0.6', '0.6 <= IOD < 0.8']})

        # print(cascade_length_dist)
        # print(num_relns_dist)
        # print(io_dist)

    # print("pass@3:", pass_at_3) 
    # print("pass@5:", pass_at_5)
    # print("pass@10:", pass_at_10)