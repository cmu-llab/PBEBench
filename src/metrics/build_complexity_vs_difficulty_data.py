import os
import sys
import json
import pathlib
import numpy as np
import editdistance
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict
from scipy.stats import spearmanr, kendalltau

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.data_generation.primitives import ProgramBFCCNode, ProgramVocabulary

def is_instance_difficult(ic: list):
    if sum(ic[:4]) >= 1: return 1
    if ic[4] >= 2: return 1
    return 0

def read_jsonl(path: str):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            data.append(json.loads(line))

    return data

def write_jsonl(data, path: str):
    with open(path, "w") as f:
        for rec in data:
            f.write(json.dumps(rec)+"\n")

MODEL_METRICS_DICT = {
    "codestral-22b": "metric_values/codestral_preds_cascaded_checked_balanced_program_transformations_dataset.json",
    # open source (qwens):
    "qwen2.5-32b-instruct": "metric_values/qwen2.5_32b_preds_cascaded_checked_balanced_program_transformations_dataset.json",
    "qwen2.5coder-32b-instruct": "metric_values/qwen2.5_32b_coder_preds_cascaded_checked_balanced_program_transformations_dataset.json",
    "qwq-32b": 'metric_values/qwq_preds_cascaded_checked_balanced_program_transformations_dataset.json',
    "qwen3-32b": "metric_values/qwen3_32b_preds_cascaded_checked_balanced_program_transformations_dataset_2048_max_token_length.json",
    "qwen3-30b-a3b": "metric_values/qwen3_30B_A3B_preds_cascaded_checked_balanced_program_transformations_dataset.json",
    # open source (deepseek):
    "deepseek-r1-distill-qwen-32b": "metric_values/deepseek_cascaded_checked_balanced_program_transformations_dataset_output.json",
    "deepseek-r1": "metric_values/deepseek_cascaded_checked_balanced_program_transformations_dataset_output.json",
    # closed source:
    "o3-mini": "metric_values/o3_mini_cascaded_checked_balanced_program_transformations_dataset.json",
    "o4-mini": "metric_values/o4_mini_cascaded_checked_balanced_program_transformations_dataset.json",
    "gemini-2.5-flash-preview": "metric_values/gemini_preds_cascaded_dataset.json",
    "claude-3.5-sonnet": "metric_values/claude_3.5_sonnet_preds_cascaded_checked_balanced_programs_transformations_dataset.json",
    "claude-3.7-sonnet": "metric_values/claude_3.7_sonnet_preds_cascaded_checked_balanced_programs_transformations_dataset.json",
    "claude-3.7-sonnet-thinking": "metric_values/claude_3.7_sonnet_preds_think_2048_12000_cascaded_checked_balanced_prgrams_transformations_dataset.json",
}

def normalized_levenshtein(s1: str, s2: str) -> float:
    distance = editdistance.eval(s1, s2)
    max_len = max(len(s1), len(s2))
    return distance / max_len if max_len > 0 else 0.0

def classify_pval(p_val: float):
    if p_val > 0.05: return "not significant"
    elif p_val > 0.01: return 'p < 0.05'
    elif p_val > 0.005: return 'p < 0.01'
    elif p_val > 0.001: return 'p < 0.005'
    elif p_val > 0.0005: return 'p < 0.0001'
    elif p_val > 0.0001: return 'p < 0.0005'
    else: return "p < 0.0001"

def classify_iod(iod: float):
    if iod < 0.2: return "IOD < 0.2"
    elif iod < 0.4: return "0.2 <= IOD < 0.4"
    elif iod < 0.6: return "0.4 <= IOD < 0.6"
    elif iod < 0.8: return "0.6 <= IOD < 0.8"
    else: return "0.8 <= IOD"

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

# main
if __name__ == "__main__":
    vocab_chars = "abcdefghijkuvwxyz"
    test_benchmark = read_jsonl("data/cascaded_checked_balanced_program_transformations_dataset.jsonl")
    # print(len(data))
    
    instance_complexity = [[0 for _ in range(7)] for _ in range(len(test_benchmark))] # index 0-3: B,F,CB,CF counts, index 4: effective cascade length, index 5: normalized edit distance.
    instance_difficulty_pass = [[] for _ in range(len(test_benchmark))] # d_pass from the paper
    instance_difficulty_edit = [[] for _ in range(len(test_benchmark))] # d_edit from the paper

    for ind,rec in enumerate(test_benchmark):
        prev_outputs = rec['inputs']
        rec["skipped_programs"] = [] # skip degenerate programs that don't alter any inputs.
        for program_ind,program in enumerate(rec['programs']):
            program = program.replace("\\","")
            program_node = ProgramBFCCNode.from_string(program, vocabulary=ProgramVocabulary(vocab_chars))
            next_inputs = program_node(prev_outputs)
            # print(prev_outputs, program, next_inputs)
            if next_inputs == prev_outputs:
                rec["skipped_programs"].append(program_ind)
            prev_outputs = next_inputs
            rec['programs'][program_ind] = program
        
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

    # print(rec["skipped_programs"])
    
    for model_name, metrics_path in MODEL_METRICS_DICT.items():
        metric_values = json.load(open(metrics_path))
        print(model_name)
        for i in range(len(test_benchmark)):
            instance_difficulty_pass[i].append(1-metric_values[i]['passing'])
            instance_difficulty_edit[i].append(1-metric_values[i]['reward'])
    
    for i in range(len(test_benchmark)):
        instance_difficulty_pass[i] = round(np.mean(instance_difficulty_pass[i]), 4)
        instance_difficulty_edit[i] = round(np.mean(instance_difficulty_edit[i]), 4)
    
    factorial_analysis_table_pass = []
    factorial_analysis_table_edit = []
    for inst_comp, inst_diff in zip(instance_complexity, instance_difficulty_pass):
        factorial_analysis_table_pass.append({
            "Bleeding Count": inst_comp[0] > 0,
            "Feeding Count": inst_comp[1] > 0,
            "Counter Bleeding Count": inst_comp[2] > 0,
            "Counter Feeding Count": inst_comp[3] > 0,
            "Cascade Length": inst_comp[4] > 0,
            "Input-Output Distance": inst_comp[5] > 0, 
            "Difficulty (pass)": inst_diff,
        })
    for inst_comp, inst_diff in zip(instance_complexity, instance_difficulty_edit):
        factorial_analysis_table_edit.append({
            "Bleeding Count": inst_comp[0] > 0,
            "Feeding Count": inst_comp[1] > 0,
            "Counter Bleeding Count": inst_comp[2] > 0,
            "Counter Feeding Count": inst_comp[3] > 0,
            "Cascade Length": inst_comp[4] > 0,
            "Input-Output Distance": inst_comp[5] > 0, 
            "Difficulty (edit)": inst_diff,
        })
    # print(factorial_analysis_table_edit)
    pd.DataFrame(factorial_analysis_table_edit).to_csv("factorial_analysis_table_diff_edit.csv", index=False)
    pd.DataFrame(factorial_analysis_table_pass).to_csv("factorial_analysis_table_diff_pass.csv", index=False)
    # print(instance_difficulty_pass[:10])
    # print(instance_difficulty_edit[:10])
    num_relns = []
    rel_types = []
    rel_type_id = ["B", "F", "CB", "CF"]
    for ic in instance_complexity:
        num_relns.append(sum([ic[k] > 0 for k in range(4)]))
        for k in range(4):
            if ic[k] > 0:
                rel_types.append(rel_type_id[k])
    print("# Relations Dist:", Counter([rel for rel in num_relns], return_counts=True))
    print("Relation Types Dist:", Counter(rel_types, return_counts=True))
    print("Cascade Dist:", Counter([ic[4] for ic in instance_complexity], return_counts=True))
    print("Input-Ouput Distance Dist:", Counter([classify_iod(ic[5]) for ic in instance_complexity], return_counts=True))
    print("Input-Output # Words Different:", Counter([ic[6] for ic in instance_complexity], return_counts=True))
    write_jsonl(test_benchmark, "data/cascaded_checked_balanced_program_transformations_dataset_with_skipped_programs.jsonl")

    # compute correlations

    features = ["B", "F", "CB", 'CF', "CL", "IOD", "NDI"]
    for i in range(6):
        X = [ic[i] for ic in instance_complexity]
        Y = instance_difficulty_pass
        r, p_val = spearmanr(X, Y)
        print(f"Spearman R: {features[i]} 𝛼 d_pass: {r:.2f} ({classify_pval(p_val)})")
        r, p_val = kendalltau(X, Y)
        print(f"Kendal Tau: {features[i]} 𝛼 d_pass: {r:.2f} ({classify_pval(p_val)})")

    
    # # correlations with any relation count
    # X = np.zeros(len(instance_complexity))
    # for i in range(4):
    #     X_rel = np.array([ic[i] for ic in instance_complexity])
    #     X += X_rel
    # Y = instance_difficulty_pass
    # r, p_val = spearmanr(X, Y)
    # print(f"Spearman R: (any reln) 𝛼 d_pass: {r:.2f} ({classify_pval(p_val)})")
    # r, p_val = kendalltau(X, Y)
    # print(f"Kendal Tau: (any reln) 𝛼 d_pass: {r:.2f} ({classify_pval(p_val)})")
    # X = [is_instance_difficult(ic) for ic in instance_complexity]
    # Y = instance_difficulty_pass
    # r, p_val = spearmanr(X, Y)
    # print(f"Spearman R: overall 𝛼 d_pass: {r:.2f} ({classify_pval(p_val)})")
    # r, p_val = kendalltau(X, Y)
    # print(f"Kendal Tau: overall 𝛼 d_pass: {r:.2f} ({classify_pval(p_val)})")

    print()

    for i in range(6):
        X = [ic[i] for ic in instance_complexity]
        Y = instance_difficulty_edit
        r, p_val = spearmanr(X, Y)
        print(f"Spearman R: {features[i]} 𝛼 d_edit: {r:.2f} ({classify_pval(p_val)})")
        r, p_val = kendalltau(X, Y)
        print(f"Kendal Tau: {features[i]} 𝛼 d_edit: {r:.2f} ({classify_pval(p_val)})")
    
    inst_diff_by_cl = defaultdict(lambda: [])
    inst_diff_by_relns = defaultdict(lambda: [])
    inst_diff_by_iod = defaultdict(lambda: [])
    for diff, ic in zip(instance_difficulty_pass, instance_complexity):
        # print(ic)
        inst_diff_by_cl[ic[4]].append(diff)
        inst_diff_by_iod[classify_iod(ic[5])].append(diff)
        inst_diff_by_relns[sum(ic[:4])].append(diff)
    inst_diff_by_cl = dict(inst_diff_by_cl)
    inst_diff_by_relns = dict(inst_diff_by_relns)
    inst_diff_by_iod = dict(inst_diff_by_iod)
    print({k: round(np.mean(v),2) for k,v in inst_diff_by_cl.items()})
    print({k: round(np.mean(v),2) for k,v in inst_diff_by_iod.items()})
    print({k: round(np.mean(v),2) for k,v in inst_diff_by_relns.items()})

    # # correlations with any relation count
    # X = np.zeros(len(instance_complexity))
    # for i in range(4):
    #     X_rel = np.array([ic[i] for ic in instance_complexity])
    #     X += X_rel
    # Y = instance_difficulty_edit
    # r, p_val = spearmanr(X, Y)
    # print(f"Spearman R: (any reln) 𝛼 d_edit: {r:.2f} ({classify_pval(p_val)})")
    # r, p_val = kendalltau(X, Y)
    # print(f"Kendal Tau: (any reln) 𝛼 d_edit: {r:.2f} ({classify_pval(p_val)})")


    # X = [sum(ic) for ic in instance_complexity]
    # Y = instance_difficulty_pass