import os
import re
import sys
import json
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.data_generation.utils import read_jsonl

def build_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--preds_path",
        type=str, required=True,
        help="Path to the predictions file"
    )
    parser.add_argument(
        "--prompts_path",
        type=str, default="data/clutrr_v1_all_test_promptsfile.json",
        help="Path to the dataset of the input prompts"
    )
    parser.add_argument(
        "--parse_cot",
        action="store_true",
        help="Whether to parse chain-of-thought in the predictions"
    )
    
    return parser

def strip_think_tokens(text: str) -> str:
    """
    Remove <think>...</think> blocks from a model response string.
    Returns the cleaned text with leading/trailing whitespace trimmed.
    """
    # # Case 1: Remove <think>...</think> blocks
    # text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # # Case 2: Remove reasoning that ends with </think> (missing opening tag)
    # # Matches:  (anything up to </think>)
    # text = re.sub(r".*?</think>", "", text, flags=re.DOTALL)

    # return text.strip()
    end = text.rfind("</think>")
    if end != -1:
        text = text[end + len("</think>"):]
    return text.strip()

# main
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    
    preds_path = args.preds_path
    parse_cot = args.parse_cot
    prompts_path = args.prompts_path
    clutrr_prompts = json.load(open(prompts_path))
    clutrr_prompts_by_index = {}
    for index in range(len(clutrr_prompts)):
        clutrr_prompts_by_index[index] = clutrr_prompts[index]

    clutrr_preds = read_jsonl(preds_path)
    matches = []
    matches_by_split = defaultdict(lambda: [])
    non_null_matches = []
    non_null_matches_by_split = defaultdict(lambda: [])
    null = 0
    for i in range(len(clutrr_preds)):
        true = clutrr_preds[i]['input']['target_text'].strip()
        index = clutrr_preds[i]['input']['index']
        pred = clutrr_preds[i]['outputs'][0]
        # task_id = clutrr_preds[i]['input']["task_name"]
        split_name = clutrr_prompts_by_index[index]['split_name']
        if pred is not None:
            pred = pred.strip()
            # strip CoT in case there is some issue with the reasoning parser (mainly for qwen models.)
            if parse_cot: pred = strip_think_tokens(pred)
            matches.append(int(true == pred))
            non_null_matches.append(int(true == pred))
            matches_by_split[split_name].append(int(true == pred))
            non_null_matches_by_split[split_name].append(int(true == pred))
        else: 
            null += 1
            matches.append(0)
            matches_by_split[split_name].append(0)
    acc = np.mean(matches)
    non_null_acc = np.mean(non_null_matches)
    print(len(matches))
    print(f"nulls: {null}")
    print(f"acc: {100*acc:.2f}%")
    print(f"non null acc: {100*non_null_acc:.2f}%")
    print("------------------------------")
    for split in ["gen_train234_test2to10", "gen_train23_test2to10", "rob_train_clean_23_test_all_23", "rob_train_disc_23_test_all_23", "rob_train_irr_23_test_all_23", "rob_train_sup_23_test_all_23"]:
        if split not in matches_by_split: continue
        split_acc = np.mean(matches_by_split[split])
        split_non_null_acc = np.mean(non_null_matches_by_split[split])
        print(f"\x1b[34;1m{split}\x1b[0m\nacc: {100*split_acc:.2f}%\tnon null acc: {100*split_non_null_acc:.2f}%")