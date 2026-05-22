import os
import re
import sys
import json
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from typing import Union
from evaluate import load
from collections import defaultdict

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.data_generation.utils import read_jsonl

# def extract_last_code_block(text):
#     # Matches any fenced code block ```...```
#     pattern = r"```([^\n]*)\n(.*?)```"
#     blocks = re.findall(pattern, text, flags=re.DOTALL)
#     if not blocks:
#         return None

#     # blocks is a list of tuples: (language, content)
#     lang, content = blocks[-1]

#     return lang.strip(), content

def extract_last_code_block(text):
    # This pattern captures:
    #   group 1: optional language tag (possibly empty)
    #   group 2: the block content
    pattern = r"```(?:([\w+-]+))?\s*\n(.*?)```"
    blocks = re.findall(pattern, text, flags=re.DOTALL)

    if not blocks:
        return None

    lang, content = blocks[-1]
    lang = lang.strip() if lang else None
    return lang, content

def extract_prolog_program(output: Union[str, None], parse_cot: bool=False):
    if output is None: return "NULL"
    if parse_cot: output = strip_cot(output)
    try: lang, content = extract_last_code_block(output)
    except TypeError: 
        return output
    return content

def strip_cot(text: str) -> str:
    # 1. Remove <think>...</think> sections if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. Extract all fenced code blocks (``` ... ```)
    code_blocks = re.findall(r"```[\s\S]*?```", text)

    if code_blocks:
        last_block = code_blocks[-1]

        # Find where the last block starts
        start_index = text.rfind(last_block)

        # Keep from the last code block to the end (usually includes short explanation)
        cleaned = text[start_index:].strip()
        return cleaned

    # 3. If no fenced code blocks exist, fallback:
    #    Remove extra blank lines and return the last few lines.
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Heuristic: keep the last 5 non-empty lines (final answer region)
    return "\n".join(lines[-5:])

def fm(metric_value): # format metric
    return round(100*metric_value, 2)

def build_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--preds_path",
        type=str, required=True,
        help="Path to the predictions file"
    )
    # parser.add_argument(
    #     "--prompts_path",
    #     type=str, default="data/clutrr_v1_all_test_promptsfile.json",
    #     help="Path to the dataset of the input prompts"
    # )
    parser.add_argument(
        "--parse_cot",
        action="store_true",
        help="Whether to parse chain-of-thought in the predictions"
    )
    
    return parser

# main
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    symbolic_judge = load("AIML-TUDA/VerifiableRewardsForScalableLogicalReasoning")
    preds_path = args.preds_path
    parse_cot = args.parse_cot
    model_preds = read_jsonl(preds_path)
    # print(preds[0]['input'])
    # print(model_preds[12]['input'])
    predictions = [extract_prolog_program(p['outputs'][0], parse_cot=parse_cot) for p in tqdm(model_preds, desc='extracting prolog code')]
    gold_preds = [rec['input']['ground-truth rule'] for rec in model_preds]
    # print(predictions)

    references = [
        {
            "validation_program": (p['input']['validation program']),
            "evaluation_config": {
                "positive_predicate": "eastbound",
                "negative_predicate": "westbound"
            }
        } for p in model_preds
    ]
    
    results = symbolic_judge.compute(predictions=predictions, references=references)
    print("accuracy:", fm(results['accuracy']), "partial score:", fm(results['partial_score']), "syntax score:", fm(results['syntax_score']), "nulls:", sum([p == "NULL" for p in predictions]))
    scores_by_curriculum_tier = defaultdict(lambda: {"accuracy": [], "partial_score": [], "syntax_score": [], "nulls": []})
    for rec, p, rec_result in zip(model_preds, predictions, results['detailed_results']):
        curriculum_tier = rec['input']['curriculum tier']
        scores_by_curriculum_tier[curriculum_tier]["accuracy"].append(int(rec_result['is_correct']))
        scores_by_curriculum_tier[curriculum_tier]["partial_score"].append(int(rec_result['partial_score']))
        scores_by_curriculum_tier[curriculum_tier]["syntax_score"].append(int(rec_result['syntax_valid']))
        scores_by_curriculum_tier[curriculum_tier]["nulls"].append(int(p == "NULL"))
    # scores aggregated by curriculum tiers:
    for curriculum_tier in scores_by_curriculum_tier:
        results = scores_by_curriculum_tier[curriculum_tier]
        for metric in results:
            if metric == 'nulls': results[metric] = sum(results[metric])
            else: results[metric] = np.mean(results[metric])
        print(f"\x1b[34;1m{curriculum_tier}\x1b[0m")
        print("accuracy:", fm(results['accuracy']), "partial score:", fm(results['partial_score']), "syntax score:", fm(results['syntax_score']), "nulls:", results['nulls'])
    # # sanity testing: (gives 1).
    # results = symbolic_judge.compute(predictions=gold_preds, references=references)
    # print("accuracy:", results['accuracy'], "partial score:", results['partial_score'], "syntax score:", results['syntax_score'], "nulls:", sum([p == "NULL" for p in predictions]))