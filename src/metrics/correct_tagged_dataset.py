import json
import re
import ast
from tqdm import tqdm
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
from src.data_generation.utils import read_jsonl, write_jsonl
from src.data_generation.primitives import ProgramBFCCNode, ProgramVocabulary

def extract_first_python_block(markdown_text):
    code_block_pattern = r"```python\s(.*?)```"
    matches = re.findall(code_block_pattern, markdown_text, re.DOTALL)
    try:
        return [block.strip() for block in matches][0]
    except IndexError:
        return "[]"

def try_ast_literal_eval(string):
    try:
        program_seq = ast.literal_eval(string)
        if isinstance(program_seq, str):
            program_seq = [program_seq]
        assert isinstance(program_seq, list)
        return program_seq
    except:
        return []

def identity(x): return x

def eval_passing_test_cases(inputs, outputs, pred_cascade, vocab_chars="abcdef"):
    pred_outputs = inputs.copy()
    for program in pred_cascade:
        if isinstance(program, str):
            try:
                program = ProgramBFCCNode.from_string(program, vocabulary=ProgramVocabulary(vocab_chars))
            except Exception as e:
                print(f"Invalid program: {e}")
                program = identity
        else:
            print(f"Non-str program: {type(program)}. Using identity.")
            program = identity
        pred_outputs = program(pred_outputs)
    return int(all(o == po for o, po in zip(outputs, pred_outputs)))

def main(input_path: str, output_path: str, vocab_chars: str = "abcdefghijkuvwxyz"):
    raw_data = read_jsonl(input_path)
    new_data = []

    for rec in tqdm(raw_data):
        item = rec["input"]
        # if not isinstance(rec["outputs"][0], str):
        if not isinstance(rec["output"], str):
            new_data.append({
                "input": item,
                "outputs": rec["outputs"],
                "correct": 0
            })
            continue

        # prediction_code = extract_first_python_block(rec["outputs"][0])
        prediction_code = extract_first_python_block(rec["output"])
        prediction_code = re.sub(r"(?<!['\"])(replace\([^)]*\))(?!['\"])", r'"\1"', prediction_code)
        pred_cascade = try_ast_literal_eval(prediction_code)

        correct = eval_passing_test_cases(
            inputs=item["inputs"],
            outputs=item["outputs"],
            pred_cascade=pred_cascade,
            vocab_chars=vocab_chars
        )

        new_data.append({
            "input": item,
            # "outputs": rec["outputs"],
            "outputs": rec["output"],
            "correct": correct
        })

    write_jsonl(new_data, output_path)
    print(f"Done. Output written to {output_path}")

if __name__ == "__main__":
    input_file = "/home/darsha/Testing/pbe-reasoning/outputs/qwen3_32b_preds_cascaded_checked_balanced_program_transformations_dataset_2048_max_token_length.jsonl"
    output_file = "/home/darsha/Testing/pbe-reasoning/outputs/tagged_qwen3_32b_preds_cascaded_checked_balanced_program_transformations_dataset_2048_max_token_length.jsonl"
    main(input_file, output_file)
