import os
import sys
import json
import random
import pathlib
import numpy as np
import pandas as od
from tqdm import tqdm
from collections import defaultdict

module_path = str(pathlib.Path(os.path.realpath(__file__)).parent.parent)
sys.path.append(module_path)

from src.data_generation.utils import read_jsonl
from src.data_generation.generate import write_jsonl

from src.data_generation.primitives import ProgramBFCCNode, ProgramVocabulary
from src.eval.prompts import unified_dsl_reasoning_prompt as UNI_DSLR_PROMPT, pbebench_task_description_prompt as TASK_DESC_PROMPT, pbebench_task_prompt as TASK_PROMPT

import random
import pandas as pd

# main
if __name__ == "__main__":
    input_path = sys.argv[1]
    EXPECTED_INPUTS = 5
    PROGRAM_NUM = 5
    vocab_chars = "abcdefghijkuvwxyz"
    PROGRAM_LENGTH = 3
    
    # load all the data
    data = json.load(open(input_path))
    for rec in tqdm(data):
        task_desc_prompt = TASK_DESC_PROMPT.format(program_num=PROGRAM_NUM, program_length=PROGRAM_LENGTH, alphabet=vocab_chars)
        task_prompt = TASK_PROMPT.format(inputs_list=rec["inputs"], outputs_list=rec["outputs"])
        # print(task_desc_prompt); exit()
        prompt = UNI_DSLR_PROMPT.format(task_description_prompt=task_desc_prompt, task_prompt=task_prompt)
        rec["prompt"] = prompt
        rec['response'] = f"""```python
{rec['programs']}
```"""
        # print(prompt)
        # exit()
    # output_path = "data/pbebench_training/pbebench_unified_dsl_reasoning_promptsfile.json"
    # print(output_path)
    # with open(output_path, "w") as f:
    #     json.dump(data, f, indent=4)

    # generate parquet files (for VeRL).
    stem, ext = os.path.splitext(input_path)
    output_path = stem + "_unified_dsl_format" + ext
    print(output_path)
    # exit()
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)