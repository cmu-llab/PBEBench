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

def write_train_test_parquet(data, train_ratio=0.8, train_path="", test_path="", seed=42):
    assert 0.0 < train_ratio < 1.0

    random.seed(seed)
    data = list(data)
    random.shuffle(data)

    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    del train_df['bfcc_dag']
    del test_df['bfcc_dag']

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    train_json_path = train_path.removesuffix(".parquet")+".json"
    test_json_path = test_path.removesuffix(".parquet")+".json"
    # with open(train_json_path, "w") as f:
    #     json.dump(train_df.to_json(orient="records"), f, indent=4)
    # with open(test_json_path, "w") as f:
    #     json.dump(test_df.to_json(orient="records"), f, indent=4)
    with open(train_json_path, "w") as f:
        json.dump(
            train_df.to_dict(orient="records"),
            f,
            indent=4,
            ensure_ascii=False,
        )

    with open(test_json_path, "w") as f:
        json.dump(
            test_df.to_dict(orient="records"),
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Wrote {train_path}")
    print(f"Wrote {test_path}")

# main
if __name__ == "__main__":
    try: input_path = sys.argv[1]
    except IndexError: input_path = "data/pbebench_training/cascades_1_20_dig_vocab_dsz_100/shard_100_50_inputs_{}_cascade.jsonl"

    try: EXPECTED_INPUTS = int(sys.argv[2])
    except IndexError: EXPECTED_INPUTS = 50

    try: PROGRAM_NUM = int(sys.argv[3])
    except IndexError: 
        # PROGRAM_NUM = 30
        PROGRAM_NUM = 20

    try: vocab_chars = sys.argv[4]
    except IndexError: 
        # vocab_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZαβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
        # vocab_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        vocab_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

    try: PROGRAM_LENGTH = int(sys.argv[5])
    except IndexError: PROGRAM_LENGTH = 3
    
    # load all the data
    data = []
    for cascade_len in range(1, 20+1):
        data.extend(read_jsonl(input_path.format(cascade_len)))
    vocab = ProgramVocabulary(list(vocab_chars))

    changed_words_per_program = []
    changed_words_per_program_per_cascade = defaultdict(lambda: [])
    for rec in tqdm(data):
        assert len(rec["inputs"]) == EXPECTED_INPUTS == len(rec["outputs"])
        # if "original_programs" not in rec:
        #     rec["original_programs"] = rec["programs"]
        # assert len(rec["original_programs"]) == len(rec["programs"])
        assert len(rec["programs"]) <= PROGRAM_NUM
        for program in rec["programs"]:
            program_node = ProgramBFCCNode.from_string(program, vocabulary=vocab)
        
        prev_outputs = rec['inputs']
        assert rec['inputs'] != rec['outputs']

        # flag degenerate programs that don't alter any inputs.
        for program_ind,program in enumerate(rec['programs']):
            # program = program.replace("\\","")
            program_node = ProgramBFCCNode.from_string(program, vocabulary=ProgramVocabulary(vocab_chars))
            next_inputs = program_node(prev_outputs)
            changed_words = sum([int(nI != pO) for nI, pO in zip(next_inputs, prev_outputs)])
            changed_words_per_program.append(changed_words)
            changed_words_per_program_per_cascade[rec['cascade_length']].append(changed_words)
            # print(changed_words)
            # print(prev_outputs, program, next_inputs)
            assert next_inputs != prev_outputs, "Found degenerate program!"
            prev_outputs = next_inputs
        assert rec['outputs'] == next_inputs == prev_outputs # the program leads to the same output as the ground truth.

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
    changed_words_per_program_per_cascade = dict(changed_words_per_program_per_cascade)
     

    print(f"On average {np.mean(changed_words_per_program):.2f} inputs are changed by a program")
    for cascade_length, cwpp in changed_words_per_program_per_cascade.items():
        print(f"For cascade_length={cascade_length} {np.mean(cwpp):.2f} inputs are changed on avg by a program")
    
    # output_path = "data/pbebench_training/pbebench_unified_dsl_reasoning_promptsfile.json"
    # print(output_path)
    # with open(output_path, "w") as f:
    #     json.dump(data, f, indent=4)

    # generate parquet files (for VeRL).

    write_train_test_parquet(
        data, train_ratio=0.8, seed=42,
        train_path="data/pbebench_training/unified_dsl_reasoning_promptsfile_dig/train.parquet",
        test_path="data/pbebench_training/unified_dsl_reasoning_promptsfile_dig/test.parquet",
    )