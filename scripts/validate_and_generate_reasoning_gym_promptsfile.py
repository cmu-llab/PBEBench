import os
import sys
import copy
import json
import random
import pathlib
import numpy as np
import pandas as od
import reasoning_gym
from tqdm import tqdm
from collections import defaultdict

module_path = str(pathlib.Path(os.path.realpath(__file__)).parent.parent)
sys.path.append(module_path)

from src.data_generation.utils import read_jsonl
from src.data_generation.generate import write_jsonl

from src.eval.prompts import unified_dsl_reasoning_prompt as UNI_DSLR_PROMPT, pbebench_task_description_prompt as TASK_DESC_PROMPT, pbebench_task_prompt as TASK_PROMPT, REASONING_GYM_PROMPT_SPLITTERS

import random
import pandas as pd

def write_train_test_jsons(data, train_ratio=0.8, train_path="", test_path="", seed=42):
    assert 0.0 < train_ratio < 1.0
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    random.seed(seed)
    data = list(data)
    random.shuffle(data)

    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # del train_df['metadata']
    # del test_df['metadata']

    # train_df.to_parquet(train_path, index=False)
    # test_df.to_parquet(test_path, index=False)

    with open(train_path, "w") as f:
        json.dump(
            train_df.to_dict(orient="records"),
            f,
            indent=4,
            ensure_ascii=False,
        )

    with open(test_path, "w") as f:
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
    instances_per_env = 10
    all_data = []
    for env, splitter in REASONING_GYM_PROMPT_SPLITTERS.items():
        print(f"Processing environment: {env}")
        env_chunk = list(reasoning_gym.create_dataset(env, size=10, seed=42))
        task_prompt = splitter(env_chunk[0]['question'])
        task_desc_prompt = copy.deepcopy(splitter.task_description_prompt)
        for instance in env_chunk:
            task_prompt = splitter(instance['question'])
            prompt = UNI_DSLR_PROMPT.format(task_description_prompt=task_desc_prompt, task_prompt=task_prompt)
            all_data.append({
                "prompt": prompt,
                "question": instance['question'],
                "response": instance['answer'],
                "metadata": instance['metadata'], 
            })

    # generate parquet files (for VeRL).

    write_train_test_jsons(
        all_data, train_ratio=0.8, seed=42,
        train_path="data/pbebench_training/unified_dsl_reasoning_gym_promptsfile/train.json",
        test_path="data/pbebench_training/unified_dsl_reasoning_gym_promptsfile/test.json",
    )