import sys
import json
from tqdm import tqdm
from pathlib import Path
import pyarrow.json as pj
import pyarrow.parquet as pq

def read_jsonl(path: str):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            data.append(json.loads(line))

    return data

# main
if __name__ == "__main__":
    path = sys.argv[1] # e.g. "data/pbebench_training/unified_dsl_reasoning_cot_sdft_offline/qwen3_4b_instruct/train_cot_responses.jsonl"
    jsonl_path = Path(path)

    parquet_path = jsonl_path.with_suffix(".parquet")
    # print(parquet_path)

    # Read JSONL
    table = pj.read_json(jsonl_path)

    # Write Parquet
    pq.write_table(table, parquet_path)

    print(f"Wrote parquet file to: {parquet_path}")