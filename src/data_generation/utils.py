import json
from tqdm import tqdm

def read_jsonl(path: str):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            data.append(json.loads(line))

    return data