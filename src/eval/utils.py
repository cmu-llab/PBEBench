import json
from tqdm import tqdm

def prompt_construct(prompt, max_rules_num, max_rule_length, characters, inputs_list, outputs_list, program_num: int, program_length: int):
    return prompt.format(rules_length=max_rule_length, characters=characters, rules_num=max_rules_num, inputs_list=inputs_list, outputs_list=outputs_list, program_length=program_length, program_num=program_num)

def read_jsonl(path: str, disable: bool=False) -> list[dict]:
    data = []
    with open(path, "r", encoding='utf-8') as f:
        for line in tqdm(f, disable=disable):
            data.append(json.loads(line.strip()))

    return data

def write_jsonl(data, path: str, disable: bool=False) -> list[dict]:
    bytes_written = 0
    with open(path, "w", encoding='utf-8') as f:
        for rec in tqdm(data, disable=disable):
            bytes_written += f.write(json.dumps(rec)+"\n")

    return bytes_written
