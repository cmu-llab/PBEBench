import os
import sys
import json
import pathlib
import datasets
from tqdm import tqdm

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
print(module_path)
sys.path.append(module_path)

from src.eval.prompts import prompt_v1

def chunk_corresponding(A, B, chunk_size):
    if len(A) != len(B):
        raise ValueError("A and B must have the same length")

    A_chunks = []
    B_chunks = []

    for i in range(0, len(A), chunk_size):
        A_chunks.append(A[i:i + chunk_size])
        B_chunks.append(B[i:i + chunk_size])

    return A_chunks, B_chunks

# main
if __name__ == "__main__":
    vocab_needed = set()
    proto_attested_pairs = [
        "atr_hawaiian",
        "atr_niue",
        "atr_rarotongan",
        "atr_samoan",
        "atr_tongan",
        "ptk_huishu",
    ]
    
    prompts = []
    chunk_size = 50
    for proto_attested_pair in proto_attested_pairs:
        inputs_list = [w.strip() for w in open(f"data/real_sli/{proto_attested_pair}/words.in").read().split("\n") if w.strip() != ""]
        outputs_list = [w.strip() for w in open(f"data/real_sli/{proto_attested_pair}/words.out").read().split("\n") if w.strip() != ""]
        assert len(inputs_list) == len(outputs_list), f"{proto_attested_pair}: {len(inputs_list)} {len(outputs_list)}"
        for w in inputs_list+outputs_list:
            for char in w: vocab_needed.add(char)
        
        inputs_lol, outputs_lol = chunk_corresponding(inputs_list, outputs_list, chunk_size=50) # lol: list of lists:
        for inputs_list, outputs_list in zip(inputs_lol, outputs_lol):
            prompt = prompt_v1.format(program_length=5, program_num=50, inputs_list=inputs_list, outputs_list=outputs_list)
            prompts.append({
                "inputs": inputs_list,
                "outputs": outputs_list,
                "prompt": prompt,
                "lang_pair": proto_attested_pair,
            })

    vocab_needed = sorted(list(vocab_needed))
    print("size of vocab:", len(vocab_needed))
    print("number of instances in dataset:", len(prompts))
    with open("data/real_sli/vocab.json", "w") as f:
        json.dump(vocab_needed, f, indent=4)
    with open("data/real_sli/real_sli_promptsfile.json", "w") as f:
        json.dump(prompts, f, indent=4)
