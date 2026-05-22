# generate initial data automatically to be reviewed by a human.

import os
import sys
import json
import random
import pathlib
from tqdm import tqdm

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(module_path)

from src.data_generation.detector import process_programs
from src.data_generation.primitives import ProgramVocabulary, ProgramBFCCNode

def write_jsonl(data, path: str):
    with open(path, "w") as f:
        for rec in tqdm(data):
            f.write(json.dumps(rec)+"\n")

def sample_inputs(vocab_chars: str, num_inputs: int=5, input_len_range: tuple[int, int]=(2,6)):
    program_vocab = ProgramVocabulary(list(vocab_chars))
    inputs = []
    input_lens = list(range(input_len_range[0], input_len_range[1]+1))
    for i in range(num_inputs):
        word_len = random.choice(input_lens)
        word = "".join(random.choices(vocab_chars, k=word_len))
        program_vocab.has_word(word)
        inputs.append(word)

    return inputs

def sample_program(vocab_chars: str, env_len_range: tuple[int, int]=(1,3)):
    program_vocab = ProgramVocabulary(list(vocab_chars))
    env_lens = list(range(env_len_range[0], env_len_range[1]+1))
    input_env_len = random.choice([1,2,3])
    input_chars = "".join(random.choices(vocab_chars, k=input_env_len))
    output_env_len = random.choice([1,2,3])
    output_chars = "".join(random.choices(vocab_chars, k=output_env_len))
    string_repr = f'replace("{input_chars}", "{output_chars}")'
    code = ProgramBFCCNode.from_string(string_repr=string_repr, vocabulary=program_vocab)
    
    return code, string_repr

def sample_instance(vocab_chars, num_inputs: int=5, input_len_range: tuple[int, int]=(2,6), env_len_range: tuple[int, int]=(1,3), seq_len: int=3):
    inputs = None
    outputs = None
    while inputs == outputs:
        inputs = sample_inputs(vocab_chars=vocab_chars, num_inputs=num_inputs, input_len_range=input_len_range)
        outputs = inputs
        programs = []
        for _ in range(seq_len):
            code, string_repr = sample_program(vocab_chars=vocab_chars, env_len_range=env_len_range)
            outputs = code(outputs)
            programs.append(string_repr)
        print('trying rule')
        if inputs != outputs: print("found rule")

    bfn_dag = process_programs(programs=programs)
        
    return {"inputs": inputs, "outputs": outputs, "programs": programs, "bfn_dag": bfn_dag}

# main
if __name__ == "__main__":
    data = []
    for seq_len in [1,2,3,4,5]:
        for _ in range(5):
            instance = sample_instance(vocab_chars="abcdefghijkxyz", num_inputs=5, input_len_range=(2,6), env_len_range=(1,3), seq_len=seq_len)
            data.append(instance)

    # write_jsonl(data, "./data/example_auto_gen_data_atharva.jsonl")
    write_jsonl(data, "/Users/manavnitinkapadnis/Documents/GitHub/pbe-reasoning/data/example_auto_gen_data_manav.jsonl")