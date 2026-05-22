import os
import sys
import json
import pathlib
import numpy as np
from tqdm import tqdm
from collections import defaultdict

module_path = str(pathlib.Path(os.path.realpath(__file__)).parent.parent)
sys.path.append(module_path)

from src.data_generation.utils import read_jsonl
from src.data_generation.generate import write_jsonl
from src.eval.prompts import prompt_v1 as PROMPT_TEMPLATE
from src.data_generation.primitives import ProgramBFCCNode, ProgramVocabulary

PROGRAM_LENGTH = 3

# main
if __name__ == "__main__":
    input_path = sys.argv[1]

    try: EXPECTED_INPUTS = int(sys.argv[2])
    except IndexError: EXPECTED_INPUTS = 5

    try: PROGRAM_NUM = int(sys.argv[3])
    except IndexError: PROGRAM_NUM = 5 

    try: vocab_chars = sys.argv[4]
    except IndexError: vocab_chars = "abcdefghijkuvwxyz"
    
    data = read_jsonl(input_path)
    vocab = ProgramVocabulary(list(vocab_chars))

    changed_words_per_program = []
    changed_words_per_program_per_cascade = defaultdict(lambda: [])
    for rec in tqdm(data):
        assert len(rec["inputs"]) == EXPECTED_INPUTS == len(rec["outputs"])
        if "original_programs" not in rec:
            rec["original_programs"] = rec["programs"]
        assert len(rec["original_programs"]) == len(rec["programs"])
        assert len(rec["programs"]) <= PROGRAM_NUM
        for program in rec["original_programs"]:
            program_node = ProgramBFCCNode.from_string(program, vocabulary=vocab)
        
        prev_outputs = rec['inputs']
        assert rec['inputs'] != rec['outputs']
        # flag degenerate programs that don't alter any inputs.
        for program_ind,program in enumerate(rec['original_programs']):
            program = program.replace("\\","")
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

        prompt = PROMPT_TEMPLATE.format(inputs_list=rec["inputs"], outputs_list=rec["outputs"], program_num=PROGRAM_NUM, program_length=PROGRAM_LENGTH)
        rec["prompt"] = prompt
        # print(prompt)
        # exit()
    changed_words_per_program_per_cascade = dict(changed_words_per_program_per_cascade)
     
    stem, ext = os.path.splitext(input_path)
    print(f"On average {np.mean(changed_words_per_program):.2f} inputs are changed by a program")
    for cascade_length, cwpp in changed_words_per_program_per_cascade.items():
        print(f"For cascade_length={cascade_length} {np.mean(cwpp):.2f} inputs are changed on avg by a program")
    output_path = stem+"_promptsfile.json"
    print(output_path)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)