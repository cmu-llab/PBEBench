import json
from itertools import permutations, islice

def remove_matching_chars_at_same_position(input_dict):
    modified_dict = {}

    for key, value in input_dict.items():
        min_length = min(len(key), len(value))
        new_key = []
        new_value = []

        for i in range(min_length):
            if key[i] != value[i]:
                new_key.append(key[i])
                new_value.append(value[i])

        if len(key) > min_length:
            new_key.extend(key[min_length:])
        if len(value) > min_length:
            new_value.extend(value[min_length:])

        modified_dict[''.join(new_key)] = ''.join(new_value)

    return modified_dict

def encompasses(s1: str, s2: str) -> bool:
    return s2 in s1 or s1 in s2

def can_merge(s1: str, s2: str) -> bool:
    for i in range(1, len(s1) + 1):
        if s2.startswith(s1[-i:]):
            return True

    for i in range(1, len(s2) + 1):
        if s1.startswith(s2[-i:]):
            return True

    return False

def process_programs(programs):
    relations = []

    for (i, prog1), (j, prog2) in permutations(enumerate(programs), 2):
        original_dict = {
            prog1.split("'")[1]: prog1.split("'")[3],
            prog2.split("'")[1]: prog2.split("'")[3]
        }
        reduced_dict = remove_matching_chars_at_same_position(original_dict)

        iterator_orig = iter(original_dict.items())
        iterator_red = iter(reduced_dict.items())
        try:
            (orig_input1, orig_output1), (orig_input2, orig_output2) = islice(iterator_orig, 2)
            (red_input1, red_output1), (red_input2, red_output2) = islice(iterator_red, 2)
        except ValueError:
            print("Darsh: Map did not contain two key-value pairs")
            continue

        result = []
        if set(red_input1) & set(orig_input2):
            result.append('B')
        if set(red_output1) & set(orig_input2):
            if encompasses(orig_output1, orig_input2) or can_merge(orig_output1, orig_input2):
                result.append('F')
        if not result:
            result.append('N')

        relations.append([i, result, j])

    return relations

def process_file_and_save(file_path, output_path):
    with open(file_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            entry = json.loads(line)
            programs = entry.get("programs", [])
            entry["bfcc_dag"] = process_programs(programs)
            outfile.write(json.dumps(entry) + '\n')

file_path = "../../data/relations-test/test_input_data.jsonl"
output_path = "../../data/relations-test/test_output_data.jsonl"
process_file_and_save(file_path, output_path)