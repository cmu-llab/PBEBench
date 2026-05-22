import json

input_file = "./adaptive_balanced_1008_complete_promptsfile.json"
output_file = "./adaptive_balanced_1008_complete_promptsfile.jsonl"

with open(input_file, "r") as f:
    data = json.load(f)

with open(output_file, "w") as f:
    for obj in data:
        f.write(json.dumps(obj) + "\n")
