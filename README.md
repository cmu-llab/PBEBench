# pbe-reasoning
A benchmark to evaluate the reasoning capabilities of LLMs using linguistics inspired BFCC sequential string manipulation programs in a programming by example/sound law induction setting.


## Running Instructions.

### Data Generation
To automatically generate samples run:
```bash
python src/data_generation/generate.py
```

### Data Validation
To valdiate automatically generated or human written samples run:
```bash
python src/data_generation/validate.py "/path/to/samples.json"
```

### Program Permutation/Reordering Task
Dataset creation command:
```bash
python src/permutation_eval/dataset.py --input "data/adaptive_balanced_1008_complete_promptsfile.jsonl" --output "data/adaptive_balanced_1008_permutation_promptsfile.jsonl" --max-attempts 10000 --seed 42 --strategy "fb_swap"
```