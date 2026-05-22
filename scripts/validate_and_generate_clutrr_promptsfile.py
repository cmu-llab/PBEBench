import os
import ast
import json
import datasets
from tqdm import tqdm
from datasets import load_dataset

CLUTRR_QA_PROMPT_TEMPLATE = """You will be given a story containing characters whose names appear in square brackets, such as "[James]". After reading the story, you must identify the kinship relation between two characters.

When answering:

- Respond with only the kinship term, with no explanation and no extra words.
- Your answer must be one of the following options:
aunt, brother, daughter, daughter-in-law, father, father-in-law, granddaughter, grandfather, grandmother, grandson, mother, mother-in-law, nephew, niece, sister, son, son-in-law, uncle
- Do not use any text outside these options.
- Give just the final relation.

Example Story: [Kristin] and her son [Justin] went to visit her mother [Carol] on a nice Sunday afternoon. They went out for a movie together and had a good time.

Example Question: How is Carol related to Justin?

Example Answer: grandmother

Now do the same for the story and question below:

Story: {story}

Question: {question}

Answer:
"""

# main
if __name__ == "__main__":
    splits = ["gen_train234_test2to10", "gen_train23_test2to10", "rob_train_clean_23_test_all_23", "rob_train_disc_23_test_all_23", "rob_train_irr_23_test_all_23", "rob_train_sup_23_test_all_23"]
    
    raw_data = []
    for split in splits:
        raw_split_data = load_dataset("CLUTRR/v1", split)['test']
        split_data = []
        for i in range(len(raw_split_data)):
            rec = raw_split_data[i]
            rec["split_name"] = split
            split_data.append(rec)
        # split_data = [split_data[i] for i in range(len(split_data))]
        raw_data.extend(split_data)
    possible_relations = set()
    for rec in tqdm(raw_data):
        possible_relations.add(rec['target_text'])
    possible_relations = sorted(list(possible_relations))
    print(possible_relations)

    prompts = []
    for rec in tqdm(raw_data): # we should test on the actual story and not the clean story (regular story has distractor facts and stuff)
        story = rec['story']
        query = ast.literal_eval(rec['query'])
        person1 = query[0]
        person2 = query[1]
        # options = ", ".join(possible_relations)
        question = f"How is {person2} related to {person1}?"
        rec['prompt'] = CLUTRR_QA_PROMPT_TEMPLATE.format(story=story, question=question)
        prompts.append(rec)
        # print(rec['prompt'])
        # exit()
    print(len(prompts))
    with open("data/clutrr_v1_all_test_promptsfile.json", "w") as f:
        json.dump(prompts, f, indent=4)