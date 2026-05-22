import os
import sys
import json
import boto3
import pathlib
import argparse
import functools
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import ClientError

def write_jsonl(data, path: str):
    bytes_written = 0
    with open(path, "w") as f:
        for rec in data:
            bytes_written += f.write(json.dumps(rec)+"\n")

    return bytes_written

parser = argparse.ArgumentParser()
parser.add_argument('--start-index', type=int, default=0, help='Stratified sampling: specify start of inference calls')
parser.add_argument('--end-index', type=int, default=-1, help='Stratified sampling: specify end of inference calls')
parser.add_argument('--model', type=str, default='claude_3.7_sonnet', 
                    choices=['claude_3.5_sonnet', 'claude_3.7_sonnet', 'claude_4_sonnet'], 
                    help='claude model variant to be queried.')
parser.add_argument('--max_tokens', type=int, default=2048, help='AWS region for boto3 client')
parser.add_argument('--cache_id', type=str, default="", help='unique identifier to separate cache dirs for different runs.')
parser.add_argument('--think_tokens', type=int, default=0, help='AWS region for boto3 client') # use 2048
parser.add_argument('--region', type=str, default='us-east-1', help='AWS region for boto3 client')
parser_args = parser.parse_args()

assert (parser_args.start_index < parser_args.end_index) or (parser_args.end_index == -1)

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
print("module_path:", module_path)
sys.path.append(module_path)

def read_jsonl(path: str, disable: bool=False):
    data = []
    with open(path, "r") as f:
        for line in tqdm(f, disable=disable):
            data.append(json.loads(line.strip()))

    return data

@functools.lru_cache(1)
def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=parser_args.region)

if parser_args.model == "claude_3.7_sonnet" and parser_args.think_tokens != 0:
    assert parser_args.max_tokens > parser_args.think_tokens 

if parser_args.model == "claude_3.5_sonnet":
    DEFAULT_INF_CONFIG = {
        "maxTokens": int(parser_args.max_tokens),  
        "temperature": 0.5, 
        "topP": 0.9,
    }
elif parser_args.model == "claude_3.7_sonnet":
    DEFAULT_INF_CONFIG = {
        "maxTokens": int(parser_args.max_tokens),  
        "temperature": 1, # need temperature of 1 for using thinking mode. 
    }
elif parser_args.model == "claude_4_sonnet":
    DEFAULT_INF_CONFIG = {
        "maxTokens": int(parser_args.max_tokens),  
        "temperature": 1, # need temperature of 1 for using thinking mode. 
    }

region2model_id = {
    "claude_3.5_sonnet": {
        'us-east-1': "anthropic.claude-3-5-sonnet-20240620-v1:0",
        'us-east-2': "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        'ap-south-1': "apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
        'ap-northeast-1': "apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
        'us-west-2': "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    },
    "claude_3.7_sonnet": {
        'us-east-1': "anthropic.claude-3-7-sonnet-20250219-v1:0",
        'us-east-2': "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        'ap-south-1': "apac.anthropic.claude-3-7-sonnet-20250219-v1:0",
        'ap-northeast-1': "apac.anthropic.claude-3-7-sonnet-20250219-v1:0",
        'us-west-2': "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    },
    "claude_4_sonnet": {
        'us-east-1': "anthropic.claude-sonnet-4-20250514-v1:0",
        'us-east-2': "us.anthropic.claude-sonnet-4-20250514-v1:0",
        'ap-south-1': "apac.anthropic.claude-sonnet-4-20250514-v1:0",
        'ap-northeast-1': "apac.anthropic.claude-sonnet-4-20250514-v1:0",
        'us-west-2': "us.anthropic.claude-sonnet-4-20250514-v1:0",   
    }
}

def get_text(resp):
    return resp['output']['message']['content'][0]['text']

def query_llm(
    message,
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    **inference_config
):
    inf_cfg = {**DEFAULT_INF_CONFIG, **inference_config}
    conversation = [
        {
            "role": "user",
            "content": [{"text": message}],
        }
    ]
    client = get_bedrock_client()
    additionalModelRequestFields={
        "thinking": {
            "type": "enabled",
            "budget_tokens": parser_args.think_tokens # Allocate 4096 tokens for Claude's thinking
        }
    }
    if parser_args.model == "claude_3.5_sonnet":
        response = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig=inf_cfg,
        )
    elif parser_args.model in ["claude_3.7_sonnet", "claude_4_sonnet"]:
        if parser_args.think_tokens == 0:        
            response = client.converse(
                modelId=model_id,
                messages=conversation,
                inferenceConfig=inf_cfg,
            )
        else:
            response = client.converse(
                modelId=model_id,
                messages=conversation,
                inferenceConfig=inf_cfg,
                additionalModelRequestFields=additionalModelRequestFields,
            )
    return response

if parser_args.model == "claude_3.5_sonnet" or parser_args.think_tokens == 0:
    cache_dir = f"./outputs/{parser_args.cache_id}{parser_args.model}_max_{parser_args.max_tokens}"
else: # it is claude_3.7_sonnet with multiple think tokens.
# elif parser_args.model == "claude_3.7_sonnet":
    cache_dir = f"./outputs/{parser_args.cache_id}{parser_args.model}_max_{parser_args.max_tokens}_think_{parser_args.think_tokens}"

os.makedirs(cache_dir, exist_ok=True)
def _query_llm(query_args):
    global parser_args
    i, prompt = query_args
    cache_file = os.path.join(cache_dir, f"{i}.json")
    if os.path.isfile(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    result = query_llm(prompt, model_id=region2model_id[parser_args.model][parser_args.region])
    with open(cache_file, "w+") as f:
        json.dump(result, f)
    return result

def run_inference(prompts, start_index=0, n_workers=8, secure_coding: bool=False):
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        args = [
            (i,rec['prompt']) for i,rec in enumerate(prompts, start_index)
        ]
        assert all("**Recommendation**:" not in arg for arg in args)
        responses = list(
            tqdm(executor.map(_query_llm, args), total=len(prompts)+start_index, initial=start_index)
        )

    return responses

# main
if __name__ == "__main__":
    # prompts.
    data = json.load(open("./data/cascaded_checked_balanced_program_transformations_dataset_1008_promptsfile.json"))
    if parser_args.end_index == -1:
        parser_args.end_index = len(data)
    
    data = data[parser_args.start_index:parser_args.end_index]
    print(len(data))
    responses = run_inference(data, start_index=parser_args.start_index, n_workers=10)
    write_data = []
    for rec, resp in zip(data, responses):
        del rec['prompt']
        # print(resp['output']['message']['content'])
        # print(resp['output']['message']['content'])
        content = resp['output']['message']['content']
        write_data.append({
            "input": rec,
            "outputs": [content[-1]['text']],
            "bedrock_response": resp['output'],
        })
        
    if parser_args.model != "claude_3.5_sonnet":
        print(write_jsonl(write_data, f"./outputs/{parser_args.cache_id}{parser_args.model}_think_{parser_args.think_tokens}_{parser_args.max_tokens}_{parser_args.start_index}_{parser_args.end_index}.jsonl"))
    else:
        print(write_jsonl(write_data, f"./outputs/{parser_args.cache_id}{parser_args.model}_{parser_args.max_tokens}_{parser_args.start_index}_{parser_args.end_index}.jsonl"))

# python src/eval/models/query_claude.py --region us-east-1 --model 'claude_3.5_sonnet' --cache_id "ada_bal_1008_" --max_tokens 2048
# python src/eval/models/query_claude.py --region us-east-2 --model 'claude_3.7_sonnet' --cache_id "ada_bal_1008_" --max_tokens 2048 --think_tokens 0