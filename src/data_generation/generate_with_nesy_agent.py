import os
import sys
import json
import pathlib
from collections import defaultdict

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent) # print(module_path)
sys.path.append(module_path)

from src.tools.coverage import CoverageChecker
from src.tools.execution import ExecutionTool
from src.tools.relation_classification import

def generate_response(rec, model_ckpt, sample_parameters):
    user_prompt = rec['prompt'] # directly use the pre-generated prompts in promptsfile.
    messages = [{"role": "user", "content": user_prompt}]

    payload = {
        "model": model_ckpt,
        "messages": messages,
        "max_tokens": sample_parameters['max_new_tokens'],
        "temperature": sample_parameters['temperature'],
        "top_p": sample_parameters['top_p'],
        "top_k": sample_parameters['top_k'],
        "seed": 42
    }

    if sample_parameters['no_think']:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    response = requests.post(VLLM_SERVER_URL, json=payload)
    response.raise_for_status()
    response_json = response.json()
    model_response = response_json["choices"][0]["message"]
    rec["model_response"] = model_response['content']
    rec["model_cot"] = model_response.get('reasoning_content')

    return rec

# main
if __name__ == "__main__":
