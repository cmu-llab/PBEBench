import os
import sys
import pathlib
from models.sglang import load_sglang_model, generate_with_sglang_engine
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from openai import OpenAI
from budget_tracker import log_and_track

import json
from concurrent.futures import ThreadPoolExecutor, as_completed


module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
sys.path.append(module_path)

from src.eval.models.vllm_engine import vLLMEngine


def _call_model(prompt, sample_parameters, engine):
    # Keep same call you used
    response = engine.with_options(timeout=1800).chat.completions.create(
        model=sample_parameters["model_ckpt"],
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=sample_parameters["max_new_tokens"],
        temperature=1,
        reasoning_effort="medium"
    )
    print(prompt)
    print(json.dumps(response.to_dict(), indent=2))
    print("\n\n")
    return response.choices[0].message.content


def load_model(model_ckpt, engine_type, tp: int=1, port: int=8001):
    if engine_type == 'sglang':
        return load_sglang_model(model_ckpt, tp)
    elif engine_type == "vllm":
        return vLLMEngine(model_ckpt=model_ckpt)
    elif engine_type == 'huggingface':
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model = AutoModelForCausalLM.from_pretrained(model_ckpt, device_map="auto")
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    elif engine_type == 'openai':
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return client

    elif engine_type == 'claude':
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")) # or just directly replace with key
        return client

    elif engine_type == 'gemini':
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY")) # or just directly replace with key
        return genai.GenerativeModel(model_ckpt)

    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")

def generate(engine, inputs, engine_type, sample_parameters):
    if engine_type == 'sglang':
        outputs = generate_with_sglang_engine(engine, inputs, sample_parameters)
        for prompt, output_text in zip(inputs, outputs):
            log_and_track(sample_parameters["model_ckpt"], prompt, output_text, log=True)
        return outputs

    elif engine_type == 'huggingface':
        outputs = []
        for input in inputs:
            response = engine(
                input, max_new_tokens=sample_parameters["max_new_tokens"],
                temperature=sample_parameters["temperature"],
                top_k=sample_parameters["top_k"],
                top_p=sample_parameters["top_p"]
            ) 
            output_text = response[0]['generated_text'][len(input):]
            print(f"RESPONSE: {output_text}")
            outputs.append(output_text)
            log_and_track(sample_parameters["model_ckpt"], input, output_text, log=True)

        return outputs

    elif engine_type == "vllm":
        outputs = []
        # for prompt in inputs:
            # response = await engine.generate_response(
            #     user_prompt=prompt,
            #     max_new_tokens=sample_parameters["max_new_tokens"],
            #     temperature=sample_parameters["temperature"],
            #     top_p=sample_parameters["top_p"]
            # )
            # output_text = response["model_response"]
            # outputs.append(output_text)
            # log_and_track(sample_parameters["model_ckpt"], prompt, output_text, log=True)
        return outputs

    elif engine_type == 'openai':
        outputs = []
        n = len(inputs)
        outputs = [None] * n
        with ThreadPoolExecutor(max_workers=n) as ex:
            future_to_idx = {
                ex.submit(_call_model, inputs[i], sample_parameters, engine): i
                for i in range(n)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    outputs[idx] = future.result()
                except Exception as e:
                    # choose how to handle errors: set None, store exception string, or re-raise
                    outputs[idx] = f"ERROR: {e}"

        # for prompt in inputs:
        #     response = engine.with_options(timeout=1800).chat.completions.create(
        #         model=sample_parameters["model_ckpt"],
        #         messages=[{"role": "user", "content": prompt}],
        #         max_completion_tokens=sample_parameters["max_new_tokens"],
        #         temperature=1,
        #         reasoning_effort="medium"
        #     )
        #     output_text = response.choices[0].message.content
        #     outputs.append(output_text)
        #     # log_and_track(sample_parameters["model_ckpt"], prompt, output_text, log=True)
        return outputs

    elif engine_type == 'claude':
        from anthropic.types import TextBlock, ThinkingBlock
        
        outputs = []
        for prompt in inputs:
            response_obj = engine.messages.create(
                model=sample_parameters["model_ckpt"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=sample_parameters["max_new_tokens"],
                temperature=sample_parameters["temperature"],
                top_p = sample_parameters["top_p"]
                # Add this block to enable reasoning
                # thinking={
                #     "type": "enabled",
                #     "budget_tokens": 2048  # must be ≥ 1024, increase for more detailed reasoning
                # }
            )
            # use this for non-reasoning
            output_text = response_obj.content[0].text
            # use this for reasoning
            # output_text = ""
            # for block in response_obj.content:
            #     if isinstance(block, ThinkingBlock):
            #         output_text += block.thinking
            #     elif isinstance(block, TextBlock):
            #         output_text += block.text
            outputs.append(output_text)
            log_and_track(sample_parameters["model_ckpt"], prompt, output_text)
        return outputs

    elif engine_type == 'gemini':
        outputs = []
        for prompt in inputs:
            response_obj = engine.generate_content(prompt)
            output_text = response_obj.text
            outputs.append(output_text)
            log_and_track(sample_parameters["model_ckpt"], prompt, output_text, log=True)
        return outputs

    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")
