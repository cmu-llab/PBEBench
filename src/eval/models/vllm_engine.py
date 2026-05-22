import os
import sys
import json
import pathlib
import requests 

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
sys.path.append(module_path)

VLLM_SERVER_URL = "http://0.0.0.0:8001/v1/chat/completions"
MAX_RETRIES = 5
MAX_SEQ_LEN = 32768
MAX_NEW_TOKENS = 2048

class vLLMEngine:
    def __init__(self, model_ckpt: str, port: int=8001, seed: int=42):
        self.port = port
        self.model_ckpt = model_ckpt
        self.seed = seed

    async def generate_response(self, user_prompt: str, temperature: float=0.7, top_p: float=0.95, max_new_tokens: int=MAX_NEW_TOKENS):
        """ Sends a request to vLLM for generating a response """
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        payload = {
            "model": self.model_ckpt,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature, 
            # greedy decoding for reproducibility
            "top_p": top_p,
            "seed": self.seed,  
            # seed for reproducibility
        }

        for _ in range(MAX_RETRIES):
            try:
                response = requests.post(VLLM_SERVER_URL, json=payload).json()
                model_response = response["choices"][0]["message"]["content"]
                return {
                    "model_response": model_response,
                    "error": False,
                }
            except Exception as e:
                print(e)
                # print(response)
                # print(e)
        return {
            "model_response": f"Got Error",
            "error": True,
        }

    # async def generate_response(self, user_prompt: str, temperature: float=0.7, top_p: float=0.95):
    #     """ Sends a request to vLLM for generating a response """
    #     messages = [
    #         {"role": "user", "content": user_prompt}
    #     ]
    #     payload = {
    #         "model": self.model_ckpt,
    #         "messages": messages,
    #         "max_tokens": MAX_NEW_TOKENS,
    #         "temperature": temperature, 
    #         # greedy decoding for reproducibility
    #         "top_p": top_p,
    #         "seed": self.seed,  
    #         # seed for reproducibility
    #     }

    #     for _ in range(MAX_RETRIES):
    #         try:
    #             response = requests.post(VLLM_SERVER_URL, json=payload).json()
    #             model_response = response["choices"][0]["message"]["content"]
    #             return {
    #                 "model_response": model_response,
    #                 "error": False,
    #             }
    #         except Exception as e: pass
    #             # print(response)
    #             # print(e)
    #     return {
    #         "model_response": f"Got Error: {str(e)}",
    #         "error": True,
    #     }