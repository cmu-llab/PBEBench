#!/bin/bash
# Create the directory if not exists
mkdir -p /workspace/hf_cache

export HF_HOME="/workspace/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export HF_XET_CACHE="${HF_HOME}/xet"
# If you want the token stored elsewhere or preserved across runs:
export HF_TOKEN_PATH="${HF_HOME}/token"

# For compatibility with older libraries:
export TRANSFORMERS_CACHE="${HF_HOME}/models"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

python -m vllm.entrypoints.openai.api_server --model "openai/gpt-oss-120b" --tokenizer "openai/gpt-oss-120b" --dtype auto --port ${1} --gpu-memory-utilization 0.95