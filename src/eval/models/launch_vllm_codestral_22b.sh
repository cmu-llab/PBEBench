#!/bin/bash

ts=$(date +%Y%m%d_%H%M%S)

nohup python -m vllm.entrypoints.openai.api_server \
  --model "mistralai/Codestral-22B-v0.1" \
  --tokenizer "mistralai/Codestral-22B-v0.1" \
  --dtype auto \
  --port ${1} \
  --tensor-parallel-size 1 \
  > vllm_logs/vllm_${1}_${ts}.log 2>&1 & echo $! > vllm_logs/vllm_${1}_${ts}.pid
  
# python -m vllm.entrypoints.openai.api_server --model "mistralai/Codestral-22B-v0.1" --tokenizer "mistralai/Codestral-22B-v0.1" --dtype auto --host 127.0.0.1 --port ${1} --gpu-memory-utilization 0.95
