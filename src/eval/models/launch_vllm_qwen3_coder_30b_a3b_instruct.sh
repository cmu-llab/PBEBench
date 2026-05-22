#!/bin/bash

ts=$(date +%Y%m%d_%H%M%S)

nohup python -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
  --tokenizer "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
  --dtype auto \
  --port ${1} \
  --tensor-parallel-size 2 \
  > vllm_logs/vllm_${1}_${ts}.log 2>&1 & echo $! > vllm_logs/vllm_${1}_${ts}.pid
