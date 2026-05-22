#!/bin/bash

ts=$(date +%Y%m%d_%H%M%S)

nohup python -m vllm.entrypoints.openai.api_server \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --tokenizer "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --dtype auto \
  --port ${1} \
  --tensor-parallel-size 2 \
  --reasoning_parser deepseek_r1 \
  > vllm_logs/vllm_${1}_${ts}.log 2>&1 & echo $! > vllm_logs/vllm_${1}_${ts}.pid