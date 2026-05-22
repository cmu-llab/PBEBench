#!/bin/bash

ts=$(date +%Y%m%d_%H%M%S)

TP_SIZE=${3:-2}

nohup python -m vllm.entrypoints.openai.api_server \
  --model ${2} \
  --tokenizer ${2} \
  --dtype auto \
  --port ${1} \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size $TP_SIZE \
  > vllm_logs/vllm_${1}_${ts}.log 2>&1 & echo $! > vllm_logs/vllm_${1}_${ts}.pid