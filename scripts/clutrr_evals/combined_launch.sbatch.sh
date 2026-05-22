#!/bin/bash

set -e

PORT=${1:-8002}
CLIENT_SCRIPT="src/eval/models/run_vllm_inference_v2.py"
MODEL="openai/gpt-oss-120b"

INPUT_PATH="data/clutrr_v1_all_test_promptsfile.json"
OUTPUT_PATH="outputs/clutrr_gpt_oss_120b_8192_async.jsonl"

LOG_DIR="vllm_logs"
mkdir -p ${LOG_DIR}

ts=$(date +%Y%m%d_%H%M%S)
SERVER_LOG="${LOG_DIR}/vllm_${PORT}_${ts}.log"
PID_FILE="${LOG_DIR}/vllm_${PORT}_${ts}.pid"

echo "[INFO] Launching vLLM server on port ${PORT}"
nohup python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tokenizer "${MODEL}" \
    --dtype auto \
    --port ${PORT} \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    > ${SERVER_LOG} 2>&1 &

SERVER_PID=$!
echo ${SERVER_PID} > ${PID_FILE}

echo "[INFO] vLLM server PID ${SERVER_PID}"
echo "[INFO] Logs at ${SERVER_LOG}"
echo "[INFO] Server can take up to 60 mins to load. Polling for readiness."

# Poll server until ready
MAX_MINUTES=60
SLEEP_TIME=5
MAX_RETRIES=$(( (MAX_MINUTES * 60) / SLEEP_TIME ))
COUNT=0

while true; do
    if curl -s "http://localhost:${PORT}/v1/models" > /dev/null; then
        echo "[INFO] Server is ready."
        break
    fi

    # Detect crash
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "[ERROR] Server process died during startup. Check logs at ${SERVER_LOG}"
        exit 1
    fi

    COUNT=$((COUNT + 1))
    if [[ ${COUNT} -ge ${MAX_RETRIES} ]]; then
        echo "[ERROR] Server did not become ready within ${MAX_MINUTES} mins."
        echo "[ERROR] You can raise MAX_MINUTES inside this script."
        exit 1
    fi

    if (( COUNT % 20 == 0 )); then
        mins=$(( (COUNT * SLEEP_TIME) / 60 ))
        echo "[INFO] Still waiting. Elapsed ${mins} mins."
    fi

    sleep ${SLEEP_TIME}
done

echo "[INFO] Running client inference."

python ${CLIENT_SCRIPT} \
    --model_ckpt "${MODEL}" \
    --input_path "${INPUT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --reasoning_effort high \
    --top_p 0.95 \
    --temperature 0.7 \
    --max_new_tokens 8192 \
    --port ${PORT}

echo "[INFO] Client finished."
echo "[INFO] All done."