#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err
#SBATCH --gres=gpu:A100:2          # Request 2 A100 GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=24:00:00

# Load conda
module load anaconda/2023.07
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpt_oss

PORT=8002

python -m vllm.entrypoints.openai.api_server \
    --model "openai/gpt-oss-120b" \
    --tokenizer "openai/gpt-oss-120b" \
    --dtype auto \
    --port ${PORT} \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 &

SERVER_PID=$!

echo "Waiting for vLLM server to become ready..."
while ! nc -z localhost $PORT; do
    sleep 5
done
echo "vLLM server is ready."

python src/eval/models/run_vllm_inference_v2.py \
    --model_ckpt "openai/gpt-oss-120b" \
    --input_path "data/adaptive_balanced_64_hard_50_inputs_2_20_cascade_bbr_promptsfile.json" \
    --output_path "outputs/ada_bal_hard_50_inputs_2_20_cascade_bbr_gpt_oss_120b_16384_multi_32_async.jsonl" \
    --top_p 0.95 \
    --temperature 0.7 \
    --max_new_tokens 16384 \
    --port ${PORT} \
    --num_samples 32

kill $SERVER_PID
