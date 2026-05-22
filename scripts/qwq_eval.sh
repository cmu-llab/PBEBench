#!/bin/bash
#SBATCH --job-name=qwq_eval
#SBATCH --output=/home/yuweia/pbe-reasoning/log/qwq_eval.log
#SBATCH --error=/home/yuweia/pbe-reasoning/log/qwq_eval.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=72G

export HF_HOME=/data/user_data/yuweia/Huggingface
export HF_HUB_CACHE=/data/hf_cache/hub
export TMPDIR=/scratch
# export NCCL_P2P_DISABLE=1

if [[ "$(hostname)" =~ ^(shire-2-(9|5)|babel-8-5|babel-4-(1|5|9|13|17|21|25|29)|babel-6-(5|9|13|29)|babel-7-(1|5|9)|babel-12-(5|9|13)|babel-13-(1|5|9|13|17|21|25|29)|babel-14-(1|5|9|13|17|21|25|29|37)|babel-5-15|babel-10-17|babel-0-19|babel-11-25|babel-9-3)$ ]]; then
  export NCCL_P2P_DISABLE=1
fi

SOURCE_DIR=/home/yuweia/pbe-reasoning
DATA_DIR=$SOURCE_DIR/data
OUTPUT_DIR=$SOURCE_DIR/output
PROMPT_DIR=$DATA_DIR/prompt

python src/eval/eval.py \
    --model_ckpt Qwen/QwQ-32B \
    --input_path $DATA_DIR/example_data.jsonl \
    --output_path $OUTPUT_DIR/example_data.jsonl \
    --prompt_path $PROMPT_DIR/prompt.txt \
    --tp 2 \
    --max_rules_num 3 \
    --max_tokens_length 2 \
    --character '[a, b, c, d, e]' \
    --batchsize 1 \
    --max_new_tokens 2048 \


