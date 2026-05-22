#!/usr/bin/env bash

MAX_GT_CASCADE_LEN=$1
VOCAB="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
DATA_SIZE=64
NUM_INPUTS=50
MAX_ATTEMPTS=$(( MAX_GT_CASCADE_LEN > 20 ? MAX_GT_CASCADE_LEN : 20 ))
PATIENCE=100000

echo "MAX_GT_CASCADE_LEN = $MAX_GT_CASCADE_LEN"
echo "VOCAB              = $VOCAB"
echo "DATA_SIZE          = $DATA_SIZE"
echo "NUM_INPUTS         = $NUM_INPUTS"
echo "MAX_ATTEMPTS       = $MAX_ATTEMPTS"
echo "PATIENCE           = $PATIENCE"

python src/data_generation/improved_generate_with_rejection_sampling.py \
    --size $DATA_SIZE \
    --min-seq-len $MAX_GT_CASCADE_LEN \
    --max-seq-len $MAX_GT_CASCADE_LEN \
    --num-inputs $NUM_INPUTS \
    --output data/adaptive_balanced_64_hard_50_inputs_${MAX_GT_CASCADE_LEN}_cascade_unified_latest.jsonl \
    --min-input-len 2 \
    --max-input-len 6 \
    --stats data/generation_stats.json \
    --vocab $VOCAB \
    --dedupe \
    --patience $PATIENCE

python scripts/validate_and_generate_c4d_promptsfile.py data/adaptive_balanced_64_hard_50_inputs_${MAX_GT_CASCADE_LEN}_cascade_unified_latest.jsonl $NUM_INPUTS $MAX_ATTEMPTS $VOCAB