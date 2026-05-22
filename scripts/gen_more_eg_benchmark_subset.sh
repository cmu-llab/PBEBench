python src/data_generation/improved_generate_with_rejection_sampling.py \
    --size 240 \
    --min-seq-len 2 \
    --max-seq-len 5 \
    --num-inputs 50 \
    --output data/adaptive_balanced_240_hard_50_inputs.jsonl \
    --stats data/generation_stats.json \
    --vocab abcdefghijkuvwxyz \
    --dedupe  # use --no-dedupe to disable