#!/bin/bash
# Example script to run training with local dataset and tokenizer

# Define the set of learning rates and normalization types
norm_type=${1:-"pre"}  # Default to pre-normalization if not specified
learning_rates=1e-3
export NORM_TYPE=$norm_type
export POST_NUM=${2:-""}  # Default to empty if not specified

echo "Training with learning rate: $learning_rates, norm type: $norm_type, using local data"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=29500 torchrun_main.py \
    --model_config configs/llama_71m.json \
    --lr $learning_rates \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --grad_clipping 0.0 \
    --run_name "71m_local_${norm_type}_lr${learning_rates}" \
    --save_dir "71m_local_${norm_type}_lr${learning_rates}" \
    --use_local_dataset \
    --local_dataset_path "small_c4" \
    --use_local_tokenizer \
    --local_tokenizer_path "t5-base" 