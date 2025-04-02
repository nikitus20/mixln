#!/bin/bash
# Example script to quickly test training with local dataset and tokenizer
# This script uses a very small number of steps to verify everything is working

# Define the set of learning rates and normalization types
norm_type=${1:-"pre"}  # Default to pre-normalization if not specified
learning_rates=1e-3
export NORM_TYPE=$norm_type
export POST_NUM=${2:-""}  # Default to empty if not specified

echo "Running quick test of training with learning rate: $learning_rates, norm type: $norm_type, using local data"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=29500 torchrun_main.py \
    --model_config configs/llama_71m.json \
    --lr $learning_rates \
    --batch_size 4 \
    --total_batch_size 8 \
    --num_training_steps 5 \
    --warmup_steps 2 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 5 \
    --optimizer adam \
    --grad_clipping 0.0 \
    --run_name "test_71m_local_${norm_type}" \
    --save_dir "test_71m_local_${norm_type}" \
    --use_local_dataset \
    --local_dataset_path "small_c4" \
    --use_local_tokenizer \
    --local_tokenizer_path "t5-base" \
    --single_gpu 