#!/bin/bash

# Script to run normalization analysis on Llama-71m model
# Usage: ./run.sh [--samples N] [--batch B] [--seq-len LEN]

# Default parameters
SAMPLES=30
BATCH=2
MAX_SEQ_LEN=512
DEVICE="cpu"  # Explicitly set to CPU
MODEL_CONFIG="configs/llama_71m.json"
MODEL_NAME="Llama-71M"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --samples)
      SAMPLES="$2"
      shift 2
      ;;
    --batch)
      BATCH="$2"
      shift 2
      ;;
    --seq-len)
      MAX_SEQ_LEN="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --samples N         Number of samples to process (default: 30)"
      echo "  --batch B           Batch size (default: 2)"
      echo "  --seq-len LEN       Maximum sequence length (default: 512)"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create timestamp for the run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="analysis_${MODEL_NAME}_${TIMESTAMP}"

# Create save directory
mkdir -p "$SAVE_DIR"

# Run analysis for each normalization type
echo "========================================================"
echo "Starting transformer normalization analysis"
echo "========================================================"
echo "Model: $MODEL_NAME ($MODEL_CONFIG)"
echo "Samples: $SAMPLES"
echo "Batch size: $BATCH"
echo "Save directory: $SAVE_DIR"
echo "Device: $DEVICE"
echo "========================================================"

# Run analyses sequentially
echo "Running analysis with Pre-LN..."
python analyzer.py \
  --model_config "$MODEL_CONFIG" \
  --n_samples "$SAMPLES" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --batch_size "$BATCH" \
  --save_dir "$SAVE_DIR" \
  --device "$DEVICE" \
  --norm_type "pre"

echo "Running analysis with Post-LN..."
python analyzer.py \
  --model_config "$MODEL_CONFIG" \
  --n_samples "$SAMPLES" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --batch_size "$BATCH" \
  --save_dir "$SAVE_DIR" \
  --device "$DEVICE" \
  --norm_type "post"

echo "Running analysis with Mix-LN (3)..."
python analyzer.py \
  --model_config "$MODEL_CONFIG" \
  --n_samples "$SAMPLES" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --batch_size "$BATCH" \
  --save_dir "$SAVE_DIR" \
  --device "$DEVICE" \
  --norm_type "post_pre" \
  --post_num 3

echo "Running analysis with Mix-LN (6)..."
python analyzer.py \
  --model_config "$MODEL_CONFIG" \
  --n_samples "$SAMPLES" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --batch_size "$BATCH" \
  --save_dir "$SAVE_DIR" \
  --device "$DEVICE" \
  --norm_type "post_pre" \
  --post_num 6

echo "Running analysis with DeepPost..."
python analyzer.py \
  --model_config "$MODEL_CONFIG" \
  --n_samples "$SAMPLES" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --batch_size "$BATCH" \
  --save_dir "$SAVE_DIR" \
  --device "$DEVICE" \
  --norm_type "deeppost"

echo "Running analysis with Sandwich..."
python analyzer.py \
  --model_config "$MODEL_CONFIG" \
  --n_samples "$SAMPLES" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --batch_size "$BATCH" \
  --save_dir "$SAVE_DIR" \
  --device "$DEVICE" \
  --norm_type "sandwich"

echo "All analyses complete!"

# Run comparison script
if [ -f "$SAVE_DIR/metrics.json" ]; then
    echo "Generating normalization comparison..."
    python compare.py \
      --metrics_file "$SAVE_DIR/metrics.json" \
      --model_name "$MODEL_NAME" \
      --save_dir "${SAVE_DIR}_comparison"

    echo "Comparison complete! Results saved to ${SAVE_DIR}_comparison"
    echo "To view results: open ${SAVE_DIR}_comparison/all_metrics_comparison.png"
    echo "To read recommendations: cat ${SAVE_DIR}_comparison/normalization_comparison_summary.txt"
else
    echo "Error: No metrics file found at $SAVE_DIR/metrics.json"
    echo "Check individual analysis runs for errors."
fi