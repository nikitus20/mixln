# Mix-LN

This repo contains the pre-release version of Mix-LN algorithm, proposed by [Mix-LN: Unleashing the Power of Deeper Layers by Combining Pre-LN and Post-LN](https://arxiv.org/pdf/2412.13795v1).

We introduce Mix-LN which combines the benefits of Pre-LN and Post-LN to encourage a more balanced training across layers, thereby improving the overall quality of the model.

<div align="center">
  <img src="https://github.com/user-attachments/assets/365b571d-1004-4fff-8878-9af1374da057" alt="Image 2" style="width: 900px; margin: 0 auto;">
</div>

## Abstract

Large Language Models (LLMs) have achieved remarkable success, yet recent findings reveal that their deeper layers often contribute minimally and can be pruned without affecting overall performance. While some view this as an opportunity for model compression, we identify it as a training shortfall rooted in the widespread use of Pre-Layer Normalization (Pre-LN). We demonstrate that Pre-LN, commonly employed in models like GPT and LLaMA, leads to diminished gradient norms in its deeper layers, reducing their effectiveness. In contrast, Post-Layer Normalization (Post-LN) preserves larger gradient norms in deeper layers but suffers from vanishing gradients in earlier layers. To address this, we introduce Mix-LN, a novel normalization technique that combines the strengths of Pre-LN and Post-LN within the same model. Mix-LN applies Post-LN to the earlier layers and Pre-LN to the deeper layers, ensuring more uniform gradient norms across layers. This allows all parts of the network—both shallow and deep layers—to contribute effectively to training. Extensive experiments with various model sizes demonstrate that Mix-LN consistently outperforms both Pre-LN and Post-LN, promoting more balanced, healthier gradient norms throughout the network, and enhancing the overall quality of LLM pre-training. Furthermore, we demonstrate that models pre-trained with Mix-LN learn better compared to those using Pre-LN or Post-LN during supervised fine-tuning, highlighting the critical importance of high-quality deep layers. By effectively addressing the inefficiencies of deep layers in current LLMs, Mix-LN unlocks their potential, enhancing model capacity without increasing model size.

### TODO

- [x] Release LLM training codes.
- [x] Release metric of Performace Drop.
- [ ] Release metric of Angular Distance.
- [ ] Adding Vision Transformer results.

## Quick Start

### Setup

Our repository is built on top of [GaLore](https://github.com/jiaweizzhao/GaLore). You can configure the environment using the following command lines:
```bash
conda create -n mixln python=3.9 -y
conda activate mixln
pip install -r exp_requirements.txt
```

### Benchmark: Pre-Training LLaMA on C4 dataset

`torchrun_main.py` is the main script for training LLaMA models on C4 with Mix-LN. Our benchmark scripts for various sizes of models are in scripts folder. For example, to train a 71m model on C4 with Pre-LN, do the following:

```bash
# LLaMA-71M, Adam, 1 A100, 1 Node
norm_type='pre'
learning_rates=1e-3
export NORM_TYPE=$norm_type

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
    --grad_clipping 1.0 \
    --run_name "71m_res_${norm_type}_lr${learning_rates}" \
    --save_dir "71m_res_${norm_type}_lr${learning_rates}"
```

To train a 71m model on C4 with Mix-LN, do the following:

```bash
# Define the set of learning rates and normalization types
norm_type='post_pre'
learning_rates=1e-3
export NORM_TYPE=$norm_type
export POST_NUM=3

# Function to run a single training task

echo "Training with learning rate: $learning_rates, norm type: $norm_type on GPU $gpu"

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
    --run_name "71m_res_${norm_type}_lr${learning_rates}" \
    --save_dir "71m_res_${norm_type}_lr${learning_rates}"
```

Additionally, you can use the `scripts` folder to run the benchmark for different model sizes and normalization types.

```bash
bash scripts/run_71m.sh post_pre 3 # post_pre means using Mix-LN, 3 means the number of Post-LN layers
bash scripts/run_1b.sh post_pre 6 # post_pre means using Mix-LN, 6 means the number of Post-LN layers
```


### Angular Distance

We make modifications based on https://github.com/sramshetty/ShortGPT/tree/hf-models, mainly to calculate the angular distance between different layers. To calculate the angular distance between two layers, you can run the following command:

```bash
cd utils
python angular_distance.py --model_path <model_path> --n_samples <n_samples>
# python angular_distance.py --model_path meta-llama/Llama-2-7b-hf --n_samples 1000
```

### Performance Drop
Calculate the performance drop after removing different layers. We use [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) to obtain evaluation results. Please refer to its installation instructions to configure `lm_eval``.
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

Then, you can run the following command to remove different layers and save the weights to a new model. The performance drop will be calculated based on the new model:
```bash
# LLaMA2-7B, Remove Layer 1
python layer_remove.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --layer_index 1 \
    --save_path ./llama_7b_removed_1
```

# Transformer Normalization Analysis Toolkit

A toolkit for analyzing and comparing different layer normalization strategies in transformer models. This project provides tools to track token metrics like representation norms, update norms, and cosine similarities, as well as gradient behavior across different normalization techniques.

## Features

- **Comprehensive metrics tracking**: Analyze token representation and gradient behavior across layers
- **Multi-normalization comparison**: Compare Pre-LN, Post-LN, Mix-LN, and other normalization strategies
- **Cross-model analysis**: Compare metrics across different model sizes
- **Visual analysis**: Generate detailed plots for all metrics
- **Recommendations**: Automatically generate performance recommendations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transformer-norm-analysis.git
cd transformer-norm-analysis
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Ensure you have the Mix-LN implementation from the original repository. The analyzer works with the code structure from [Mix-LN](https://github.com/user-attachments/assets/365b571d-1004-4fff-8878-9af1374da057).

## Quick Start

Run analysis on a model with the default settings:

```bash
./run.sh
```

This will analyze the 71M parameter LLaMA model with different normalization techniques and save the results to a timestamped directory.

## Usage

### Basic Usage

Run analysis on a specific model size:

```bash
./run.sh --model 71m --samples 30
```

Available model sizes: 9m, 20m, 35m, 40m, 60m, 71m, 100m, 130m, 250m, 350m, 1b, 3b, 7b

### Skip Gradient Computation

For faster analysis (without gradient metrics):

```bash
./run.sh --model 71m --skip-gradients
```

### Compare Models

Compare metrics across different model sizes:

```bash
python compare.py --all --comprehensive
```

Or compare a specific metric and normalization type:

```bash
python compare.py --all --norm_type "Mix-LN (3)" --metric gradient_norms
```

## Files

- `analyzer.py`: Main analysis script for transformer models
- `run.sh`: Convenient launcher script
- `compare.py`: Tool for cross-model analysis

## Example Output

The toolkit generates:

1. Detailed metric plots showing layer-by-layer behavior
2. Comparison plots across model sizes
3. Recommendations for optimal normalization techniques
4. Summary reports with key findings

## Metrics Explained

- **Token Norms**: Average L2 norm of token representations
- **Update Norms**: How much token representations change between layers
- **Cosine Similarities**: How similar tokens are to each other
- **Gradient Norms**: Magnitude of gradients flowing through the network
- **Weight Update Ratios**: How much weights would change relative to current values

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This work builds on the [Mix-LN](https://arxiv.org/pdf/2412.13795v1) implementation
- Thanks to the original authors for releasing their code
