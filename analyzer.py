import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from peft_pretraining.modeling_llama import LlamaForCausalLM
import json
from datetime import datetime


class TransformerLayerAnalyzer:
    def __init__(self, model_config, norm_type, post_num=None, device='cuda'):
        """
        Initialize transformer analyzer for a specific normalization type
        
        Args:
            model_config (str): Path to model config file
            norm_type (str): Normalization type (pre, post, post_pre, etc.)
            post_num (int, optional): Number of Post-LN layers for Mix-LN
            device (str): Device to run on
        """
        self.device = device
        self.norm_type = norm_type
        self.post_num = post_num
        
        # Set environment variables for normalization type
        os.environ['NORM_TYPE'] = norm_type
        if post_num is not None:
            os.environ['POST_NUM'] = str(post_num)
        
        # Load model configuration and initialize model
        self.config = AutoConfig.from_pretrained(model_config)
        self.model = LlamaForCausalLM(self.config).to(device)
        
        # Ensure model is in training mode for gradient computation
        self.model.train()
        
        # Get tokenizer (using T5 tokenizer as in the original code)
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        
        # Initialize metric trackers
        self.layer_count = self.config.num_hidden_layers
        
        # Activation metrics - one entry per sample for each layer
        self.token_norms = [[] for _ in range(self.layer_count + 1)]  # +1 for embedding
        self.update_norms = [[] for _ in range(self.layer_count)]
        self.cosine_similarities = [[] for _ in range(self.layer_count + 1)]  # +1 for embedding
        
        # Gradient metrics - one entry per sample for each layer
        self.input_gradient_norms = [[] for _ in range(self.layer_count + 1)]  # +1 for embedding
        
        # For storing intermediate activations and gradients
        self.layer_inputs = {}
        self.layer_gradients = {}
        
        # For storing hook handles
        self.hook_handles = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture inputs and gradients for each layer"""
        # Register forward hooks for the embedding layer and all transformer layers
        def forward_hook(name):
            def hook_fn(module, inputs, outputs):
                # Get input without detaching to preserve computation graph
                input_tensor = inputs[0]
                
                # Store input for activation metrics
                self.layer_inputs[name] = input_tensor
                
                # For non-leaf tensors that require gradients, use retain_grad()
                if input_tensor.requires_grad:
                    input_tensor.retain_grad()  # Enables capturing gradients for non-leaf tensors
                
            return hook_fn
        
        # Register hooks for embedding layer
        handle = self.model.model.embed_tokens.register_forward_hook(forward_hook("embedding"))
        self.hook_handles.append(handle)
        
        # Register hooks for each transformer layer
        for i in range(self.layer_count):
            handle = self.model.model.layers[i].register_forward_hook(forward_hook(f"layer_{i}"))
            self.hook_handles.append(handle)
    
    def _collect_gradients(self):
        """Collect gradients after backward pass for all layers that have them"""
        for name, tensor in self.layer_inputs.items():
            if hasattr(tensor, 'grad') and tensor.grad is not None:
                self.layer_gradients[name] = tensor.grad.detach().clone()
    
    def reset_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    def _compute_token_metrics(self):
        """Compute metrics based on token representations at each layer"""
        # Process embedding and each layer
        for i in range(self.layer_count + 1):
            layer_name = "embedding" if i == 0 else f"layer_{i-1}"
            
            if layer_name in self.layer_inputs:
                # Get token representations (detach now for metrics computation)
                tokens = self.layer_inputs[layer_name].detach()  # [batch_size, seq_len, hidden_dim]
                
                # Skip metrics for input tensors that don't have the right shape
                # Token IDs tensor has shape [batch_size, seq_len] without hidden_dim
                if len(tokens.shape) < 3:
                    continue
                
                # Ensure tokens is a floating point tensor for norm calculation
                if not torch.is_floating_point(tokens):
                    tokens = tokens.float()
                
                # Compute token norms (L2 norm across hidden dimension)
                hidden_dim = tokens.shape[-1]  # Get the last dimension (hidden dim)
                token_norm = torch.norm(tokens, dim=-1).mean().item()  # Use -1 for last dimension
                self.token_norms[i].append(token_norm)
                
                # Compute cosine similarity between tokens in the sequence
                # First, normalize the tokens
                tokens_normalized = tokens / (torch.norm(tokens, dim=-1, keepdim=True) + 1e-8)
                
                # Compute pairwise cosine similarity matrix
                batch_size, seq_len, _ = tokens.shape
                cosine_sim = torch.bmm(tokens_normalized, tokens_normalized.transpose(1, 2))
                
                # Average the upper triangular part (excluding diagonal)
                mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(self.device)
                avg_cosine_sim = (cosine_sim * mask.unsqueeze(0)).sum() / (mask.sum() * batch_size)
                self.cosine_similarities[i].append(avg_cosine_sim.item())
                
                # Compute update norms (if not the last layer)
                if i < self.layer_count:
                    next_layer = f"layer_{i}"
                    if next_layer in self.layer_inputs:
                        next_tokens = self.layer_inputs[next_layer].detach()
                        
                        # Ensure next_tokens has the right shape
                        if len(next_tokens.shape) == 3:
                            # Convert to float if needed
                            if not torch.is_floating_point(next_tokens):
                                next_tokens = next_tokens.float()
                                
                            update = next_tokens - tokens
                            update_norm = torch.norm(update, dim=-1).mean().item()
                            self.update_norms[i].append(update_norm)
    
    def _compute_gradient_metrics(self):
        """Compute metrics based on gradients with respect to layer inputs"""
        for i in range(self.layer_count + 1):
            layer_name = "embedding" if i == 0 else f"layer_{i-1}"
            
            if layer_name in self.layer_gradients:
                grad = self.layer_gradients[layer_name]
                
                # Skip metrics for gradient tensors that don't have the right shape
                if len(grad.shape) < 3:
                    continue
                
                # Ensure grad is a floating point tensor for norm calculation
                if not torch.is_floating_point(grad):
                    grad = grad.float()
                    
                # Compute gradient norm (L2 norm across hidden dimension)
                grad_norm = torch.norm(grad, dim=-1).mean().item()
                self.input_gradient_norms[i].append(grad_norm)
    
    def compute_metrics(self, input_ids):
        """
        Compute metrics for a single batch of inputs
        
        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
        """
        # Clear previous storage
        self.layer_inputs = {}
        self.layer_gradients = {}
        
        # Ensure model is in training mode for gradient computation
        self.model.train()
        
        # Zero gradients and run forward/backward pass
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids.to(self.device), 
            labels=input_ids.to(self.device),
            output_hidden_states=True
        )
        
        # Compute loss and run backward pass to get gradients
        loss = outputs.loss
        loss.backward()
        
        # Collect gradients
        self._collect_gradients()
        
        # Compute metrics
        self._compute_token_metrics()
        self._compute_gradient_metrics()
    
    def process_dataset(self, n_samples=50, max_seq_len=512, batch_size=4, use_local=False, local_path="small_c4"):
        """
        Process samples from the C4 dataset
        
        Args:
            n_samples (int): Number of samples to process
            max_seq_len (int): Maximum sequence length
            batch_size (int): Batch size for processing
            use_local (bool): Whether to use the local dataset
            local_path (str): Path to the local dataset
        """
        # Load dataset - either local or from HuggingFace
        if use_local:
            print(f"Loading local dataset from {local_path}")
            try:
                # Load from local path using arrow files
                data = load_dataset(
                    'arrow', 
                    data_files={
                        'train': f'{local_path}/train/data-00000-of-00001.arrow',
                        'validation': f'{local_path}/validation/data-00000-of-00001.arrow',
                        'test': f'{local_path}/test/data-00000-of-00001.arrow'
                    },
                    split="train"
                )
                # Convert to streaming dataset for consistency with non-local path
                data = data.to_iterable_dataset()
            except Exception as e:
                print(f"Error loading local dataset: {str(e)}")
                print("Falling back to HuggingFace C4 dataset")
                data = load_dataset("allenai/c4", "en", split="train", streaming=True)
        else:
            # Load from HuggingFace
            data = load_dataset("allenai/c4", "en", split="train", streaming=True)
        
        data = data.shuffle(seed=42)
        
        samples_processed = 0
        current_batch = []
        
        # Process dataset in batches
        for example in tqdm(data, desc=f"Processing {self.norm_type}"):
            try:
                # Tokenize the text
                tokenized = self.tokenizer(
                    example["text"],
                    max_length=max_seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                current_batch.append(tokenized["input_ids"])
                
                # Process batch when it reaches the desired size
                if len(current_batch) == batch_size:
                    batch_tensor = torch.cat(current_batch, dim=0)
                    self.compute_metrics(batch_tensor)
                    current_batch = []
                    samples_processed += batch_size
                
                # Stop after processing enough samples
                if samples_processed >= n_samples:
                    break
            except KeyError as e:
                print(f"Skipping an example due to missing key: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing example: {str(e)}")
                continue
        
        # Process any remaining samples
        if current_batch:
            batch_tensor = torch.cat(current_batch, dim=0)
            self.compute_metrics(batch_tensor)
        
        # Clean up hooks to avoid memory leaks
        self.reset_hooks()
    
    def get_average_metrics(self):
        """Calculate average metrics across all processed samples"""
        avg_token_norms = [np.mean(layer_norms) if layer_norms else 0 
                           for layer_norms in self.token_norms]
        avg_update_norms = [np.mean(layer_norms) if layer_norms else 0 
                            for layer_norms in self.update_norms]
        avg_cosine_sims = [np.mean(layer_sims) if layer_sims else 0 
                           for layer_sims in self.cosine_similarities]
        avg_input_grad_norms = [np.mean(layer_norms) if layer_norms else 0 
                                for layer_norms in self.input_gradient_norms]
        
        return {
            'token_norms': avg_token_norms,
            'update_norms': avg_update_norms, 
            'cosine_similarities': avg_cosine_sims,
            'input_gradient_norms': avg_input_grad_norms,
        }


def plot_metrics(metrics_dict, save_dir='plots', model_name="", n_samples=None):
    """
    Plot metrics for different normalization techniques with the requested organization:
    1. For each normalization approach: token norms and update norms together
    2. One plot for all gradient norms across approaches
    3. One plot for all cosine similarities across approaches
    
    Args:
        metrics_dict (dict): Dictionary of metrics for each normalization type
        save_dir (str): Directory to save plots
        model_name (str): Name of the model configuration
        n_samples (int, optional): Number of samples used in the analysis
    """
    os.makedirs(save_dir, exist_ok=True)
    
    norm_types = list(metrics_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(norm_types)))
    
    # Set a global title base for the plots
    main_title_base = f"Model: {model_name}" if model_name else "Model Analysis"
    if n_samples:
        main_title_base += f" (Samples: {n_samples})"
    
    # 1. Individual plots for each normalization approach (token norms + update norms)
    for i, norm_type in enumerate(norm_types):
        if 'token_norms' in metrics_dict[norm_type] and 'update_norms' in metrics_dict[norm_type]:
            fig, ax1 = plt.subplots(figsize=(12, 7))
            
            # Token norms (left y-axis)
            token_layers = range(len(metrics_dict[norm_type]['token_norms']))
            token_color = 'tab:blue'
            ax1.set_xlabel('Layer')
            ax1.set_ylabel('Token Norm', color=token_color)
            ax1.plot(token_layers, metrics_dict[norm_type]['token_norms'], 
                    marker='o', color=token_color, label='Token Norms')
            ax1.tick_params(axis='y', labelcolor=token_color)
            
            # Update norms (right y-axis)
            ax2 = ax1.twinx()
            update_layers = range(len(metrics_dict[norm_type]['update_norms']))
            update_color = 'tab:red'
            ax2.set_ylabel('Update Norm', color=update_color)
            ax2.plot(update_layers, metrics_dict[norm_type]['update_norms'], 
                    marker='s', color=update_color, label='Update Norms')
            ax2.tick_params(axis='y', labelcolor=update_color)
            
            # Add a title
            plt.title(f"{norm_type} Normalization - {main_title_base}")
            
            # Add combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Enable grid
            ax1.grid(True, alpha=0.3)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{norm_type}_token_update_norms.png', dpi=300)
            plt.close()
    
    # 2. One plot for all gradient norms
    if any('input_gradient_norms' in metrics_dict[nt] for nt in norm_types):
        plt.figure(figsize=(12, 7))
        for i, norm_type in enumerate(norm_types):
            if 'input_gradient_norms' in metrics_dict[norm_type]:
                layers = range(len(metrics_dict[norm_type]['input_gradient_norms']))
                plt.plot(layers, metrics_dict[norm_type]['input_gradient_norms'], 
                       marker='o', label=norm_type, color=colors[i])
        
        plt.xlabel('Layer')
        plt.ylabel('Average Input Gradient Norm')
        plt.title(f'Input Gradient Norms - {main_title_base}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visibility
        plt.tight_layout()
        plt.savefig(f'{save_dir}/all_gradient_norms.png', dpi=300)
        plt.close()
    
    # 3. One plot for all cosine similarities
    if any('cosine_similarities' in metrics_dict[nt] for nt in norm_types):
        plt.figure(figsize=(12, 7))
        for i, norm_type in enumerate(norm_types):
            if 'cosine_similarities' in metrics_dict[norm_type]:
                layers = range(len(metrics_dict[norm_type]['cosine_similarities']))
                plt.plot(layers, metrics_dict[norm_type]['cosine_similarities'], 
                       marker='o', label=norm_type, color=colors[i])
        
        plt.xlabel('Layer')
        plt.ylabel('Average Cosine Similarity')
        plt.title(f'Token Cosine Similarities - {main_title_base}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/all_cosine_similarities.png', dpi=300)
        plt.close()
    
    # Also save the original combined metrics plot for reference
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    axes = axes.flatten()  # Flatten to make indexing easier
    
    fig.suptitle(main_title_base, fontsize=16)
    
    # Token norms plot
    for i, norm_type in enumerate(norm_types):
        if 'token_norms' in metrics_dict[norm_type]:
            layers = range(len(metrics_dict[norm_type]['token_norms']))
            axes[0].plot(layers, metrics_dict[norm_type]['token_norms'], 
                       marker='o', label=norm_type, color=colors[i])
    
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Average Token Norm')
    axes[0].set_title('Token Representation Norms')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Update norms plot
    for i, norm_type in enumerate(norm_types):
        if 'update_norms' in metrics_dict[norm_type]:
            layers = range(len(metrics_dict[norm_type]['update_norms']))
            axes[1].plot(layers, metrics_dict[norm_type]['update_norms'], 
                       marker='o', label=norm_type, color=colors[i])
    
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Average Update Norm')
    axes[1].set_title('Layer Update Norms')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Cosine similarities plot
    for i, norm_type in enumerate(norm_types):
        if 'cosine_similarities' in metrics_dict[norm_type]:
            layers = range(len(metrics_dict[norm_type]['cosine_similarities']))
            axes[2].plot(layers, metrics_dict[norm_type]['cosine_similarities'], 
                       marker='o', label=norm_type, color=colors[i])
    
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Average Cosine Similarity')
    axes[2].set_title('Token Cosine Similarities')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Input gradient norms plot
    for i, norm_type in enumerate(norm_types):
        if 'input_gradient_norms' in metrics_dict[norm_type]:
            layers = range(len(metrics_dict[norm_type]['input_gradient_norms']))
            axes[3].plot(layers, metrics_dict[norm_type]['input_gradient_norms'], 
                       marker='o', label=norm_type, color=colors[i])
    
    axes[3].set_xlabel('Layer')
    axes[3].set_ylabel('Average Input Gradient Norm')
    axes[3].set_title('Input Gradient Norms')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_yscale('log')  # Log scale for better visibility
    
    # Save the combined plot
    plt.savefig(f'{save_dir}/all_metrics.png', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze transformer layers with different normalization types')
    parser.add_argument('--model_config', type=str, default='configs/llama_71m.json',
                        help='Path to model configuration file')
    parser.add_argument('--n_samples', type=int, default=30,
                        help='Number of samples to process')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=2, 
                        help='Batch size for processing')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save plots and metrics')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    parser.add_argument('--norm_type', type=str, default="pre",
                        help='Normalization type (pre, post, post_pre, deeppost, sandwich)')
    parser.add_argument('--post_num', type=int, default=None,
                        help='Number of Post-LN layers for Mix-LN')
    parser.add_argument('--local', action='store_true',
                        help='Use local small_c4 dataset instead of HuggingFace')
    parser.add_argument('--local_path', type=str, default="small_c4",
                        help='Path to local dataset (default: small_c4)')
    
    args = parser.parse_args()
    
    # Create a timestamped save directory if not specified
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(args.model_config).replace('.json', '')
        args.save_dir = f"analysis_{model_name}_{timestamp}"
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save the run configuration
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Define normalization configurations to test
    norm_configs = [
        {'type': 'pre', 'post_num': None, 'display_name': 'Pre-LN'},
        {'type': 'post', 'post_num': None, 'display_name': 'Post-LN'},
        {'type': 'post_pre', 'post_num': 3, 'display_name': 'Mix-LN (3)'},
        {'type': 'post_pre', 'post_num': 6, 'display_name': 'Mix-LN (6)'},
        {'type': 'deeppost', 'post_num': None, 'display_name': 'DeepPost'},
        {'type': 'sandwich', 'post_num': None, 'display_name': 'Sandwich'}
    ]
    
    # Process each normalization type
    all_metrics = {}
    
    for config in norm_configs:
        print(f"\nProcessing {config['display_name']} normalization...")
        
        try:
            # Initialize analyzer
            analyzer = TransformerLayerAnalyzer(
                model_config=args.model_config,
                norm_type=config['type'],
                post_num=config['post_num'],
                device=args.device
            )
            
            # Process dataset
            analyzer.process_dataset(
                n_samples=args.n_samples,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                use_local=args.local,
                local_path=args.local_path
            )
            
            # Get metrics
            metrics = analyzer.get_average_metrics()
            all_metrics[config['display_name']] = metrics
            
            # Clear GPU memory
            del analyzer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {config['display_name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Plot metrics
    if all_metrics:
        print("\nPlotting metrics...")
        # Extract model name without path and extension
        model_name = os.path.basename(args.model_config).replace('.json', '')
        plot_metrics(all_metrics, save_dir=args.save_dir, model_name=model_name, n_samples=args.n_samples)
        
        # Save metrics to file
        with open(f"{args.save_dir}/metrics.json", 'w') as f:
            serializable_metrics = {k: {m: v.tolist() if isinstance(v, np.ndarray) else v 
                                       for m, v in metrics.items()} 
                                   for k, metrics in all_metrics.items()}
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"Metrics saved to {args.save_dir}/metrics.json")
        print(f"Plots saved to {args.save_dir}/")
    else:
        print("No metrics were collected. Check for errors above.")


if __name__ == "__main__":
    main()