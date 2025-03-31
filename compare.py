import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def load_metrics(metrics_file):
    """Load metrics from a JSON file"""
    if not os.path.exists(metrics_file):
        print(f"Error: Metrics file {metrics_file} not found")
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def plot_normalization_comparison(metrics, metric_name, save_dir, model_name):
    """Plot comparison of a specific metric across normalization types"""
    plt.figure(figsize=(12, 8))
    
    # Extract normalization types
    norm_types = list(metrics.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(norm_types)))
    
    for i, norm_type in enumerate(norm_types):
        if metric_name in metrics[norm_type]:
            metric_data = metrics[norm_type][metric_name]
            layers = range(len(metric_data))
            plt.plot(layers, metric_data, marker='o', label=norm_type, color=colors[i], linewidth=2)
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    plt.title(f'{metric_name.replace("_", " ").title()} Comparison - {model_name} (Random Init)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis to log scale for gradient-related metrics
    if 'gradient' in metric_name:
        plt.yscale('log')
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{metric_name}_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path


def generate_summary(metrics, save_dir, model_name):
    """Generate a summary report of normalization comparison"""
    summary_file = os.path.join(save_dir, "normalization_comparison_summary.txt")
    norm_types = list(metrics.keys())
    
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"NORMALIZATION COMPARISON SUMMARY - {model_name} (Random Init)\n")
        f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Normalization types analyzed: {', '.join(norm_types)}\n\n")
        
        # Token norm analysis
        if all('token_norms' in metrics[norm] for norm in norm_types):
            f.write("TOKEN NORM ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            for norm_type in norm_types:
                token_norms = np.array(metrics[norm_type]['token_norms'])
                first_layer = token_norms[0]
                last_layer = token_norms[-1]
                increase = (last_layer - first_layer) / first_layer * 100
                
                f.write(f"{norm_type}:\n")
                f.write(f"  - First layer token norm: {first_layer:.4f}\n")
                f.write(f"  - Last layer token norm: {last_layer:.4f}\n")
                f.write(f"  - Change: {increase:.1f}%\n")
                
                # Check for stability
                norms_std = np.std(token_norms)
                norms_mean = np.mean(token_norms)
                cv = norms_std / norms_mean  # Coefficient of variation
                
                if cv < 0.1:
                    stability = "Very stable"
                elif cv < 0.2:
                    stability = "Stable"
                elif cv < 0.3:
                    stability = "Moderately stable"
                else:
                    stability = "Unstable"
                
                f.write(f"  - Stability: {stability} (CV: {cv:.2f})\n\n")
        
        # Gradient analysis
        if all('input_gradient_norms' in metrics[norm] for norm in norm_types):
            f.write("\nGRADIENT ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            # Calculate gradient ratios
            grad_ratios = {}
            for norm_type in norm_types:
                grad_norms = np.array(metrics[norm_type]['input_gradient_norms'])
                max_grad = np.max(grad_norms)
                min_grad = np.min(grad_norms)
                ratio = max_grad / (min_grad + 1e-8)
                grad_ratios[norm_type] = ratio
                
                first_grad = grad_norms[0]
                last_grad = grad_norms[-1]
                grad_ratio = first_grad / (last_grad + 1e-8)
                
                f.write(f"{norm_type}:\n")
                f.write(f"  - First layer gradient: {first_grad:.6f}\n")
                f.write(f"  - Last layer gradient: {last_grad:.6f}\n")
                f.write(f"  - First-to-last ratio: {grad_ratio:.2f}\n")
                f.write(f"  - Max-to-min ratio: {ratio:.2f}\n")
                
                # Analyze gradient distribution
                first_half = grad_norms[:len(grad_norms)//2]
                second_half = grad_norms[len(grad_norms)//2:]
                first_half_mean = np.mean(first_half)
                second_half_mean = np.mean(second_half)
                half_ratio = first_half_mean / (second_half_mean + 1e-8)
                
                f.write(f"  - First half to second half ratio: {half_ratio:.2f}\n")
                
                if half_ratio > 5:
                    f.write("  - Early layers dominate gradients\n")
                elif half_ratio < 0.2:
                    f.write("  - Later layers dominate gradients\n")
                else:
                    f.write("  - Relatively balanced gradient distribution\n")
                
                f.write("\n")
            
            # Find best normalization for gradient balance
            best_norm = min(grad_ratios.items(), key=lambda x: x[1])
            f.write(f"Best normalization for gradient balance: {best_norm[0]} (ratio: {best_norm[1]:.2f})\n")
            
            worst_norm = max(grad_ratios.items(), key=lambda x: x[1])
            f.write(f"Worst normalization for gradient balance: {worst_norm[0]} (ratio: {worst_norm[1]:.2f})\n\n")
        
        # Update norms analysis
        if all('update_norms' in metrics[norm] for norm in norm_types):
            f.write("\nUPDATE NORMS ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            for norm_type in norm_types:
                update_norms = np.array(metrics[norm_type]['update_norms'])
                
                f.write(f"{norm_type}:\n")
                f.write(f"  - Average update norm: {np.mean(update_norms):.4f}\n")
                f.write(f"  - Update norm variation: {np.std(update_norms):.4f}\n")
                
                # Find layers with highest updates
                top_layers = np.argsort(update_norms)[-3:][::-1]  # Top 3 layers
                f.write(f"  - Layers with highest updates: {', '.join(map(str, top_layers))}\n\n")
        
        # Cosine similarity analysis
        if all('cosine_similarities' in metrics[norm] for norm in norm_types):
            f.write("\nTOKEN SIMILARITY ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            for norm_type in norm_types:
                cosine_sims = np.array(metrics[norm_type]['cosine_similarities'])
                
                f.write(f"{norm_type}:\n")
                f.write(f"  - Initial token similarity: {cosine_sims[0]:.4f}\n")
                f.write(f"  - Final token similarity: {cosine_sims[-1]:.4f}\n")
                
                # Check if similarity increases significantly
                if cosine_sims[-1] > 0.9:
                    f.write("  - WARNING: High final token similarity, may lead to representation collapse\n")
                elif cosine_sims[-1] - cosine_sims[0] > 0.2:
                    f.write("  - Significant increase in token similarity through layers\n")
                else:
                    f.write("  - Healthy token similarity progression\n")
                
                f.write("\n")
        
        # Overall recommendations
        f.write("\nRECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        
        if 'input_gradient_norms' in metrics[list(metrics.keys())[0]]:
            best_norm = min(grad_ratios.items(), key=lambda x: x[1])[0]
            f.write(f"Based on gradient distribution, {best_norm} appears to be the most balanced normalization type.\n")
            
            # Add specific observations
            for norm_type in norm_types:
                if 'input_gradient_norms' in metrics[norm_type] and 'token_norms' in metrics[norm_type]:
                    grad_norms = np.array(metrics[norm_type]['input_gradient_norms'])
                    token_norms = np.array(metrics[norm_type]['token_norms'])
                    
                    if norm_type == "Pre-LN" and np.mean(grad_norms[len(grad_norms)//2:]) < 0.01 * np.mean(grad_norms[:len(grad_norms)//2]):
                        f.write(f"- {norm_type}: Shows typical gradient vanishing in deeper layers. May limit the contribution of deeper layers during training.\n")
                    
                    elif norm_type == "Post-LN" and np.std(token_norms) / np.mean(token_norms) > 0.3:
                        f.write(f"- {norm_type}: Shows potential training instability due to high token norm variance.\n")
                    
                    elif "Mix-LN" in norm_type and np.mean(grad_norms[len(grad_norms)//2:]) > 0.1 * np.mean(grad_norms[:len(grad_norms)//2]):
                        f.write(f"- {norm_type}: Shows promising balance between stability and deeper layer utilization.\n")
        
        f.write("\nNOTE: This analysis is based on random initialization. Actual training dynamics may differ.\n")
        f.write("Consider monitoring these metrics during training to confirm these observations.\n")
    
    return summary_file


def main():
    parser = argparse.ArgumentParser(description='Compare different normalization techniques on LLaMA model')
    parser.add_argument('--metrics_file', type=str, required=True,
                        help='Path to metrics JSON file')
    parser.add_argument('--model_name', type=str, default="Llama-71M",
                        help='Name of the model being analyzed')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save comparison plots and report')
    
    args = parser.parse_args()
    
    # Create save directory with timestamp if not specified
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"norm_comparison_{args.model_name}_{timestamp}"
    
    # Load metrics
    metrics = load_metrics(args.metrics_file)
    if metrics is None:
        return
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Plot comparisons for each metric
    metric_names = ['token_norms', 'update_norms', 'cosine_similarities', 'input_gradient_norms']
    generated_plots = []
    
    print(f"Generating comparisons for {args.model_name}...")
    
    for metric in metric_names:
        if all(metric in metrics[norm_type] for norm_type in metrics):
            plot_path = plot_normalization_comparison(metrics, metric, args.save_dir, args.model_name)
            generated_plots.append(plot_path)
            print(f"Generated {metric} comparison plot")
    
    # Generate summary report
    summary_file = generate_summary(metrics, args.save_dir, args.model_name)
    print(f"Generated summary report: {summary_file}")
    
    # Create a comprehensive visualization with all metrics
    plt.figure(figsize=(20, 16))
    
    for i, metric in enumerate(metric_names):
        if all(metric in metrics[norm_type] for norm_type in metrics):
            plt.subplot(2, 2, i+1)
            
            norm_types = list(metrics.keys())
            colors = plt.cm.tab10(np.linspace(0, 1, len(norm_types)))
            
            for j, norm_type in enumerate(norm_types):
                metric_data = metrics[norm_type][metric]
                layers = range(len(metric_data))
                plt.plot(layers, metric_data, marker='o', label=norm_type, color=colors[j], linewidth=2)
            
            plt.xlabel('Layer')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(metric.replace('_', ' ').title())
            plt.grid(True, alpha=0.3)
            
            if 'gradient' in metric:
                plt.yscale('log')
            
            # Only show legend on the first subplot
            if i == 0:
                plt.legend(loc='upper right')
    
    plt.suptitle(f'Normalization Comparison - {args.model_name} (Random Init)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    all_metrics_path = os.path.join(args.save_dir, 'all_metrics_comparison.png')
    plt.savefig(all_metrics_path, dpi=300)
    plt.close()
    
    print(f"Generated comprehensive comparison plot: {all_metrics_path}")
    print(f"All comparison results saved to {args.save_dir}")


if __name__ == "__main__":
    main()