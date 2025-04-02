#!/usr/bin/env python
"""
Script to download a small subset of the C4 dataset for local use.
This creates a smaller version of the dataset for faster experimentation.
"""

import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

def download_small_c4(num_samples=1000, output_dir="small_c4", seed=42):
    """
    Download a small subset of the C4 dataset and save it locally.
    
    Args:
        num_samples: Number of samples to download
        output_dir: Directory to save the dataset
        seed: Random seed for reproducibility
    """
    print(f"Downloading {num_samples} samples from C4 dataset...")
    
    # Load the full dataset in streaming mode
    c4_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    # Shuffle and take a subset
    c4_dataset = c4_dataset.shuffle(seed=seed)
    
    # Create the small dataset by taking a subset
    small_dataset = []
    for i, example in enumerate(tqdm(c4_dataset, desc="Collecting samples")):
        if i >= num_samples:
            break
        small_dataset.append(example)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to a HuggingFace dataset and save
    from datasets import Dataset
    small_c4 = Dataset.from_list(small_dataset)
    
    # Save the dataset
    small_c4.save_to_disk(output_dir)
    
    print(f"Successfully saved {len(small_dataset)} samples to {output_dir}")
    print(f"To use this dataset with analyzer.py, run with: --local --local-path {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a small subset of the C4 dataset")
    parser.add_argument("--samples", type=int, default=1000, 
                        help="Number of samples to download (default: 1000)")
    parser.add_argument("--output", type=str, default="small_c4",
                        help="Directory to save the dataset (default: small_c4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    download_small_c4(args.samples, args.output, args.seed) 