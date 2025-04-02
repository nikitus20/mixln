#!/usr/bin/env python3
"""
Test script to verify that the PreprocessedIterableDataset works correctly with local datasets.
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader

from peft_pretraining.dataloader import PreprocessedIterableDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Test the PreprocessedIterableDataset with local datasets")
    parser.add_argument("--local_dataset_path", type=str, default="small_c4",
                        help="Path to local dataset directory")
    parser.add_argument("--local_tokenizer_path", type=str, default="t5-base",
                        help="Path to local tokenizer directory")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for dataloader")
    parser.add_argument("--num_batches", type=int, default=3,
                        help="Number of batches to process for testing")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Testing PreprocessedIterableDataset with:")
    print(f"  Dataset path: {args.local_dataset_path}")
    print(f"  Tokenizer path: {args.local_tokenizer_path}")
    print(f"  Max length: {args.max_length}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num batches to test: {args.num_batches}")
    
    try:
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer_path, model_max_length=args.max_length)
        print(f"Tokenizer loaded successfully. Vocabulary size: {len(tokenizer)}")
        
        # Load dataset
        print("\nLoading dataset...")
        train_data = load_dataset(
            'arrow', 
            data_files={
                'train': f'{args.local_dataset_path}/train/data-00000-of-00001.arrow',
            },
            split="train"
        )
        print(f"Dataset loaded successfully. Size: {len(train_data)} examples")
        
        # Convert to iterable dataset
        print("\nConverting to iterable dataset...")
        iterable_data = train_data.to_iterable_dataset()
        
        # Create PreprocessedIterableDataset
        print("\nCreating PreprocessedIterableDataset...")
        dataset = PreprocessedIterableDataset(
            data=iterable_data,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        # Create DataLoader
        print("\nCreating DataLoader...")
        dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
        
        # Process batches
        print(f"\nProcessing {args.num_batches} batches...")
        for i, batch in enumerate(tqdm(dataloader, total=args.num_batches)):
            if i >= args.num_batches:
                break
                
            # Print batch information
            print(f"\nBatch {i+1}:")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Attention mask shape: {batch['attention_mask'].shape}")
            
            # Check for NaNs or infinities
            if torch.isnan(batch['input_ids']).any() or torch.isinf(batch['input_ids']).any():
                print("  WARNING: Input IDs contain NaN or Inf values!")
            
            if torch.isnan(batch['attention_mask']).any() or torch.isinf(batch['attention_mask']).any():
                print("  WARNING: Attention mask contains NaN or Inf values!")
                
            # Check types
            print(f"  Input IDs dtype: {batch['input_ids'].dtype}")
            print(f"  Attention mask dtype: {batch['attention_mask'].dtype}")
            
            # First few tokens
            print(f"  First example, first 10 tokens: {batch['input_ids'][0, :10].tolist()}")
            
            # Check for padding
            padding_count = (batch['attention_mask'] == 0).sum().item()
            total_tokens = batch['attention_mask'].numel()
            print(f"  Padding percentage: {padding_count / total_tokens * 100:.2f}%")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 