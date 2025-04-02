#!/usr/bin/env python3
"""
Test script to verify that local dataset and tokenizer loading works correctly.
This script tests the functionality we added to torchrun_main.py without running the full training process.
"""

import os
import torch
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Test loading local dataset and tokenizer")
    parser.add_argument("--local_dataset_path", type=str, default="small_c4",
                        help="Path to local dataset directory")
    parser.add_argument("--local_tokenizer_path", type=str, default="t5-base",
                        help="Path to local tokenizer directory")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to process for testing")
    return parser.parse_args()

def test_local_dataset(dataset_path, tokenizer, max_length, num_samples):
    """Test loading and processing the local dataset"""
    print(f"\n=== Testing Local Dataset Loading from {dataset_path} ===")
    
    try:
        # Load the training split
        print("Loading training data...")
        train_data = load_dataset(
            'arrow', 
            data_files={
                'train': f'{dataset_path}/train/data-00000-of-00001.arrow',
            },
            split="train"
        )
        print(f"Training dataset loaded successfully. Size: {len(train_data)} examples")
        
        # Check first example
        first_example = train_data[0]
        print(f"\nFirst example keys: {list(first_example.keys())}")
        if 'text' in first_example:
            print(f"Sample text: {first_example['text'][:100]}...")
        
        # Process some samples
        print(f"\nProcessing {num_samples} samples with tokenizer...")
        for i, example in enumerate(tqdm(train_data.select(range(num_samples)))):
            if 'text' not in example:
                print(f"Warning: Example {i} does not have 'text' field. Keys: {list(example.keys())}")
                continue
                
            tokenized = tokenizer(
                example["text"],
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Show stats for first sample
            if i == 0:
                print(f"\nFirst tokenized example shape: {tokenized['input_ids'].shape}")
                print(f"Number of tokens: {(tokenized['attention_mask'] == 1).sum().item()}")
        
        # Load the validation split
        print("\nLoading validation data...")
        val_data = load_dataset(
            'arrow', 
            data_files={
                'validation': f'{dataset_path}/validation/data-00000-of-00001.arrow',
            },
            split="validation"
        )
        print(f"Validation dataset loaded successfully. Size: {len(val_data)} examples")
        
        return True
        
    except Exception as e:
        print(f"Error loading local dataset: {str(e)}")
        return False

def test_local_tokenizer(tokenizer_path, max_length):
    """Test loading and using the local tokenizer"""
    print(f"\n=== Testing Local Tokenizer Loading from {tokenizer_path} ===")
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=max_length)
        print(f"Tokenizer loaded successfully. Vocabulary size: {len(tokenizer)}")
        
        # Test tokenization
        test_text = "This is a test sentence to check if the tokenizer works properly."
        print(f"\nTokenizing test text: '{test_text}'")
        
        tokenized = tokenizer(
            test_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        print(f"Input IDs shape: {tokenized['input_ids'].shape}")
        print(f"Attention mask shape: {tokenized['attention_mask'].shape}")
        print(f"Number of actual tokens: {(tokenized['attention_mask'] == 1).sum().item()}")
        
        # Decode back
        decoded = tokenizer.decode(tokenized['input_ids'][0])
        print(f"\nDecoded text: '{decoded[:100]}...'")
        
        return tokenizer
        
    except Exception as e:
        print(f"Error loading local tokenizer: {str(e)}")
        return None

def test_to_iterable_dataset(dataset, tokenizer, max_length, num_samples):
    """Test converting to iterable dataset and iterating over it"""
    print("\n=== Testing Conversion to Iterable Dataset ===")
    
    try:
        # Convert to iterable dataset
        print("Converting to iterable dataset...")
        iterable_data = dataset.to_iterable_dataset()
        
        # Process first few samples
        print(f"Processing {num_samples} samples from iterable dataset...")
        
        count = 0
        for example in iterable_data:
            if count >= num_samples:
                break
                
            if 'text' in example:
                tokenized = tokenizer(
                    example["text"],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                if count == 0:
                    print(f"\nFirst tokenized example shape: {tokenized['input_ids'].shape}")
                
            count += 1
            
        print(f"Successfully processed {count} samples from iterable dataset")
        return True
        
    except Exception as e:
        print(f"Error testing iterable dataset: {str(e)}")
        return False

def main():
    args = parse_args()
    
    # Test tokenizer loading
    tokenizer = test_local_tokenizer(args.local_tokenizer_path, args.max_length)
    if tokenizer is None:
        print("Tokenizer loading failed. Exiting.")
        return
    
    # Test dataset loading
    success = test_local_dataset(args.local_dataset_path, tokenizer, args.max_length, args.num_samples)
    if not success:
        print("Dataset loading failed. Exiting.")
        return
        
    # Load dataset for iterable test
    try:
        dataset = load_dataset(
            'arrow', 
            data_files={
                'train': f'{args.local_dataset_path}/train/data-00000-of-00001.arrow',
            },
            split="train"
        )
        
        # Test conversion to iterable dataset
        success = test_to_iterable_dataset(dataset, tokenizer, args.max_length, args.num_samples)
        if not success:
            print("Iterable dataset test failed.")
        
    except Exception as e:
        print(f"Error in iterable dataset test: {str(e)}")
    
    print("\n=== All Tests Completed ===")

if __name__ == "__main__":
    main() 