#!/usr/bin/env python3
"""
Simplified version of the training script that runs on CPU to test the local dataset and tokenizer.
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Test training with local dataset and tokenizer on CPU")
    parser.add_argument("--local_dataset_path", type=str, default="small_c4",
                        help="Path to local dataset directory")
    parser.add_argument("--local_tokenizer_path", type=str, default="t5-base",
                        help="Path to local tokenizer directory")
    parser.add_argument("--model_config", type=str, default="configs/llama_71m.json",
                        help="Path to model configuration file")
    parser.add_argument("--max_length", type=int, default=64,
                        help="Maximum sequence length for tokenization (reduced for CPU)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for dataloader (reduced for CPU)")
    parser.add_argument("--num_training_steps", type=int, default=3,
                        help="Number of training steps to perform")
    parser.add_argument("--norm_type", type=str, default="pre",
                        help="Normalization type (pre, post, post_pre, etc.)")
    parser.add_argument("--post_num", type=int, default=None,
                        help="Number of post-norm layers for Mix-LN")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Testing simplified CPU training with:")
    print(f"  Dataset path: {args.local_dataset_path}")
    print(f"  Tokenizer path: {args.local_tokenizer_path}")
    print(f"  Model config: {args.model_config}")
    print(f"  Max length: {args.max_length}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Training steps: {args.num_training_steps}")
    print(f"  Norm type: {args.norm_type}")
    if args.post_num is not None:
        print(f"  Post num: {args.post_num}")
    
    # Set environment variables for normalization type
    os.environ['NORM_TYPE'] = args.norm_type
    if args.post_num is not None:
        os.environ['POST_NUM'] = str(args.post_num)
    
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
        iterable_data = train_data.to_iterable_dataset().shuffle(seed=42)
        
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
        
        # Load model configuration
        print("\nLoading model configuration...")
        model_config = AutoConfig.from_pretrained(args.model_config)
        
        # Create model
        print("\nCreating model...")
        model = LlamaForCausalLM(model_config)
        print(f"Model created successfully. Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
        # Set model to training mode
        model.train()
        
        # Create optimizer
        print("\nCreating optimizer...")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Train for a few steps
        print(f"\nTraining for {args.num_training_steps} steps...")
        total_loss = 0
        
        for step, batch in enumerate(tqdm(dataloader, total=args.num_training_steps)):
            if step >= args.num_training_steps:
                break
                
            # Create labels for causal language modeling
            labels = batch["input_ids"].clone()
            pad_idx = tokenizer.pad_token_id
            labels[labels == pad_idx] = -100
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Print statistics
            print(f"  Step {step+1}/{args.num_training_steps}, Loss: {loss.item():.4f}")
        
        # Print final statistics
        avg_loss = total_loss / args.num_training_steps
        print(f"\nTraining completed. Average loss: {avg_loss:.4f}")
        
        # Load validation dataset and compute validation loss
        print("\nLoading validation dataset...")
        val_data = load_dataset(
            'arrow', 
            data_files={
                'validation': f'{args.local_dataset_path}/validation/data-00000-of-00001.arrow',
            },
            split="validation"
        )
        val_iterable = val_data.to_iterable_dataset().shuffle(seed=42)
        
        val_dataset = PreprocessedIterableDataset(
            data=val_iterable,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=0)
        
        # Evaluate on a few batches
        print("\nComputing validation loss...")
        model.eval()
        val_loss = 0
        num_val_batches = 2
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader, total=num_val_batches)):
                if i >= num_val_batches:
                    break
                
                # Create labels for causal language modeling
                labels = batch["input_ids"].clone()
                pad_idx = tokenizer.pad_token_id
                labels[labels == pad_idx] = -100
                
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / num_val_batches
        print(f"\nValidation completed. Average validation loss: {avg_val_loss:.4f}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 