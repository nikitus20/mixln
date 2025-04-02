from transformers import AutoTokenizer
import os

# Create a directory to save the tokenizer
os.makedirs("t5_tokenizer", exist_ok=True)

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# Save the tokenizer files locally
tokenizer.save_pretrained("./t5_tokenizer")