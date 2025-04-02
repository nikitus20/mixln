from datasets import load_dataset
import os
import tarfile

# Create directories
os.makedirs("small_c4", exist_ok=True)

# Download the pre-packaged small C4 dataset (very fast compared to original C4)
print("Downloading small C4 dataset...")
small_c4 = load_dataset("brando/small-c4-dataset")

# Save all splits to disk in efficient Arrow format
print("Saving training split...")
small_c4["train"].save_to_disk("./small_c4/train")

print("Saving validation split...")
small_c4["validation"].save_to_disk("./small_c4/validation")

print("Saving test split...")
small_c4["test"].save_to_disk("./small_c4/test")

# Create an archive for easy transfer
print("Creating archive for transfer...")
with tarfile.open("small_c4_dataset.tar.gz", "w:gz") as tar:
    tar.add("small_c4")

print(f"Done! Archive created at: {os.path.abspath('small_c4_dataset.tar.gz')}")
print(f"Dataset size: {len(small_c4['train'])} training, {len(small_c4['validation'])} validation, {len(small_c4['test'])} test examples")