import torch
import os

# Set environment variable for the Llama model
os.environ['NORM_TYPE'] = 'pre'

from peft_pretraining.modeling_llama import LlamaForCausalLM
from transformers import AutoConfig, AutoTokenizer

# Create a simple model for testing hooks
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 20)
        self.linear2 = torch.nn.Linear(20, 5)
        
    def forward(self, x):
        h1 = self.linear1(x)
        h2 = self.linear2(h1)
        return h2

# Test hooks with a simple model first
print("=== Testing hooks with a simple model ===")
simple_model = SimpleModel()
simple_input = torch.randn(2, 10, requires_grad=True)

# Collection for storing hook data
hook_data = {}

# Test different hook registration methods
def test_forward_hook(name):
    def hook_fn(module, inputs, outputs):
        print(f"{name} hook called")
        print(f" - inputs[0] requires_grad: {inputs[0].requires_grad}")
        print(f" - outputs requires_grad: {outputs.requires_grad}")
        
        # Store the input tensor
        hook_data[f"{name}_input"] = inputs[0]
        
        # Try to register a backward hook on the input
        try:
            hook_handle = inputs[0].register_hook(lambda grad: print(f"{name} grad hook called"))
            print(f" - Successfully registered backward hook on input")
            hook_data[f"{name}_hook_handle"] = hook_handle
        except RuntimeError as e:
            print(f" - Failed to register backward hook on input: {e}")
            
        # Store a detached copy
        hook_data[f"{name}_input_detached"] = inputs[0].detach()
        
        # Try to register hook on detached input
        try:
            detached = inputs[0].detach()
            hook_handle = detached.register_hook(lambda grad: print("This will never be called"))
            print(f" - Successfully registered hook on DETACHED input (unexpected!)")
        except RuntimeError as e:
            print(f" - Failed to register hook on DETACHED input: {e}")
            
    return hook_fn

# Register hooks
simple_model.linear1.register_forward_hook(test_forward_hook("linear1"))
simple_model.linear2.register_forward_hook(test_forward_hook("linear2"))

# Forward and backward pass
output = simple_model(simple_input)
loss = output.sum()
loss.backward()

print("\n=== Testing hooks with the Llama model ===")
# Now try with the Llama model
config = AutoConfig.from_pretrained("configs/llama_71m.json")
llama_model = LlamaForCausalLM(config)
tokenizer = AutoTokenizer.from_pretrained("t5-base")
inputs = tokenizer("Testing hook registration", return_tensors="pt")

# Set up hook collection
llama_hook_data = {}

def llama_layer_hook(name):
    def hook_fn(module, inputs, outputs):
        print(f"{name} hook called")
        print(f" - inputs[0] shape: {inputs[0].shape}")
        print(f" - inputs[0] requires_grad: {inputs[0].requires_grad}")
        
        # Store the input tensor directly (without detaching)
        llama_hook_data[f"{name}_input"] = inputs[0]
        
        # Try to register a backward hook
        try:
            handle = inputs[0].register_hook(lambda grad: print(f"{name} grad hook called"))
            print(f" - Successfully registered backward hook")
            llama_hook_data[f"{name}_hook_handle"] = handle
        except RuntimeError as e:
            print(f" - Failed to register backward hook: {e}")
            
        # Additionally try with retain_grad
        if inputs[0].requires_grad:
            inputs[0].retain_grad()
            print(f" - Called retain_grad() on input tensor")
            
    return hook_fn

# Register hooks on embed_tokens and first layer
llama_model.model.embed_tokens.register_forward_hook(llama_layer_hook("embed_tokens"))
llama_model.model.layers[0].register_forward_hook(llama_layer_hook("layer_0"))

# Forward and backward pass
llama_model.zero_grad()
outputs = llama_model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
loss = outputs.loss
print(f"Loss: {loss.item()}, requires_grad: {loss.requires_grad}")
loss.backward()

# Check gradients
print("\n=== Checking gradients ===")
for name, tensor in llama_hook_data.items():
    if isinstance(tensor, torch.Tensor) and not name.endswith("_hook_handle"):
        has_grad = hasattr(tensor, 'grad') and tensor.grad is not None
        print(f"{name} has gradient: {has_grad}")
        if has_grad:
            print(f" - gradient shape: {tensor.grad.shape}")
            print(f" - gradient mean: {tensor.grad.abs().mean().item()}") 