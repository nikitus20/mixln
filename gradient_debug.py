import torch
from transformers import AutoConfig, AutoTokenizer
from peft_pretraining.modeling_llama import LlamaForCausalLM

# Load model
config = AutoConfig.from_pretrained("configs/llama_71m.json")
model = LlamaForCausalLM(config)

# Check if model is in training mode
print(f"Model training mode: {model.training}")

# Check requires_grad for parameters
params_require_grad = [p.requires_grad for p in model.parameters()]
print(f"Parameters requiring gradients: {sum(params_require_grad)}/{len(params_require_grad)}")

# Create a sample input
tokenizer = AutoTokenizer.from_pretrained("t5-base")
inputs = tokenizer("This is a test", return_tensors="pt")

# Run model with gradient tracking
model.zero_grad()
outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"], output_hidden_states=True)

# Check hidden states
hidden_states = outputs.hidden_states
for i, hidden_state in enumerate(hidden_states):
    print(f"Layer {i} hidden state requires_grad: {hidden_state.requires_grad}")

# Try to compute gradients
loss = outputs.loss
print(f"Loss requires_grad: {loss.requires_grad}")
if loss.requires_grad:
    loss.backward()
    
    # Check if gradients were computed for hidden states
    for i, hidden_state in enumerate(hidden_states):
        has_grad = hasattr(hidden_state, 'grad') and hidden_state.grad is not None
        print(f"Layer {i} hidden state has gradient: {has_grad}")

# Check hook registration on first layer input
if len(hidden_states) > 1:
    first_layer_input = hidden_states[0]
    print(f"\nAnalyzing first layer input (embedding output):")
    print(f"  - requires_grad: {first_layer_input.requires_grad}")
    print(f"  - is leaf tensor: {first_layer_input.is_leaf}")
    print(f"  - has gradient function: {first_layer_input.grad_fn is not None}")
    
    # Try to register a hook
    try:
        handle = first_layer_input.register_hook(lambda grad: print("Hook called with grad"))
        print("  - Successfully registered hook")
        handle.remove()  # Clean up
    except RuntimeError as e:
        print(f"  - Failed to register hook: {str(e)}")

# Try explicitly setting requires_grad
print("\nTrying with explicit requires_grad=True:")
# Get embedding output and try setting requires_grad manually
embedding_output = model.model.embed_tokens(inputs["input_ids"])
embedding_output.requires_grad_(True)
print(f"  - embedding output requires_grad: {embedding_output.requires_grad}")

# Try hooking this tensor
try:
    handle = embedding_output.register_hook(lambda grad: print("Hook called on explicit grad"))
    print("  - Successfully registered hook on explicit tensor")
    handle.remove()  # Clean up
except RuntimeError as e:
    print(f"  - Failed to register hook on explicit tensor: {str(e)}")

# Let's also try evaluation vs training mode
print("\nComparing eval vs train mode:")
model.eval()
print(f"  - Model in eval mode: {not model.training}")
eval_outputs = model(input_ids=inputs["input_ids"], output_hidden_states=True)
eval_hidden = eval_outputs.hidden_states[0]
print(f"  - Hidden state in eval mode requires_grad: {eval_hidden.requires_grad}")

model.train()
print(f"  - Model in train mode: {model.training}")
train_outputs = model(input_ids=inputs["input_ids"], output_hidden_states=True)
train_hidden = train_outputs.hidden_states[0]
print(f"  - Hidden state in train mode requires_grad: {train_hidden.requires_grad}") 