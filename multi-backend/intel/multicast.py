import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_some_linear_layer(model):
    if model.config.model_type == "gpt2":
        return model.transformer.h[0].mlp.c_fc
    return model.transformer.h[0].mlp.dense_4h_to_h

class LoRALayer(nn.Module):
    """Wraps a linear layer with LoRA-like adapter - Used for testing purposes only"""

    def __init__(self, module: nn.Module, rank: int):
        super().__init__()
        self.module = module
        self.adapter = nn.Sequential(
            nn.Linear(module.in_features, rank, bias=False),
            nn.Linear(rank, module.out_features, bias=False),
        )
        small_std = (2.0 / (5 * min(module.in_features, module.out_features))) ** 0.5
        nn.init.normal_(self.adapter[0].weight, std=small_std)
        nn.init.zeros_(self.adapter[1].weight)
        self.adapter.to(module.weight.device)

    def forward(self, input, *args, **kwargs):
        return self.module(input, *args, **kwargs) + self.adapter(input)

# Step 1: freeze all parameters
model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)#, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)

for param in model.parameters():
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

# Step 2: add adapters
for _, module in model.named_modules():
    if "OPTAttention" in repr(type(module)):
        print(module.q_proj.weight.dtype)
        module.q_proj = LoRALayer(module.q_proj, rank=16)
        module.k_proj = LoRALayer(module.k_proj, rank=16)
        module.v_proj = LoRALayer(module.v_proj, rank=16)

# Step 3: dummy batch
batch = tokenizer("Test batch ", return_tensors="pt")
# Step 4: Check if the gradient is not None
with torch.autocast("cpu"):
    out = model.forward(**batch)
    out.logits.norm().backward()