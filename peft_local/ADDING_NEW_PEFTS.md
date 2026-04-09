## Adding New PEFT Methods

Adding a new Parameter-Efficient Fine-Tuning (PEFT) method to the codebase is a three-step process:
1. Writing the layer wrapper classes.
2. Registering the new PEFT type and wrappers in the central registry.
3. Updating the layer injection logic so the pipeline knows how to apply it to models.

---

### Step 1: Writing the Wrapper Classes

Create a new file in the `peft_local/` directory (e.g., `peft_local/my_peft.py`). A PEFT method works by wrapping an existing `nn.Module` (like a Linear layer), adding new trainable parameters, and overriding the `forward` pass.

Depending on your method, you may need different wrappers for different parts of the network (e.g., one for standard linear projections, one for combined QKV layers, or one for entire blocks). 

#### Boilerplate Template
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyPEFTWrapper(nn.Module):
    def __init__(self, layer: nn.Module, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.layer = layer
        self.r = r
        self.alpha = alpha
        
        # 1. Initialize your custom trainable parameters here
        # Make sure to handle the case where r <= 0 (disabled)
        if self.r > 0:
            self.my_param = nn.Parameter(torch.randn(...) * 0.01)
        else:
            self.register_parameter("my_param", None)

    def forward(self, x, *args, **kwargs):
        # 2. Fallback to base layer if PEFT is disabled
        if self.r <= 0:
            return self.layer(x, *args, **kwargs)

        # 3. Compute your PEFT logic
        # Example: adapted_weight = self.layer.weight + self.my_param
        # return F.linear(x, adapted_weight, self.layer.bias)
        
        # Or if wrapping an entire block (like VPT):
        # modified_x = ... 
        # return self.layer(modified_x, *args, **kwargs)
```

---

### Step 2: Registering the Method in `peft_wrapper.py`

Once your wrappers are written, you need to add them to the central factory registry so `create_peft_wrapper` can instantiate them.

Open `peft_local/peft_wrapper.py` and make the following additions:

**1. Import your new wrappers:**
```python
from peft_local.my_peft import MyPEFTWrapper, MyPEFTQKVWrapper # etc.
```

**2. Add to the `PeftType` Enum:**
```python
class PeftType(str, Enum):
    LORA = "lora"
    DORA = "dora"
    VPT = "vpt"
    ADALN = "adaln"
    MYPEFT = "mypeft"  # <--- Add here
```

**3. Update `_WRAPPER_REGISTRY`:**
Map your new `PeftType` to the specific `PeftTarget`s it supports.
```python
_WRAPPER_REGISTRY = {
    # ... existing types ...
    PeftType.MYPEFT: {                               # <--- Add here
        PeftTarget.PROJ: MyPEFTWrapper,
        PeftTarget.QKV: MyPEFTQKVWrapper,
        # Add LIN, BLOCK, or ADALN targets if applicable
    },
}
```

---

### Step 3: Updating the Injection Logic in `peft_func.py`

Finally, tell the codebase how to traverse the model and inject your new wrappers. Open `peft_local/peft_func.py`. 

The steps depend on how your PEFT method targets layers:

**Scenario A: It targets standard Attention/Linear layers (like LoRA/DoRA)**
If your method targets `qkv`, `proj`, `q_proj`, etc., you can simply piggyback off the existing `_add_attn_peft` and `_add_clip_attn_peft` traversal functions. Update the main entry points:

```python
def add_peft(model, r=4, alpha=1.0, peft_type=PeftType.LORA):
    # Add your new type to this tuple:
    if peft_type in (PeftType.LORA, PeftType.DORA, PeftType.MYPEFT): 
        _add_attn_peft(model=model, r=r, alpha=alpha, peft_type=peft_type)
        return
    # ...
```
*(Do the exact same thing for `add_peft_clip`)*.

**Scenario B: It requires custom targeting (like VPT or AdaLN)**
If your method targets specific custom blocks (e.g., `resblocks`), write a new traversal function:

```python
DEFAULT_MYPEFT_LAYER_NAMES = ("my_layer_1", "my_layer_2")

def add_my_peft(model, r=4, alpha=1.0, layer_names=DEFAULT_MYPEFT_LAYER_NAMES):
    layer_name_set = set(layer_names)
    for name, module in model.named_children():
        if name in layer_name_set:
            wrapped = create_peft_wrapper(
                layer=module,
                target=PeftTarget.BLOCK, # Or whatever target you defined
                peft_type=PeftType.MYPEFT,
                r=r,
                alpha=alpha,
            )
            setattr(model, name, wrapped)
        else:
            add_my_peft(module, r=r, alpha=alpha, layer_names=layer_names)
```

Then, hook your custom function into the main entry points (`add_peft` and `add_peft_clip`):
```python
def add_peft(model, r=4, alpha=1.0, peft_type=PeftType.LORA):
    # ... existing checks ...
    if peft_type == PeftType.MYPEFT:
        add_my_peft(model=model, r=r, alpha=alpha)
        return
```

*(Optional)* Create standalone helper functions if needed for consistency:
```python
def add_mypeft(model, r=4, alpha=1.0):
    add_peft(model=model, r=r, alpha=alpha, peft_type=PeftType.MYPEFT)
```