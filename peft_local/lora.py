import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAQKVWrapper(nn.Module):
    def __init__(self, layer: nn.Module, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.layer = layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        
        in_dim = self.layer.weight.shape[1]
        out_dim = self.layer.weight.shape[0] // 3

        if r > 0:
            self.lora_A_q = nn.Parameter(torch.randn(r, in_dim) * 0.01)
            self.lora_B_q = nn.Parameter(torch.zeros(out_dim, r))

            self.lora_A_v = nn.Parameter(torch.randn(r, in_dim) * 0.01)
            self.lora_B_v = nn.Parameter(torch.zeros(out_dim, r))
        else:
            self.register_parameter("lora_A_q", None)
            self.register_parameter("lora_B_q", None)
            self.register_parameter("lora_A_v", None)
            self.register_parameter("lora_B_v", None)

    def forward(self, x):
        qkv = self.layer(x)
        if self.r <= 0:
            return qkv

        q, k, v = qkv.chunk(3, dim=-1)
        
        lora_q = F.linear(x, self.lora_A_q)
        lora_q = F.linear(lora_q, self.lora_B_q)

        lora_v = F.linear(x, self.lora_A_v)
        lora_v = F.linear(lora_v, self.lora_B_v)
        
        q = q + self.scaling * lora_q
        v = v + self.scaling * lora_v

        # 4. Re-concatenate and return
        return torch.cat([q, k, v], dim=-1)


class LoRAWrapper(nn.Module):
    def __init__(self, layer: nn.Module, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.layer = layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        
        in_dim = self.layer.weight.shape[1]
        out_dim = self.layer.weight.shape[0]

        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x):
        original_out = self.layer(x)
        if self.r <= 0:
            return original_out

        lora_out = F.linear(x, self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        
        return original_out + self.scaling * lora_out