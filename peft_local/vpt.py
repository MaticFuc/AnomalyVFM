import torch
import torch.nn as nn


class VPTWrapper(nn.Module):
    def __init__(self, layer: nn.Module, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.layer = layer
        self.prompt_tokens = max(int(r), 0)
        self.alpha = alpha
        self._prompts = None

    def _ensure_prompts(self, dim: int, device, dtype):
        if self._prompts is not None:
            return
        prompt = torch.randn(self.prompt_tokens, dim, device=device, dtype=dtype) * 0.01
        self._prompts = nn.Parameter(prompt)

    def forward(self, x, *args, **kwargs):
        if self.prompt_tokens <= 0 or x.ndim != 3:
            return self.layer(x, *args, **kwargs)

        self._ensure_prompts(x.shape[-1], x.device, x.dtype)
        prompts = self._prompts.unsqueeze(0).expand(x.shape[0], -1, -1)
        x_with_prompt = torch.cat([prompts, x], dim=1)

        out = self.layer(x_with_prompt, *args, **kwargs)

        if torch.is_tensor(out) and out.ndim == 3 and out.shape[1] >= self.prompt_tokens:
            return out[:, self.prompt_tokens :, :]

        return out
