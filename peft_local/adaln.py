import torch
import torch.nn as nn


class AdaLNWrapper(nn.Module):
    def __init__(self, layer: nn.Module, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.layer = layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0

        self.down = None
        self.up = None

    def _build_adapter(self, in_dim: int, out_dim: int, device, dtype):
        if self.down is not None and self.up is not None:
            return

        self.down = nn.Linear(in_dim, self.r, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(self.r, out_dim, bias=False, device=device, dtype=dtype)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def _adapt(self, x: torch.Tensor, out: torch.Tensor):
        if self.r <= 0:
            return out
        if x.ndim < 2 or out.ndim < 2:
            return out

        in_dim = x.shape[-1]
        out_dim = out.shape[-1]
        self._build_adapter(in_dim, out_dim, x.device, x.dtype)

        delta = self.up(self.down(x))
        if delta.shape == out.shape:
            return out + self.scaling * delta
        return out

    def forward(self, x, *args, **kwargs):
        out = self.layer(x, *args, **kwargs)

        if torch.is_tensor(out):
            return self._adapt(x, out)

        if isinstance(out, tuple) and len(out) > 0 and torch.is_tensor(out[0]):
            adapted_first = self._adapt(x, out[0])
            return (adapted_first,) + out[1:]

        return out
