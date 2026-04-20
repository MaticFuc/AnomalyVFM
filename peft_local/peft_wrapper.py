from enum import Enum

from peft_local.dora import DoRAQKVWrapper, DoRAWrapper
from peft_local.lora import LoRAQKVWrapper, LoRAWrapper
from peft_local.vpt import VPTWrapper


class PeftType(str, Enum):
    LORA = "lora"
    DORA = "dora"
    VPT = "vpt"

    def __str__(self):
        return self.value

class PeftTarget(str, Enum):
    QKV = "qkv"
    PROJ = "proj"
    LIN = "lin"
    BLOCK = "block"

    def __str__(self):
        return self.value


_WRAPPER_REGISTRY = {
    PeftType.LORA: {
        PeftTarget.QKV: LoRAQKVWrapper,
        PeftTarget.PROJ: LoRAWrapper,
        PeftTarget.LIN: LoRAWrapper,
    },
    PeftType.DORA: {
        PeftTarget.QKV: DoRAQKVWrapper,
        PeftTarget.PROJ: DoRAWrapper,
        PeftTarget.LIN: DoRAWrapper,
    },
    PeftType.VPT: {
        PeftTarget.BLOCK: VPTWrapper,
    },
}


def create_peft_wrapper(layer, target: PeftTarget, peft_type: PeftType = PeftType.LORA, r: int = 4, alpha: float = 1.0):
    try:
        wrapper_cls = _WRAPPER_REGISTRY[peft_type][target]
    except KeyError as exc:
        raise ValueError(f"Unsupported PEFT target {target} for type {peft_type}") from exc
    return wrapper_cls(layer=layer, r=r, alpha=alpha)
