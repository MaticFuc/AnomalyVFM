import torch.nn as nn

from peft_local.peft_wrapper import PeftTarget, PeftType, create_peft_wrapper


def _add_attn_peft(model, r=4, alpha=1.0, peft_type=PeftType.LORA):
    for name, module in model.named_children():
        if name == "qkv":
            wrapped = create_peft_wrapper(
                layer=module,
                target=PeftTarget.QKV,
                peft_type=peft_type,
                r=r,
                alpha=alpha,
            )
            setattr(model, name, wrapped)
        elif name == "proj":
            wrapped = create_peft_wrapper(
                layer=module,
                target=PeftTarget.PROJ,
                peft_type=peft_type,
                r=r,
                alpha=alpha,
            )
            setattr(model, name, wrapped)
        else:
            _add_attn_peft(module, r=r, alpha=alpha, peft_type=peft_type)


def _add_clip_attn_peft(model, r=4, alpha=1.0, peft_type=PeftType.LORA):
    w_names = ["q_proj", "v_proj", "out_proj"]
    for name, module in model.named_children():
        if name in w_names:
            wrapped = create_peft_wrapper(
                layer=module,
                target=PeftTarget.LIN,
                peft_type=peft_type,
                r=r,
                alpha=alpha,
            )
            setattr(model, name, wrapped)
        else:
            _add_clip_attn_peft(module, r=r, alpha=alpha, peft_type=peft_type)


def _wrap_module_list_children(model, r=4, alpha=1.0):
    for name, module in model.named_children():
        wrapped = create_peft_wrapper(
            layer=module,
            target=PeftTarget.BLOCK,
            peft_type=PeftType.VPT,
            r=r,
            alpha=alpha,
        )
        setattr(model, name, wrapped)


def add_vpt(model, r=8, alpha=1.0, layer_names=["none"]):#DEFAULT_VPT_LAYER_NAMES):
    if isinstance(model, (nn.ModuleList, nn.Sequential)):
        _wrap_module_list_children(model, r=r, alpha=alpha)
        return

    layer_name_set = set(layer_names)
    for name, module in model.named_children():
        if name in layer_name_set:
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                _wrap_module_list_children(module, r=r, alpha=alpha)
            else:
                wrapped = create_peft_wrapper(
                    layer=module,
                    target=PeftTarget.BLOCK,
                    peft_type=PeftType.VPT,
                    r=r,
                    alpha=alpha,
                )
                setattr(model, name, wrapped)
        else:
            add_vpt(module, r=r, alpha=alpha, layer_names=layer_names)


def add_peft(model, r=4, alpha=1.0, peft_type=PeftType.LORA):
    if peft_type in (PeftType.LORA, PeftType.DORA):
        _add_attn_peft(model=model, r=r, alpha=alpha, peft_type=peft_type)
        return
    if peft_type == PeftType.VPT:
        add_vpt(model=model, r=r, alpha=alpha)
        return

    raise ValueError(f"Unsupported PEFT type: {peft_type}")


def add_peft_clip(model, r=4, alpha=1.0, peft_type=PeftType.LORA):
    if peft_type in (PeftType.LORA, PeftType.DORA):
        _add_clip_attn_peft(model=model, r=r, alpha=alpha, peft_type=peft_type)
        return
    if peft_type == PeftType.VPT:
        add_vpt(model=model, r=r, alpha=alpha)
        return

    raise ValueError(f"Unsupported PEFT type: {peft_type}")


def add_lora(model, r=4, alpha=1.0):
    add_peft(model=model, r=r, alpha=alpha, peft_type=PeftType.LORA)


def add_lora_clip(model, r=4, alpha=1.0):
    add_peft_clip(model=model, r=r, alpha=alpha, peft_type=PeftType.LORA)


def add_dora(model, r=4, alpha=1.0):
    add_peft(model=model, r=r, alpha=alpha, peft_type=PeftType.DORA)


def add_dora_clip(model, r=4, alpha=1.0):
    add_peft_clip(model=model, r=r, alpha=alpha, peft_type=PeftType.DORA)


def add_vpt_clip(model, r=8, alpha=1.0, layer_names=["none"]): #DEFAULT_VPT_LAYER_NAMES):
    add_vpt(model=model, r=r, alpha=alpha, layer_names=layer_names)
