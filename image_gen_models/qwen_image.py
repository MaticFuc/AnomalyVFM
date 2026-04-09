import torch
import torch.nn as nn
from diffusers import QwenImageInpaintPipeline, QwenImagePipeline


class QwenImage:

    def __init__(self, inpaint):
        self.inpaint = inpaint
        if inpaint:
            self.gen = QwenImageInpaintPipeline.from_pretrained(
                "Qwen/Qwen-Image", torch_dtype=torch.bfloat16
            ).to("cuda")
        else:
            self.gen = QwenImagePipeline.from_pretrained(
                "Qwen/Qwen-Image", torch_dtype=torch.bfloat16
            ).to("cuda")
        self.steps = 50

    def __call__(self, txt, **kwargs):
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)

        init_image = kwargs.get("init_image", None)
        bg_mask = kwargs.get("bg_mask", None)

        save_path = kwargs["save_path"]
        seed = kwargs["seed"]

        if self.inpaint:
            bg_mask = bg_mask.to("cpu")
            img = self.gen(
                txt,
                "",
                image=init_image,
                mask_image=bg_mask,
                strength=1.0,
                num_inference_steps=self.steps,
                height=height,
                width=width,
                guidance_scale=4.0,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).images[0]
        else:
            img = self.gen(
                txt,
                "",
                height=height,
                width=width,
                num_inference_steps=self.steps,
                guidance_scale=4.0,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).images[0]
        return img, save_path
