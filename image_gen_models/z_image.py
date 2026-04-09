import torch
import torch.nn as nn
from diffusers import ZImageInpaintPipeline, ZImagePipeline


class ZImage:

    def __init__(self, inpaint):
        self.inpaint = inpaint
        if inpaint:
            self.gen = ZImageInpaintPipeline.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            ).to("cuda")
        else:
            self.gen = ZImagePipeline.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            ).to("cuda")
        self.steps = 9

    def __call__(self, txt, **kwargs):
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)

        init_image = kwargs.get("init_image", None)
        bg_mask = kwargs.get("bg_mask", None)

        save_path = kwargs["save_path"]
        seed = kwargs["seed"]

        if self.inpaint:
            img = self.gen(
                prompt=txt,
                image=init_image,
                mask_image=bg_mask.unsqueeze(0),
                height=height,
                width=width,
                num_inference_steps=self.steps,
                guidance_scale=4.0,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).images[0]
        else:
            img = self.gen(
                prompt=txt,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                num_inference_steps=self.steps,
                guidance_scale=0.0,
            ).images[0]
        return img, save_path
