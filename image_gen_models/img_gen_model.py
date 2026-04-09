from enum import Enum

import torch
import torch.nn as nn

from image_gen_models.flux import Flux
from image_gen_models.flux2 import Flux2
from image_gen_models.qwen_image import QwenImage
from image_gen_models.z_image import ZImage


class IMG_GEN_MODELS(Enum):
    FLUX = "flux"
    FLUX2 = "flux2"
    QWEN_IMAGE = "qwen_image"
    Z_IMAGE = "zimage"

    def __str__(self):
        return self.value 


class Generator(nn.Module):

    def __init__(self, model_name, inpaint=False):
        super().__init__()
        if model_name == IMG_GEN_MODELS.FLUX:
            self.model = Flux("flux-dev-krea", "cuda", False)
        elif model_name == IMG_GEN_MODELS.FLUX2:
            self.model = Flux2("flux.2-dev", "cuda", False)
        elif model_name == IMG_GEN_MODELS.QWEN_IMAGE:
            self.model = QwenImage(inpaint)
        elif model_name == IMG_GEN_MODELS.Z_IMAGE:
            self.model = ZImage(inpaint)
        else:
            raise Exception("Img Gen Model not supported")

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)
