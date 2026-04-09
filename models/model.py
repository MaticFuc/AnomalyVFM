from enum import Enum

import torch.nn as nn

from models.clip import CLIP
from models.dinov2 import DINOv2
from models.dinov3 import DINOv3
from models.radio import RADIO
from models.siglip2 import SigLIP2


class BACKBONES(Enum):
    RADIO = "radio"
    DINOV3 = "dinov3"
    DINOV2 = "dinov2"
    SIGLIP2 = "siglip2"
    CLIP = "clip"

    def __str__(self):
        return self.value  # For prettier argparse help output


class FeatureExtractor(nn.Module):

    def __init__(self, model_name, height=768):
        super().__init__()
        if model_name == BACKBONES.RADIO:
            self.model = RADIO(height=height)
        elif model_name == BACKBONES.DINOV3:
            self.model = DINOv3(height=height)
        elif model_name == BACKBONES.DINOV2:
            self.model = DINOv2(height=height)
        elif model_name == BACKBONES.SIGLIP2:
            self.model = SigLIP2(height=height)
        elif model_name == BACKBONES.CLIP:
            self.model = CLIP(height=height)
        else:
            raise Exception("Model not supported")

    def forward(self, x):
        return self.model(x)
