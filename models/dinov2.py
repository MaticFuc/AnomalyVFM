import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

from peft_local.peft_func import add_peft


class DINOv2(nn.Module):

    def __init__(self, height):
        super().__init__()
        self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
        self.feature_dim = 1024
        self.patch_size = 14
        self.H = height

    def get_img_transform(self):
        return T.Compose(
            [
                T.Resize((self.H, self.H)),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def add_peft(self, r=64, peft_type="lora"):
        add_peft(self.net.blocks, r=r, peft_type=peft_type)

    def forward(self, x):
        ftrs = self.net.forward_features(x)
        summary, ftrs = ftrs["x_norm_clstoken"], ftrs["x_norm_patchtokens"]
        return summary, ftrs
