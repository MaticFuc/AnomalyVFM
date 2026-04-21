import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from transformers import AutoModel

from peft_local.peft_func import add_peft


class TIPSv2(nn.Module):

    def __init__(self, height):
        super().__init__()
        self.net = AutoModel.from_pretrained(
            "google/tipsv2-l14", trust_remote_code=True
        ).vision_encoder
        self.feature_dim = 1024
        self.patch_size = 14
        self.H = height

    def get_img_transform(self):
        return T.Compose(
            [
                T.Resize((self.H, self.H)),
                T.ToTensor(),
            ]
        )

    def add_peft(self, r=64, peft_type="lora"):
        add_peft(self.net, r=r, peft_type=peft_type)

    def forward(self, x):
        cls_1, cls_2, ftrs = self.net(x)
        summary = torch.cat([cls_1, cls_2], dim=2).squeeze(1)
        return summary, ftrs
