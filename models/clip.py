import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoModel

from peft_local.peft_func import add_peft_clip


class CLIP(nn.Module):

    def __init__(self, height):
        super().__init__()
        self.net = AutoModel.from_pretrained(
            "openai/clip-vit-large-patch14-336"
        ).vision_model
        self.feature_dim = 1024
        self.patch_size = 14
        self.H = height

    def get_img_transform(self):
        return T.Compose(
            [
                T.Resize((self.H, self.H)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def add_peft(self, r=64, peft_type="lora"):
        add_peft_clip(self.net, r=r, peft_type=peft_type)

    def forward(self, x):
        ftrs = self.net(x, interpolate_pos_encoding=True).last_hidden_state
        summary, ftrs = ftrs[:, 0, :], ftrs[:, 1:, :]
        return summary, ftrs
