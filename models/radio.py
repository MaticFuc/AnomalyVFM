import torch
import torch.nn as nn
import torchvision.transforms as T

from peft_local.peft_func import add_peft


class RADIO(nn.Module):

    def __init__(self, height):
        super().__init__()
        self.net = torch.hub.load(
            "NVlabs/RADIO", "radio_model", version="radio_v2.5-l", skip_validation=True
        )
        self.feature_dim = 1024
        self.patch_size = 16
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
        output = self.net(x)
        return output.summary, output.features
