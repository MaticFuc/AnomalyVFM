import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

from peft_local.peft_func import add_peft

from radio_local import RADIOModel


class RADIO(nn.Module):

    def __init__(self, height, use_local = False):
        super().__init__()
        if use_local:
            self.net = RADIOModel()
        else:
            self.net = torch.hub.load(
                "NVlabs/RADIO", "radio_model", version="radio_v2.5-l", skip_validation=True
            )
        
        self.feature_dim = 1024
        self.patch_size = 16
        self.H = height
        self.local_impl = use_local

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

        if self.local_impl:
            return output[0], output[1]
        return output.summary, output.features