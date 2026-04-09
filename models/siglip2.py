import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoModel

from peft_local.peft_func import add_peft_clip


class SigLIP2(nn.Module):

    def __init__(self, height):
        super().__init__()
        self.net = AutoModel.from_pretrained(
            "google/siglip2-so400m-patch16-naflex"
        ).vision_model
        self.feature_dim = 1152
        self.patch_size = 16
        self.H = height

    def get_img_transform(self):
        return T.Compose(
            [
                T.Resize((self.H, self.H)),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

    def add_peft(self, r=64, peft_type="lora"):
        add_peft_clip(self.net, r=r, peft_type=peft_type)

    def forward(self, x):
        num_patches = self.H // self.patch_size
        image = (
            x.reshape(-1, 3, num_patches, self.patch_size, num_patches, self.patch_size)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(
                -1, num_patches * num_patches, self.patch_size * self.patch_size * 3
            )
        )
        inputs = {
            "pixel_values": image,
            "attention_mask": torch.ones(
                image.shape[0], num_patches * num_patches
            ).cuda(),
            "spatial_shapes": torch.tensor([[num_patches, num_patches]])
            .repeat(image.shape[0], 1)
            .cuda(),
        }
        ftrs = self.net(**inputs)
        summary, ftrs = ftrs.pooler_output, ftrs.last_hidden_state
        return summary, ftrs
