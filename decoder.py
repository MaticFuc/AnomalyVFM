import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        out_dim = in_dim
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.bn1 = nn.GroupNorm(32, out_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.bn2 = nn.GroupNorm(32, out_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DecoderBlockBilinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.bn1 = nn.GroupNorm(32, out_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.bn2 = nn.GroupNorm(32, out_dim)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SimpleDecoder(nn.Module):

    def __init__(self, in_dim, upsample_blocks=2, out_dim=1):
        super().__init__()
        self.blocks = []
        ch = in_dim
        self.bot = BottleNeck(in_dim)
        for _ in range(upsample_blocks):
            self.blocks.append(DecoderBlockBilinear(ch, ch // 2))
            ch = ch // 2
        self.blocks = nn.ModuleList(self.blocks)
        self.final = nn.Conv2d(ch, out_dim, 1)
        self.final_conf = nn.Conv2d(ch, out_dim, 1)

    def forward(self, x):
        x = self.bot(x)
        for b in self.blocks:
            x = b(x)
        mask = self.final(x)
        conf = self.final_conf(x)
        return mask, conf


class SimplePredictor(nn.Module):

    def __init__(self, dim, add_attentive_probing=False):
        super().__init__()
        self.lin = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.lin(x)
        return x
