import glob

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class Br35hTestDataset(Dataset):

    def __init__(
        self,
        path,
        category,
        transform=T.Compose([T.ToTensor()]),
        mask_transform=T.Compose([T.ToTensor()]),
    ):
        super().__init__()
        self.files = list(sorted(glob.glob(f"{path}/yes/*"))) + list(
            sorted(glob.glob(f"{path}/no/*"))
        )
        self.transform = transform
        self.mask_transform = mask_transform
        self.path = path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        img = Image.open(file).convert("RGB")
        img = self.transform(img)
        is_anom = "yes" in file

        mask = torch.ones((1, img.shape[-1], img.shape[-1]))
        mask = self.mask_transform(mask)
        mask = is_anom * mask
        mask = torch.where(mask > 0.5, 1, 0)

        sample = {"image": img, "mask": mask, "is_anom": int(is_anom), "path": file.replace(self.path,"")}
        return sample
