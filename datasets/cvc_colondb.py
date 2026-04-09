import glob

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class CVC_ColonDBTestDataset(Dataset):

    def __init__(
        self,
        path,
        category,
        transform=T.Compose([T.ToTensor()]),
        mask_transform=T.Compose([T.ToTensor()]),
    ):
        super().__init__()
        self.files = list(sorted(glob.glob(f"{path}/images/*.png")))
        self.transform = transform
        self.mask_transform = mask_transform
        self.st = 0
        self.path = path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        img = Image.open(file).convert("RGB")
        img = self.transform(img)
        is_anom = self.st
        self.st = abs(self.st - 1)

        mask_path = file.replace("images", "masks")
        mask = Image.open(mask_path).convert("L")
        mask = T.ToTensor()(mask)
        mask = self.mask_transform(mask)
        mask = torch.where(mask > 0.5, 1, 0)

        sample = {"image": img, "mask": mask, "is_anom": int(is_anom), "path": file.replace(self.path,"")}
        return sample
