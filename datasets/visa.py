import glob

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class VisATestDataset(Dataset):

    def __init__(
        self,
        path,
        category,
        transform=T.Compose([T.ToTensor()]),
        mask_transform=T.Compose([T.ToTensor()]),
    ):
        super().__init__()
        self.files = list(sorted(glob.glob(f"{path}/{category}/test/*/*.JPG")))
        self.transform = transform
        self.mask_transform = mask_transform
        self.path = path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        is_anom = "good" not in file
        img = Image.open(file)
        img = self.transform(img)
        if is_anom:
            mask_file = file.replace("test", "ground_truth").replace(".JPG", ".png")
            mask = Image.open(mask_file).convert("L")
            mask = T.ToTensor()(mask)
            mask = self.mask_transform(mask)
            mask = torch.where(mask > 0.5, 1.0, 0.0)
        else:
            mask = torch.zeros((1, img.shape[-1], img.shape[-1]))
            mask = self.mask_transform(mask)
            mask = torch.where(mask > 0.5, 1.0, 0.0)

        sample = {"image": img, "mask": mask, "is_anom": float(int(is_anom)), "path": file.replace(self.path, "")}
        return sample
