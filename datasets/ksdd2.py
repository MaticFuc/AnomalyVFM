import glob

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class KSDD2TestDataset(Dataset):

    def __init__(
        self,
        path,
        category,
        transform=T.Compose([T.ToTensor()]),
        mask_transform=T.Compose([T.ToTensor()]),
    ):
        super().__init__()
        self.files = list(sorted(glob.glob(f"{path}/test/*.png")))
        self.files = list(sorted(filter(lambda x: "_GT.png" not in x, self.files)))
        self.transform = transform
        self.mask_transform = mask_transform
        self.path=path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        img = Image.open(file).convert("RGB")
        img = self.transform(img)
        mask_file = file.replace(".png", "_GT.png")
        mask = Image.open(mask_file).convert("L")
        mask = T.ToTensor()(mask)
        mask = self.mask_transform(mask)
        mask = torch.where(mask > 0.5, 1, 0)

        is_anom = int(torch.any(mask == 1))
        sample = {"image": img, "mask": mask, "is_anom": int(is_anom), "path": file.replace(self.path,"")}
        return sample
