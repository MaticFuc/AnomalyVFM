import json

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

class RealIAD3DTestDataset(Dataset):

    def __init__(self, path, category, transform = T.Compose([T.ToTensor()]), mask_transform=T.Compose([T.ToTensor()]), use_depth=False):
        super().__init__()
        split_path = f"{path}/realiad_d3_jsons/{category}.json"
        self.root = path
        self.category = category
        with open(split_path) as f:
            split = json.load(f)
        self.files = split["test"]
        self.transform = transform
        self.mask_transform = mask_transform
        self.path = path

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]
        is_anom = "OK" not in file["anomaly_class"]
        img = Image.open(f'{self.root}/realiad_d3_raw/{self.category}/{file["image_path"]}').convert("RGB")
        img = self.transform(img)
        if is_anom:
            mask_file = f'{self.root}/realiad_d3_raw/{self.category}/{file["mask_path"]}'
            mask = Image.open(mask_file).convert("L")
            mask = T.ToTensor()(mask)
            mask = self.mask_transform(mask)
            mask = torch.where(mask > 0.5, 1, 0)
        else:
            mask = torch.zeros((1, img.shape[-1], img.shape[-1]))
            mask = self.mask_transform(mask)
            mask = torch.where(mask > 0.5, 1, 0)

        sample = {
            "image": img,
            "mask": mask,
            "is_anom": int(is_anom),
            "path": f"{self.category}/{file['image_path']}"
        }
        return sample
