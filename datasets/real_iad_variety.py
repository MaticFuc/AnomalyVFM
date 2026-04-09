import json

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

class RealIADVarietyTestDataset(Dataset):

    def __init__(self, path, category, transform = T.Compose([T.ToTensor()]), mask_transform=T.Compose([T.ToTensor()])):
        super().__init__()
        self.root = path
        split_path = f"{path}/splits/{category}.json"

        
        
        with open(split_path) as f:
            split = json.load(f)
        self.files = split["test"]
        
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]
        is_anom = file["mask_path"] is not None
        img = Image.open(f'{self.root}/{file["category"]}/{file["image_path"]}').convert("RGB")
        img = self.transform(img)
        if is_anom:
            mask_file = f'{self.root}/{file["category"]}/{file["mask_path"]}'
            mask = Image.open(mask_file).convert("L")
            mask = T.ToTensor()(mask)
            mask = self.mask_transform(mask)
            mask = torch.where(mask > 0.5, 1.0, 0.0)
        else:
            mask = torch.zeros((1, img.shape[-1], img.shape[-1]))
            mask = self.mask_transform(mask)
            mask = torch.where(mask > 0.5, 1.0, 0.0)

        sample = {
            "image": img,
            "mask": mask,
            "is_anom": float(is_anom),
            "path": f"{file['category']}/{file['image_path']}"
        }
        return sample