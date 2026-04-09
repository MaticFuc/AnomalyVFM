## Adding New Datasets

Adding a new dataset to the evaluation pipeline follows a two-step procedure: 
1. Writing the dataset class to handle data loading.
2. Registering the dataset and its categories in the main dataset registry.

---

### Step 1: Writing the Dataset Class

Create a new file for your dataset in the `datasets/` directory (e.g., `datasets/my_new_dataset.py`). Your new class must inherit from `torch.utils.data.Dataset`.

To ensure compatibility with the testing pipeline, your class **must** implement specific initialization parameters and return a strictly formatted dictionary from `__getitem__`.

#### Required `__init__` Parameters
* `path` (str): The root path to the dataset. **Note:** You should store this as an instance variable (e.g., `self.path = path`) so it can be used to calculate relative paths later.
* `category` (str): The specific class or category within the dataset being evaluated (e.g., "bottle", "cable").
* `transform`: The torchvision transforms to apply to the input image. Default to `T.Compose([T.ToTensor()])`.
* `mask_transform`: The torchvision transforms to apply to the ground truth anomaly mask. Default to `T.Compose([T.ToTensor()])`.

#### Required Methods
* `__len__(self)`: Returns the total number of images in the test set.
* `__getitem__(self, index)`: Loads the image and mask at the given index. **Crucially, this must return a dictionary with the following exact keys:**
  * `"image"`: The transformed RGB image tensor.
  * `"mask"`: The transformed grayscale mask tensor (1.0 for anomalous pixels, 0.0 for normal). If the image is normal ("good"), this should be a tensor of zeros matching the image dimensions.
  * `"is_anom"`: An integer or float representing the image-level label (0 for normal, 1 for anomalous).
  * `"path"`: A string representing the relative path of the file (e.g., `/category/test/good/000.png`). This is typically achieved by stripping the root `self.path` from the absolute file path.

#### Boilerplate Template
```python
import glob
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

class MyNewTestDataset(Dataset):
    def __init__(
        self,
        path,
        category,
        transform=T.Compose([T.ToTensor()]),
        mask_transform=T.Compose([T.ToTensor()]),
    ):
        super().__init__()
        # 1. Store path and transforms
        self.path = path
        self.transform = transform
        self.mask_transform = mask_transform
        
        # 2. Gather your test files based on the path and category
        # (Modify this logic depending on how your dataset is structured)
        self.files = list(sorted(glob.glob(f"{path}/{category}/test/*/*.png")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        
        # 1. Determine if image is an anomaly (modify condition as needed)
        is_anom = "good" not in file 
        
        # 2. Load and transform the main image
        img = Image.open(file).convert("RGB")
        img = self.transform(img)
        
        # 3. Load or generate the mask
        if is_anom:
            # Locate corresponding mask file
            mask_file = file.replace("test", "ground_truth").replace(".png", "_mask.png")
            mask = Image.open(mask_file).convert("L")
            mask = T.ToTensor()(mask)
            mask = self.mask_transform(mask)
            mask = torch.where(mask > 0.5, 1.0, 0.0) # Binarize mask
        else:
            # Generate empty mask for normal images
            mask = torch.zeros((1, img.shape[-1], img.shape[-1]))
            mask = self.mask_transform(mask)
            mask = torch.where(mask > 0.5, 1.0, 0.0)

        # 4. Return standard dictionary including the relative path
        sample = {
            "image": img, 
            "mask": mask, 
            "is_anom": float(int(is_anom)),
            "path": file.replace(self.path, "")
        }
        return sample
```