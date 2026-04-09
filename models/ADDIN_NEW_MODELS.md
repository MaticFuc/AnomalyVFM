## Adding New Models

Adding a new backbone model to the codebase follows a straightforward, two-step procedure: 
1. Writing the individual model class.
2. Registering the model in `model.py`.

---

### Step 1: Writing the Model Class

Create a new file for your model in the `models/` directory (e.g., `models/my_new_model.py`). Your new model must inherit from `torch.nn.Module`. 

To ensure compatibility with the rest of the pipeline, your class **must** implement the following attributes and methods:

#### Required Attributes (Set in `__init__`)
* `self.net`: The core pre-trained model architecture.
* `self.feature_dim` (int): The dimension of the output features (e.g., `1024`).
* `self.patch_size` (int): The patch size of the vision model (e.g., `14` or `16`).
* `self.H` (int): The image height passed during initialization.

#### Required Methods
* `__init__(self, height)`: Initializes the network, downloads the pre-trained weights, and sets the required attributes listed above.
* `get_img_transform(self)`: Returns a `torchvision.transforms.Compose` object containing the specific resizing, tensor conversion, and normalization required by your specific model.
* `add_peft(self, r=64, peft_type="lora")`: Injects the Parameter-Efficient Fine-Tuning (PEFT) adapters into `self.net`. You will likely import a helper function from `peft_local.peft_func` to do this.
* `forward(self, x)`: Runs the forward pass. **Crucially, this must return a tuple of `(summary, features)`.** * `summary`: The global class token or pooled representation.
    * `features`: The sequence of patch embeddings in `BxNxD` format.

#### Boilerplate Template
```python
import torch.nn as nn
import torchvision.transforms as T
from peft_local.peft_func import add_peft # or your specific peft function

class MyNewModel(nn.Module):
    def __init__(self, height):
        super().__init__()
        # 1. Load your base model
        self.net = ... 
        
        # 2. Define required attributes
        self.feature_dim = 1024 
        self.patch_size = 14
        self.H = height

    def get_img_transform(self):
        return T.Compose([
            T.Resize((self.H, self.H)),
            T.ToTensor(),
            # Add T.Normalize if your model requires specific mean/std
        ])

    def add_peft(self, r=64, peft_type="lora"):
        add_peft(self.net, r=r, peft_type=peft_type)

    def forward(self, x):
        output = self.net(x)
        # Process output as needed to separate summary and features
        # ...
        return summary, features
```

---

### Step 2: Registering the Model in `model.py`

Once your class is written, you need to expose it to the factory class so the pipeline can route to it. Open `model.py` and make the following three additions:

**1. Import your new model:**
```python
from models.my_new_model import MyNewModel
```

**2. Add it to the `BACKBONES` Enum:**
Give it a clean string identifier. This is the string users will pass via command line arguments.
```python
class BACKBONES(Enum):
    RADIO = "radio"
    DINOV3 = "dinov3"
    DINOV2 = "dinov2"
    SIGLIP2 = "siglip2"
    CLIP = "clip"
    MY_MODEL = "my_model" # <--- Add here
```

**3. Update the `FeatureExtractor` router:**
Add an `elif` block in the `__init__` method of the `FeatureExtractor` to instantiate your model when requested.
```python
class FeatureExtractor(nn.Module):
    def __init__(self, model_name, height=768):
        super().__init__()
        if model_name == BACKBONES.RADIO:
            self.model = RADIO(height=height)
        elif model_name == BACKBONES.DINOV3:
            self.model = DINOv3(height=height)
        elif model_name == BACKBONES.DINOV2:
            self.model = DINOv2(height=height)
        elif model_name == BACKBONES.SIGLIP2:
            self.model = SigLIP2(height=height)
        elif model_name == BACKBONES.CLIP:
            self.model = CLIP(height=height)
        elif model_name == BACKBONES.MY_MODEL:        # <--- Add here
            self.model = MyNewModel(height=height)    # <--- Add here
        else:
            raise Exception("Model not supported")

    def forward(self, x):
        return self.model(x)
```