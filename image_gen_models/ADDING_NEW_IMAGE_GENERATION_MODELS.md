## Adding New Image Generation Models

Adding a new image generation model (e.g., via Diffusers or custom pipelines) to the evaluation suite follows a straightforward, two-step process:
1. Writing the model wrapper class to handle inference.
2. Registering the model in the main generator router.

---

### Step 1: Writing the Model Wrapper

Create a new file for your model in the `image_gen_models/` directory (e.g., `image_gen_models/my_new_gen.py`). 

Your class acts as an interface between the pipeline's uniform generation requests and your specific model's architecture. It does not strictly need to inherit from `nn.Module` (unless you require it), but it **must** be callable and accept specific keyword arguments.

#### Required Methods
* `__init__(self, ...)`: Initializes the underlying generative model or pipeline. If your model supports distinct standard and inpainting modes, it is common to accept an `inpaint` boolean here.
* `__call__(self, txt, **kwargs)` or `forward(self, txt, **kwargs)`: Executes the generation. 

#### Required `kwargs` Parsing
Your generation method must safely parse and utilize the following parameters from `kwargs`:
* `width` (int): Target image width (defaulting to 1024).
* `height` (int): Target image height (defaulting to 1024).
* `init_image` (Tensor/PIL): An optional initial image for image-to-image or inpainting tasks.
* `bg_mask` (Tensor/PIL): An optional mask for inpainting.
* `save_path` (str): The requested output path for the image.
* `seed` (int): The random seed for reproducibility.

**Crucially, the method must return a tuple of `(img, save_path)`**, where `img` is the generated image (typically a PIL Image).

#### Boilerplate Template
```python
import torch
import torch.nn as nn
# Import your specific generation pipeline, e.g., from diffusers
# from diffusers import MyCustomPipeline 

class MyNewGenModel:
    def __init__(self, inpaint=False):
        self.inpaint = inpaint
        self.steps = 50
        
        # Initialize your underlying model/pipeline here
        if self.inpaint:
            # Load inpaint specific pipeline
            self.gen = ... 
        else:
            # Load standard generation pipeline
            self.gen = ...

    def __call__(self, txt, **kwargs):
        # 1. Parse standardized kwargs
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)
        init_image = kwargs.get("init_image", None)
        bg_mask = kwargs.get("bg_mask", None)
        save_path = kwargs["save_path"]
        seed = kwargs["seed"]

        # 2. Execute your specific model logic
        # (Be sure to handle device placement, masking, and seeds according to your pipeline's API)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        if self.inpaint and bg_mask is not None:
            # Run inpainting generation
            img = self.gen(
                prompt=txt,
                image=init_image,
                mask_image=bg_mask,
                # ...
            )
        else:
            # Run standard text-to-image generation
            img = self.gen(
                prompt=txt,
                width=width,
                height=height,
                # ...
            )

        # 3. Extract the final image and return the required tuple
        # e.g., final_image = img.images[0]
        final_image = img 
        
        return final_image, save_path
```

---

### Step 2: Registering the Model

Once your wrapper is complete, you must expose it to the `Generator` routing class so the main scripts can instantiate it via command line arguments.

Open the file containing the generator registry (which includes the `IMG_GEN_MODELS` enum and `Generator` class) and make the following three additions:

**1. Import your new model wrapper:**
```python
from image_gen_models.my_new_gen import MyNewGenModel
```

**2. Add it to the `IMG_GEN_MODELS` Enum:**
Give it a clear string identifier.
```python
class IMG_GEN_MODELS(Enum):
    FLUX = "flux"
    FLUX2 = "flux2"
    QWEN_IMAGE = "qwen_image"
    Z_IMAGE = "zimage"
    MY_MODEL = "my_model"  # <--- Add here
```

**3. Update the `Generator` router:**
Add an `elif` block in the `__init__` method of the `Generator` class to map the enum to your new class.
```python
class Generator(nn.Module):
    def __init__(self, model_name, inpaint=False):
        super().__init__()
        if model_name == IMG_GEN_MODELS.FLUX:
            self.model = Flux("flux-dev-krea", "cuda", False)
        # ... existing models
        elif model_name == IMG_GEN_MODELS.MY_MODEL:        # <--- Add here
            self.model = MyNewGenModel(inpaint=inpaint)    # <--- Add here
        else:
            raise Exception("Img Gen Model not supported")

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)
```