import os
import zipfile
from huggingface_hub import hf_hub_download


repo_id = "MaticFuc/AnomalyVFM_Synthetic_Dataset"
filename = "dataset.zip"
destination_folder = "."

zip_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset"
)

os.makedirs(destination_folder, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(destination_folder)
