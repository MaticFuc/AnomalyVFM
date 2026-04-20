import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from decoder import SimpleDecoder, SimplePredictor
from models.model import BACKBONES, FeatureExtractor
from peft_local.peft_func import PeftType

from transformers import PretrainedConfig


class AnomalyVFMConfig(PretrainedConfig):
    model_type = "anomalyvfm"
    def __init__(self, model_name=BACKBONES.RADIO, image_size=768, peft_type=PeftType.DORA, peft_rank=64, **kwargs):
        super().__init__(**kwargs)
        self.model_name = str(model_name)
        self.image_size = image_size
        self.peft_type = str(peft_type)
        self.peft_rank = peft_rank

class AnomalyVFM(nn.Module, 
        PyTorchModelHubMixin,
        repo_url="https://github.com/MaticFuc/AnomalyVFM",
        paper_url="https://arxiv.org/abs/2601.20524",
        ):
    config_class = AnomalyVFMConfig

    def __init__(self, config: AnomalyVFMConfig):
        super().__init__()
        if isinstance(config, AnomalyVFMConfig):
            self.config = config
        else:
            self.config=AnomalyVFMConfig(**config)
        self.model = FeatureExtractor(self.config.model_name, self.config.image_size, use_local=True).model
        self.model.add_peft(self.config.peft_rank, peft_type=self.config.peft_type)
        feat_dim = self.model.feature_dim
        self.feat_size = self.config.image_size // self.model.patch_size

        self.decoder = SimpleDecoder(feat_dim, 1, 1)

        if self.config.model_name == BACKBONES.RADIO:
            self.predictor = SimplePredictor(feat_dim * 3)
        else:
            self.predictor = SimplePredictor(feat_dim)

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        self.config.save_pretrained(save_directory)

    def forward(self, img):
        B = img.shape[0]

        device_type = img.device.type
        
        with torch.autocast(device_type = device_type, dtype=torch.bfloat16):
            with torch.no_grad():
                summary, ftrs = self.model(img)
                ftrs = ftrs.permute(0, 2, 1)
                ftrs = ftrs.reshape(B, -1, self.feat_size, self.feat_size)

                anomaly_score = self.predictor(summary).sigmoid()
                anomaly_mask, _ = self.decoder(ftrs)
                anomaly_mask = anomaly_mask.sigmoid()

        return anomaly_score.float(), anomaly_mask.float()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
    
        clean_dict = {}
        for k, v in state_dict.items():
            clean_dict[k] = v.clone().contiguous()
            
        return clean_dict
