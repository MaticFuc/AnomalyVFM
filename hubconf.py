dependencies = ['torch', 'torchvision', 'transformers']

from hf_model import AnomalyVFM

def anomalyvfm_radio(pretrained=True, **kwargs):
    model = AnomalyVFM.from_pretrained("MaticFuc/anomalyvfm_radio")    
    return model

def anomalyvfm_dinov2(pretrained=True, **kwargs):
    model = AnomalyVFM.from_pretrained("MaticFuc/anomalyvfm_dinov2")
    return model

def anomalyvfm_clip(pretrained=True, **kwargs):
    model = AnomalyVFM.from_pretrained("MaticFuc/anomalyvfm_clip")
    return model

def anomalyvfm_siglip2(pretrained=True, **kwargs):
    model = AnomalyVFM.from_pretrained("MaticFuc/anomalyvfm_siglip2")
    return model