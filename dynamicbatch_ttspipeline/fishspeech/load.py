from .vqgan.modules.firefly import (
    FireflyArchitecture,
    ConvNeXtEncoder,
    HiFiGANGenerator,
)
from .vqgan.modules.fsq import DownsampleFiniteScalarQuantize
from .utils.spectrogram import LogMelSpectrogram
from huggingface_hub import hf_hub_download
import torch
import requests
import yaml

config_url = 'https://raw.githubusercontent.com/fishaudio/fish-speech/refs/heads/main/fish_speech/configs/firefly_gan_vq.yaml'

def load_vqgan(config_url = config_url, device = 'cuda'):
    r = requests.get(config_url)
    data = yaml.safe_load(r._content)
    data.pop('_target_')
    for k in data.keys():
        data[k].pop('_target_')

    backbone = ConvNeXtEncoder(**data['backbone'])
    head = HiFiGANGenerator(**data['head'])
    quantizer = DownsampleFiniteScalarQuantize(**data['quantizer'])
    spec_transform = LogMelSpectrogram(**data['spec_transform'])
    model = FireflyArchitecture(
        backbone=backbone,
        head=head,
        quantizer=quantizer,
        spec_transform=spec_transform,
    )
    model_path = hf_hub_download(
        repo_id="fishaudio/fish-speech-1.5", 
        filename="firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    )
    state_dict = torch.load(model_path,map_location='cpu', mmap=True, weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    model.to(device)

    return model
    
