from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub import HfFileSystem
from vocos.feature_extractors import EncodecFeatures
from dynamicbatch_ttspipeline.bigvgan import model as bigvgan
from vocos import Vocos
from glob import glob
from .cfm import CFM
from .dit import DiT
from .utils import get_tokenizer
import torch
import os
import logging

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.bfloat16 if "cuda" in device and torch.cuda.get_device_properties(device).major >= 6 else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    try:
        del checkpoint
        torch.cuda.empty_cache()
    except:
        pass

    return model.to(device)

def load_f5_tts(
    model_name = 'SWivid/F5-TTS', 
    mel_spec_type = "vocos",
    use_ema = True,
    device = 'cuda',
    dtype = torch.float16,
    custom_model_path = None,
    custom_vocab_path = None,
):
    if custom_model_path is not None and custom_vocab_path is not None:
        logging.info('overriding checkpoint path by using custom model path.')
        ckpt_path = custom_model_path
        vocab_file = custom_vocab_path
    else:
        fs = HfFileSystem()
        checkpoints = fs.glob(os.path.join(model_name, '**', '*.pt'))
        checkpoints.extend(fs.glob(os.path.join(model_name, '**', '*.pth')))
        checkpoints = [f for f in checkpoints if '_bigvgan' not in f and 'full-checkpoint' not in f and 'checkpoints' not in f]
        ckpt_path = checkpoints[0].split(model_name, 1)[1][1:]
        vocab_file = os.path.join(os.path.split(ckpt_path)[0], 'vocab.txt')
        ckpt_path = hf_hub_download(model_name, ckpt_path)
        vocab_file = hf_hub_download(model_name, vocab_file)
    
    tokenizer = "custom"
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)
    return model

    
def load_vocoder(repo_id = 'charactr/vocos-mel-24khz', device='cuda', vocoder_type='vocos'):
    if vocoder_type == 'vocos':
        config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
        model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)

    else:
        vocoder = bigvgan.BigVGAN.from_pretrained(repo_id, use_cuda_kernel=False)
        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    
    return vocoder
    
    