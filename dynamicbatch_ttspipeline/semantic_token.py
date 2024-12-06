from dynamicbatch_ttspipeline.env import args
from dynamicbatch_ttspipeline.fishspeech.load import load_vqgan
import torch

model = None
device = args.device
torch_dtype = getattr(torch, args.torch_dtype)

def load_model():
    global model
    model = load.load_vqgan(device = device)

