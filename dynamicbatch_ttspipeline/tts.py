from dynamicbatch_ttspipeline.env import args
import torch
import base64
import io
import soundfile as sf
import asyncio
import time
import logging

device = args.device
torch_dtype = getattr(torch, args.torch_dtype)