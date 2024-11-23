import torchaudio
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.parametrize import remove_parametrizations
from torchaudio.functional import resample
from torchaudio.transforms import MelSpectrogram
from dynamicbatch_ttspipeline.resemble_enhance.enhancer.inference import load_enhancer
from dynamicbatch_ttspipeline.resemble_enhance.inference import (
    remove_weight_norm_recursively,
    merge_chunks,
)
from dynamicbatch_ttspipeline.env import args
import base64
import io
import soundfile as sf
import asyncio
import time
import logging

model = None
npad = 441
hp = None
sr = None
chunk_length = None
overlap_length = None
hop_length = None

device = 'cpu'
if args.accelerator_type == 'cuda':
    if not torch.cuda.is_available():
        logging.warning('CUDA is not available, fallback to CPU.')
    else:
        device = 'cuda'
dtype = torch.float32


@torch.no_grad()
def load_model():
    global model, hp, sr, chunk_length, overlap_length, hop_length
    model = load_enhancer(run_dir = None, device = device, dtype = dtype)
    model.configurate_(nfe=64, solver='midpoint', lambd=0.9, tau=0.5)
    remove_weight_norm_recursively(model)
    hp = model.hp
    sr = hp.wav_rate
    chunk_seconds = 10.0
    overlap_seconds = 1.0
    chunk_length = int(sr * chunk_seconds)
    overlap_length = int(sr * overlap_seconds)
    hop_length = chunk_length - overlap_length


step_queue = asyncio.Queue()

async def step():
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(args.dynamic_batching_microsleep)
        try:
            need_sleep = True
            batch = []
            while not step_queue.empty():
                try:
                    request = await asyncio.wait_for(step_queue.get(), timeout=1e-9)
                    batch.append(request)
                    if len(batch) >= args.dynamic_batching_speech_enhancement_batch_size:
                        need_sleep = False
                        break
                        
                except asyncio.TimeoutError:
                    break

            if not len(batch):
                continue

            futures = [batch[i][0] for i in range(len(batch))]
            input_audio = [batch[i][1] for i in range(len(batch))]

            with torch.no_grad():
                audios, lengths, abs_maxes = [], [], []
                for chunk in input_audio:
                    lengths.append(chunk.shape[-1])
                    abs_max = chunk.abs().max().clamp(min=1e-7)
                    abs_maxes.append(abs_max)
                    chunk = chunk.type(dtype).to(device)
                    chunk = chunk / abs_max
                    chunk = F.pad(chunk, (0, npad))
                    audios.append(chunk)
                
                audios = pad_sequence(audios, batch_first=True)
                hwav = model(audios).cpu()
                for i in range(len(futures)):
                    futures[i].set_result((hwav[i][:lengths[i]] * abs_maxes[i],))
        
        except Exception as e:
            logging.error(e)
            try:
                futures = [batch[i][0] for i in range(len(batch))]
                for i in range(len(futures)):
                    if not futures[i].done():
                        futures[i].set_exception(e)
            except:
                pass

async def predict(
    file,
    request = None,
):
    dwav, sr_ = torchaudio.load(file)
    dwav = dwav.mean(dim=0)
    dwav = resample(
        dwav,
        orig_freq=sr_,
        new_freq=sr,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )
    futures = []
    before = time.time()
    for start in range(0, dwav.shape[-1], hop_length):
        future = asyncio.Future()
        chunk = dwav[start : start + chunk_length]
        await step_queue.put((future, chunk))
        futures.append(future)
    
    results = await asyncio.gather(*futures)
    after = time.time()
    results = [r[0] for r in results]
    hwav = merge_chunks(results, chunk_length, hop_length, sr=sr, length=dwav.shape[-1])
    buffer = io.BytesIO()
    sf.write(buffer, hwav, samplerate=sr, format='WAV')
    buffer.seek(0)
    audio_binary = buffer.read()
    audio_base64 = base64.b64encode(audio_binary).decode('utf-8')
    stats = {
        'total_length': dwav.shape[-1] / sr,
        'seconds_per_second': (dwav.shape[-1] / sr) / (after - before),
    }
    return {
        'audio': audio_base64,
        'stats': stats,
    }


