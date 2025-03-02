from dynamicbatch_ttspipeline.env import args
from dynamicbatch_ttspipeline.resemble_enhance.enhancer.inference import load_enhancer
from dynamicbatch_ttspipeline.resemble_enhance.inference import (
    remove_weight_norm_recursively,
    merge_chunks,
)
from torch.nn.utils.rnn import pad_sequence
from torchaudio.functional import resample

import torchaudio
import torch
import torch.nn.functional as F
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

device = args.device
# currently SFFT not yet support bfloat16, so we hard coded float32
torch_dtype = torch.float32

def load_model():
    global model, hp, sr, chunk_length, overlap_length, hop_length

    model = load_enhancer(run_dir = None, device = device, dtype = torch_dtype)
    model.configurate_(nfe=64, solver='midpoint', lambd=0.9, tau=0.5)
    remove_weight_norm_recursively(model)
    model.normalizer.eval()
    hp = model.hp
    sr = hp.wav_rate
    chunk_seconds = 15.0
    overlap_seconds = 1.0
    chunk_length = int(sr * chunk_seconds)
    overlap_length = int(sr * overlap_seconds)
    hop_length = chunk_length - overlap_length

    if args.torch_compile:
        logging.info('enabling torch compile for speech enhancement')
        model.lcfm.ae.forward = torch.compile(
            model.lcfm.ae.forward,
        )
        model.lcfm.ae.forward = torch.compile(
            model.lcfm.ae.forward,
        )
        model.lcfm.cfm.emb.forward = torch.compile(
            model.lcfm.cfm.emb.forward,
        )
        model.lcfm.cfm.net.forward = torch.compile(
            model.lcfm.cfm.net.forward,
        )
        model.denoiser.forward = torch.compile(
            model.denoiser.forward,
        )
        model.vocoder.forward = torch.compile(
            model.vocoder.forward,
        )
        """
        torch.Size([2, 441441])
        """
        with torch.no_grad():
            for i in range(1, args.dynamic_batching_speech_enhancement_batch_size + 1):
                audios = torch.zeros(i, chunk_length + npad).to(device)
                logging.info(f'{i}, warming up speech enhancement, {audios.shape}')
                hwav = model(audios)
                del audios, hwav


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
                    chunk = chunk.type(torch_dtype).to(device)
                    chunk = chunk / abs_max
                    if args.torch_compile:
                        n = (chunk_length + npad) - chunk.shape[-1]
                    else:
                        n = npad
                    chunk = F.pad(chunk, (0, n))
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
    request=None,
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
    
    stats = {
        'total_length': dwav.shape[-1] / sr,
        'seconds_per_second': (dwav.shape[-1] / sr) / (after - before),
    }
    return {
        'audio': hwav,
        'sr': sr,
        'stats': stats,
    }


