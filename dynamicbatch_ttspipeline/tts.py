from dynamicbatch_ttspipeline.env import args
import torch
import base64
import io
import soundfile as sf
import asyncio
import time
import logging
import malaya_speech
import numpy as np
import librosa
from malaya_speech import Pipeline
from transformers import pipeline
from dynamicbatch_ttspipeline.f5_tts.load import (
    load_f5_tts,
    load_vocoder,
    target_sample_rate,
)
from dynamicbatch_ttspipeline.f5_tts.utils import (
    chunk_text,
    convert_char_to_pinyin,
)

p = Pipeline()
vad = malaya_speech.vad.webrtc()
pipeline_left = (
    p.map(malaya_speech.generator.frames, frame_duration_ms = 30, sample_rate = 16000)
)
pipeline_right = (
    p.map(malaya_speech.resample, old_samplerate = sr, new_samplerate = 16000)
    .map(malaya_speech.astype.float_to_int)
    .map(malaya_speech.generator.frames, frame_duration_ms = 30, sample_rate = 16000,
         append_ending_trail = False)
    .foreach_map(vad)
)
pipeline_left.foreach_zip(pipeline_right).map(malaya_speech.combine.without_silent, silent_trail = 1000)

model = None
vocoder = None
asr_pipe = None
device = args.device
torch_dtype = getattr(torch, args.torch_dtype)

def load_model():
    global model, vocoder, asr_pipe
    """
    1. must use float16
    2. if use bfloat16, default sway_sampling_coef which is `-1` doesnt generate a correct `t` for `odeint(fn, y0, t, **self.odeint_kwargs)`
    """
    model = load_f5_tts(args.model_tts_name, device = device, dtype = torch.float16)
    vocoder = load_vocoder(device = device)
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=torch_dtype,
        device=device,
    )
    if args.torch_compile:
        logging.info('enabling torch compile for TTS')
        model.transformer.forward = torch.compile(
            model.transformer.forward
        )
        vocoder.forward = torch.compile(
            vocoder.forward,
        )

asr_queue = asyncio.Queue()

async def asr():
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(args.dynamic_batching_microsleep)
        try:
            need_sleep = True
            batch = []
            while not asr_queue.empty():
                try:
                    request = await asyncio.wait_for(asr_queue.get(), timeout=1e-9)
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
                r_asr = asr_pipe(
                    input_audio,
                    chunk_length_s=30,
                    batch_size=8,
                    generate_kwargs={"task": "transcribe"},
                    return_timestamps=False,
                )
                for i in range(len(futures)):
                    futures[i].set_result((r_asr[i]['text'].strip(),))
        
        except Exception as e:
            logging.error(e)
            try:
                futures = [batch[i][0] for i in range(len(batch))]
                for i in range(len(futures)):
                    if not futures[i].done():
                        futures[i].set_exception(e)
            except:
                pass

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
    text,
    audio_input,
    transcription_input,
    remove_silent_input,
    remove_silent_output,
    target_rms=0.1,
    cross_fade_duration=0.15,
    speed=1,
    request=None,
):
    dwav, sr_ = torchaudio.load(audio_input)
    dwav = dwav.mean(dim=0).numpy()
    if remove_silent_input:
        results = p(dwav)
        dwav = results['without_silent']
    
    if len(dwav / sr_) > 20:
        logging.warning('audio input is longer than 20 seconds, clipping short.')
        dwav = np.concatenate([dwav[:20 * sr], np.zeros(int(0.05 * sr),)])
    
    if transcription_input is None:
        logging.info('transcription input is empty, transcribing using whisper.')
        future = asyncio.Future()
        await asr_queue.put((future, dwav))
        transcription_input = await future
        transcription_input = transcription_input[0]
    
    audio = dwav
    ref_text = transcription_input

    rms = np.sqrt(np.mean(np.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr_) * (25 - audio.shape[-1] / sr_))
    gen_text_batches = chunk_text(text, max_chars=max_chars)

    if sr_ != target_sample_rate:
        audio = librosa.resample(audio, orig_sr = sr_, target_sr = target_sample_rate)
    
    audio = torch.Tensor(audio[None,:])
    audio = audio.to(device)
    
    
    

