from dynamicbatch_ttspipeline.env import args
from dynamicbatch_ttspipeline.f5_tts.load import (
    load_f5_tts,
    load_vocoder,
    target_sample_rate,
    hop_length,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
)
from dynamicbatch_ttspipeline.f5_tts.utils import (
    chunk_text,
    convert_char_to_pinyin,
)
from malaya_speech import Pipeline
from transformers import pipeline
import numpy as np
import torch.nn.functional as F
import malaya_speech
import librosa
import torchaudio
import torch
import asyncio
import time
import logging

vad = malaya_speech.vad.webrtc()
model = None
vocoder = None
asr_pipe = None
device = args.device
torch_dtype = getattr(torch, args.torch_dtype)
sr_whisper = 16000

def load_model():
    global model, vocoder, asr_pipe
    """
    1. must use float16
    2. if use bfloat16, default sway_sampling_coef which is `-1` doesnt generate a correct `t` for `odeint(fn, y0, t, **self.odeint_kwargs)`
    """
    model = load_f5_tts(args.model_tts_name, device = device, dtype = torch.float16)
    vocoder = load_vocoder(args.model_vocoder_name, device = device)
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=torch_dtype,
        device=device,
    )
    convert_char_to_pinyin(['helo'])
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
            audios = [batch[i][1] for i in range(len(batch))]
            gen_texts = [batch[i][2] for i in range(len(batch))]
            ref_texts = [batch[i][3] for i in range(len(batch))]
            speeds = [batch[i][4] for i in range(len(batch))]

            audios = [torch.Tensor(audios[i]).to('cuda') for i in range(len(audios))]
            audios_length = [audios[i].shape[0] for i in range(len(audios))]
            maxlen_audios_length = max(audios_length)
            audios = torch.stack([F.pad(audios[i], (maxlen_audios_length - audios_length[i], 0)) for i in range(len(audios))], 0)

            ref_audio_len = audios.shape[-1] // hop_length
            final_text_lists, durations, after_durations = [], [], []
            for i in range(len(gen_texts)):
                ref_text = ref_texts[i]
                gen_text = gen_texts[i]
                dur = audios_length[i] // hop_length
                speed = speeds[i]
                text_list = [ref_text + gen_text]
                final_text_list = convert_char_to_pinyin(text_list)
                ref_text_len = len(ref_text.encode("utf-8"))
                gen_text_len = len(gen_text.encode("utf-8"))
                after_duration = int(dur / ref_text_len * gen_text_len / speed)
                final_text_lists.append(final_text_list[0])
                durations.append(ref_audio_len + after_duration)
                after_durations.append(after_duration)
            
            lengths = [len(l) for l in final_text_lists]
            maxlen = max(lengths)
            batch_final_text_lists = []
            for t in final_text_lists:
                batch_final_text_lists.append(t + ['.'] * (maxlen - len(t)))

            with torch.no_grad():
                generated, _ = model.sample(
                    cond=audios,
                    text=batch_final_text_lists,
                    duration=torch.Tensor(durations).to(device).type(torch.long),
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                )
                generated = generated.to(torch.float32)
                generated = generated[:, ref_audio_len:, :]
                generated_mel_spec = generated.permute(0, 2, 1)
                generated_wave = vocoder.decode(generated_mel_spec)
                generated_wave = generated_wave.cpu().numpy()
            
            actual_after_durations = [d * hop_length for d in after_durations]
            for i in range(len(actual_after_durations)):
                futures[i].set_result((generated_wave[i, :actual_after_durations[i]],))
        
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
    remove_silent_input=False,
    remove_silent_input_threshold=0.1,
    remove_silent_output=False,
    remove_silent_output_threshold=0.1,
    target_rms=0.1,
    cross_fade_duration=0.15,
    speed=1,
    request=None,
):
    if isinstance(request, dict):
        uuid = request['uuid']
    else:
        uuid = request.scope['request']['uuid']
    dwav, sr_ = torchaudio.load(audio_input)
    dwav = dwav.mean(dim=0).numpy()

    if remove_silent_input:
        p = Pipeline()
        pipeline_left = (
            p.map(malaya_speech.generator.frames, frame_duration_ms = 30, sample_rate = 16000)
        )
        pipeline_right = (
            p.map(malaya_speech.resample, old_samplerate = sr_, new_samplerate = 16000)
            .map(malaya_speech.astype.float_to_int)
            .map(malaya_speech.generator.frames, frame_duration_ms = 30, sample_rate = 16000,
                append_ending_trail = False)
            .foreach_map(vad)
        )
        pipeline_left.foreach_zip(pipeline_right).map(
            malaya_speech.combine.without_silent, 
            threshold_to_stop = remove_silent_input_threshold,
            silent_trail = 1000,
        )
        results = p(dwav)
        dwav = results['without_silent']
    
    if (len(dwav) / sr_) > 20:
        logging.warning(f'{uuid} audio input is longer than 20 seconds, clipping short.')
        dwav = np.concatenate([dwav[:int(20 * sr_)], np.zeros(int(0.05 * sr_),)])
    
    if transcription_input is None or len(transcription_input) < 1:
        logging.info(f'{uuid} transcription input is empty, transcribing using whisper.')
        if sr_ != sr_whisper:
            dwav_ = librosa.resample(dwav, orig_sr = sr_, target_sr = sr_whisper)
        else:
            dwav_ = dwav
        future = asyncio.Future()
        await asr_queue.put((future, dwav_))
        transcription_input = await future
        transcription_input = transcription_input[0]
        logging.info(f'{uuid} transcription_input, {transcription_input}')
    
    audio = dwav
    ref_text = transcription_input

    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    rms = np.sqrt(np.mean(np.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr_) * (25 - audio.shape[-1] / sr_))
    gen_text_batches = chunk_text(text, max_chars=max_chars)
    logging.info(f'{uuid} gen_text_batches, {gen_text_batches}')

    if sr_ != target_sample_rate:
        audio = librosa.resample(audio, orig_sr = sr_, target_sr = target_sample_rate)
    
    futures = []
    before = time.time()
    for t in gen_text_batches:
        future = asyncio.Future()
        await step_queue.put((future, audio, t, ref_text, speed))
        futures.append(future)
    
    results = await asyncio.gather(*futures)
    after = time.time()
    results = [r[0] for r in results]
    if cross_fade_duration <= 0:
        results = np.concatenate(results)
    else:
        final_wave = results[0]
        for i in range(1, len(results)):
            prev_wave = final_wave
            next_wave = results[i]
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))
            if cross_fade_samples <= 0:
                final_wave = np.concatenate([prev_wave, next_wave])
                continue
            
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in
            new_wave = np.concatenate(
                [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
            )
            final_wave = new_wave
        results = final_wave

    if rms < target_rms:
        results = results * rms / target_rms

    if remove_silent_output:
        p = Pipeline()
        pipeline_left = (
            p.map(malaya_speech.generator.frames, frame_duration_ms = 30, sample_rate = 16000)
        )
        pipeline_right = (
            p.map(malaya_speech.resample, old_samplerate = target_sample_rate, new_samplerate = 16000)
            .map(malaya_speech.astype.float_to_int)
            .map(malaya_speech.generator.frames, frame_duration_ms = 30, sample_rate = 16000,
                append_ending_trail = False)
            .foreach_map(vad)
        )
        pipeline_left.foreach_zip(pipeline_right).map(
            malaya_speech.combine.without_silent, 
            threshold_to_stop = remove_silent_output_threshold,
            silent_trail = 1000,
        )
        results = p(results)
        results = results['without_silent']
    
    stats = {
        'total_length': results.shape[-1] / target_sample_rate,
        'seconds_per_second': (results.shape[-1] / target_sample_rate) / (after - before),
    }
    return {
        'audio': results,
        'sr': target_sample_rate,
        'stats': stats,
    }
    
    

