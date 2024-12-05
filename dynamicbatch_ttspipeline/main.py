from fastapi import FastAPI, Request
from fastapi import File, Form, UploadFile
from fastapi import HTTPException
from fastapi.responses import FileResponse
from transformers_openai.middleware import InsertMiddleware
from dynamicbatch_ttspipeline.env import args
from dynamicbatch_ttspipeline.function import return_type
from dynamicbatch_ttspipeline.speech_enhancement import (
    load_model as speech_enhancement_load_model, 
    predict as speech_enhancement_predict,
    step as speech_enhancement_step,
)
from dynamicbatch_ttspipeline.tts import (
    load_model as tts_load_model, 
    predict as tts_predict,
    asr as tts_asr,
    step as tts_step,
)
import uvicorn
import asyncio
import logging

app = FastAPI()
app.add_middleware(InsertMiddleware, max_concurrent=args.max_concurrent)

if args.enable_speech_enhancement:
    logging.info('enabling speech enhancement')

    @app.post('/speech_enhancement')
    async def speech_enhancement(
        file: bytes = File(),
        file_response: bool = True,
        response_format: str = 'mp3',
        request: Request = None,
    ):
        """
        Speech enhancement for audio file. Support return as file or base64 string.
        """
        r = await speech_enhancement_predict(file=file, request=request)
        return return_type(file_response, response_format, r)

    speech_enhancement_load_model()

    @app.on_event("startup")
    async def startup_event():
        app.state.background_speech_enhancement_step = asyncio.create_task(speech_enhancement_step())

    @app.on_event("shutdown")
    async def shutdown_event():
        app.state.background_speech_enhancement_step.cancel()
        try:
            await app.state.background_speech_enhancement_step
        except asyncio.CancelledError:
            pass

if args.enable_tts:
    logging.info('enabling TTS')

    @app.post('/tts')
    async def tts(
        text: str,
        audio_input: bytes = File(),
        transcription_input: str = None,
        remove_silent_input: bool = False,
        remove_silent_output: bool = False,
        target_rms: float = 0.1,
        cross_fade_duration: float = 0.15,
        speed: float = 1,
        file_response: bool = True,
        response_format: str = 'mp3',
        request: Request = None,
    ):
        """
        Text to Speech with voice cloning, only use text normalization natively provided by the model. 
        Support return as file or base64 string.
        """
        r = await tts_predict(
            text=text,
            audio_input=audio_input,
            transcription_input=transcription_input,
            remove_silent_input=remove_silent_input,
            remove_silent_output=remove_silent_output,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            speed=speed,
            request=request,
        )
        return return_type(file_response, response_format, r)
    
    tts_load_model()
    
    @app.on_event("startup")
    async def startup_event():
        app.state.background_tts_asr = asyncio.create_task(tts_asr())
        app.state.background_tts_step = asyncio.create_task(tts_step())

    @app.on_event("shutdown")
    async def shutdown_event():
        app.state.background_tts_asr.cancel()
        app.state.background_tts_step.cancel()
        try:
            await app.state.background_tts_asr
        except asyncio.CancelledError:
            pass
        try:
            await app.state.background_tts_step
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    uvicorn.run(
        'dynamicbatch_ttspipeline.main:app',
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
        reload=args.reload,
    )