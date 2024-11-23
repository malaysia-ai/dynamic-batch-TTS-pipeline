from fastapi import FastAPI, Request
from fastapi import File, Form, UploadFile
from fastapi import HTTPException
from dynamicbatch_ttspipeline.env import args
from transformers_openai.middleware import InsertMiddleware
from dynamicbatch_ttspipeline.speech_enhancement import (
    load_model as speech_enhancement_load_model, 
    predict as speech_enhancement_predict,
    step as speech_enhancement_step,
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
    ):
        """
        Speech enhancement for audio file. Will return base64 WAV format.
        """
        r = await speech_enhancement_predict(file)
        return r

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


if __name__ == "__main__":
    uvicorn.run(
        'dynamicbatch_ttspipeline.main:app',
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
        reload=args.reload,
    )