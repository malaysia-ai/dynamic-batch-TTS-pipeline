from fastapi import FastAPI, Request
from fastapi import File, Form, UploadFile
from fastapi import HTTPException
from dynamicbatch_ttspipeline.env import args
from transformers_openai.middleware import InsertMiddleware
import uvicorn
import logging

app = FastAPI()
app.add_middleware(InsertMiddleware, max_concurrent=args.max_concurrent)

if __name__ == "__main__":
    uvicorn.run(
        'dynamicbatch_ttspipeline.main:app',
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
        reload=args.reload,
    )