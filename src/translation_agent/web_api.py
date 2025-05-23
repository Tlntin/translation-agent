import argparse
import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from .utils import MAX_TOKENS_PER_CHUNK
from .utils import translate


app = FastAPI()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", type=bool, default=False)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


class Data(BaseModel):
    source_lang: str
    target_lang: str
    source_text: str
    text_type: str
    identity_description: str = "You are an overseas study expert, if text contains `International Year` you need translate `International Year Zero` or `International Year 0` or `International Year` to `预科`, `International Year One` or `International Year 1` to `国际大一`, `International Year Two` or `International Year 2` to `国际大二`, other `Internation Year xx` do the same as so on."
    country: str = "China"
    model: str = os.getenv("DEFAULT_MODEL")
    chunk_model: str = os.getenv("DEFAULT_CHUNK_MODEL")
    max_tokens: int = MAX_TOKENS_PER_CHUNK

@app.get("/")
def read_root():
    now_dir = os.path.dirname(__file__)
    with open(os.path.join(now_dir, 'index.html'), encoding='utf-8') as f:
        str1 = f.read()
    return HTMLResponse(content=str1, status_code=200)



@app.post("/translate/")
def translate_api(data: Data):
    return translate(
        source_lang=data.source_lang,
        target_lang=data.target_lang,
        source_text=data.source_text,
        text_type=data.text_type,
        identity_description=data.identity_description,
        country=data.country,
        model=data.model,
        chunk_model=data.chunk_model,
        max_tokens=data.max_tokens,
    ).strip("\"")


if __name__ == "__main__":
    args = parse_args()
    log_config_path = os.path.join(os.path.dirname(__file__), "log_conf.yaml")
    uvicorn.run(
        app="__main__:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_config=log_config_path,
    )
