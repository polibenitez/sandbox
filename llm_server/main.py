"""
OpenAI-compatible embeddings server using SentenceTransformers.
- Exposes /v1/embeddings (OpenAI-style) and /embed (simple) endpoints
- Auto GPU/CPU fallback with clear logging
- Optional API key check via ENV OPENAI_API_KEY
- CORS enabled for public access (configure origins as needed)

Run:
    pip install fastapi uvicorn sentence-transformers[all] numpy pydantic
    UVICORN_CMD: uvicorn openai_compatible_embeddings_server:app --host 0.0.0.0 --port 8000

Security: use a firewall, reverse proxy with TLS, or ngrok for testing.
"""
from typing import List, Union, Optional
import os
import logging
import traceback
import torch

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
from sentence_transformers import SentenceTransformer

# --------- Configuration ---------
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-en")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
API_KEY = os.environ.get("OPENAI_API_KEY")  # optional: if set, requests must include Authorization: Bearer <key>
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "64"))
ALLOW_ORIGINS = os.environ.get("ALLOW_ORIGINS", "*")  # comma-separated or '*' for all

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emb-server")

app = FastAPI(title="OpenAI-compatible Embeddings Server")

# Simple CORS config (adjust origins in production)
if ALLOW_ORIGINS == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------- Request / Response models ---------
class OpenAIEmbRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None

class SimpleEmbRequest(BaseModel):
    texts: List[str]


# --------- Model loading with fallback ---------
def load_model(model_name: str):
    """Attempt to load model on GPU, fallback to CPU if any CUDA error occurs."""
    # prefer CUDA if available
    device = "cuda" if (hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available()) else "cpu"
    logger.info(f"Trying to load model {model_name} on device={device}")
    try:
        model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"torch_dtype": torch.float16}
        )

        logger.info(f"Loaded model on {device}")
        return model
    except Exception as e:
        # if CUDA failure occurs, try CPU
        logger.warning(f"Failed to load on {device}: {e}")
        if device == "cuda":
            try:
                logger.info("Falling back to CPU...")
                model = SentenceTransformer(model_name, device="cpu")
                logger.info("Loaded model on cpu")
                return model
            except Exception as e2:
                logger.error(f"Failed to load on CPU as well: {e2}")
                logger.error(traceback.format_exc())
                raise
        else:
            logger.error(traceback.format_exc())
            raise


MODEL = load_model(MODEL_NAME)


# --------- Utility functions ---------
def require_api_key(authorization: Optional[str]):
    if API_KEY:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid Authorization header")
        token = authorization.split(" ", 1)[1]
        if token != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")


def make_embedding_response(embeddings: np.ndarray, model_name: str = MODEL_NAME):
    """Return OpenAI compatible JSON object for embeddings.
    embeddings: numpy array of shape (N, D)
    """
    data = []
    for i, emb in enumerate(embeddings.tolist()):
        data.append({
            "object": "embedding",
            "embedding": emb,
            "index": i,
        })
    return {
        "object": "list",
        "data": data,
        "model": model_name,
        "usage": {},
    }


# --------- Endpoints ---------
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/embed")
async def embed_simple(req: SimpleEmbRequest, Authorization: Optional[str] = Header(None)):
    require_api_key(Authorization)
    texts = req.texts
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    try:
        embeddings = MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return {"embeddings": embeddings.tolist(), "model": MODEL_NAME}
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings")
async def openai_embeddings(req: OpenAIEmbRequest, request: Request, Authorization: Optional[str] = Header(None)):
    """OpenAI-compatible embeddings endpoint.
    Accepts request.input as str or list[str]. Returns data[] with embeddings.
    """
    require_api_key(Authorization)

    # accept alternate shape: sometimes OpenAI clients send {input: "..."}
    inputs = req.input
    if isinstance(inputs, str):
        inputs = [inputs]
    if not isinstance(inputs, list) or not inputs:
        raise HTTPException(status_code=400, detail="`input` must be a non-empty string or list of strings")

    try:
        # batch if needed
        embeddings_list = []
        for i in range(0, len(inputs), MAX_BATCH_SIZE):
            batch = inputs[i:i+MAX_BATCH_SIZE]
            emb = MODEL.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings_list.append(emb)
            del emb
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        embeddings = np.vstack(embeddings_list)
        return make_embedding_response(embeddings, model_name=req.model or MODEL_NAME)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# Optional: simple root to show server is running
@app.get("/")
def root():
    return {"message": "Embeddings server running", "model": MODEL_NAME}


# If run directly, start uvicorn (useful for dev)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("openai_compatible_embeddings_server:app", host=HOST, port=PORT, reload=False)
