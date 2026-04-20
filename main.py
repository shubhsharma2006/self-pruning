"""
Self-Pruning NN — Production Inference API
==========================================
FastAPI server for serving trained PrunableLinear models.

Endpoints:
  GET  /health          — liveness probe (Docker/k8s)
  GET  /models          — list available checkpoints
  GET  /stats/{model}   — sparsity + accuracy for a specific model
  POST /predict         — run inference, log to DB
  GET  /logs            — recent inference log

Run:
  python main.py
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, field_validator

from database import InferenceLog, SessionLocal, TrainingResult, init_db, log_training, logger
from train import SelfPruningNet

# Config
RESULTS_DIR  = Path(os.getenv("RESULTS_DIR", "results"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "model_lambda_0_001.pt")
API_KEY_VALUE = os.getenv("API_KEY", "dev-secret-change-me")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Auth
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: Optional[str] = Security(api_key_header)):
    if key != API_KEY_VALUE:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key

# Model registry
# Holds all loaded models: {model_id: {"model": SelfPruningNet, "meta": dict}}
_registry: Dict[str, Any] = {}

def _load_model(path: Path) -> Dict:
    """Load a checkpoint and return model + metadata."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    # Read input_dim from checkpoint (stored at training time)
    input_dim = ckpt.get("input_dim", 64)
    model = SelfPruningNet(input_dim=input_dim)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return {
        "model"    : model,
        "lambda"   : ckpt.get("lambda", 0.0),
        "accuracy" : ckpt.get("accuracy", 0.0),
        "sparsity" : ckpt.get("sparsity", 0.0),
        "input_dim": input_dim,
        "path"     : str(path),
    }

def _discover_and_load():
    """Load all .pt files in RESULTS_DIR into the registry."""
    for pt in RESULTS_DIR.glob("model_lambda_*.pt"):
        model_id = pt.stem              # e.g. model_lambda_0_001
        if model_id not in _registry:
            try:
                _registry[model_id] = _load_model(pt)
                logger.info(f"Loaded model: {model_id}  (input_dim={_registry[model_id]['input_dim']})")
            except Exception as e:
                logger.error(f"Failed to load {pt.name}: {e}")

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + load all available models."""
    init_db()
    _discover_and_load()
    if not _registry:
        logger.warning("No model checkpoints found in results/. Train first with: python train.py --fast")
    yield
    # Shutdown cleanup (if needed)
    logger.info("Server shutting down.")

# App
app = FastAPI(
    title       = "Self-Pruning NN API",
    description = "Inference server for PrunableLinear neural networks",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)

# Schemas
class PredictRequest(BaseModel):
    data    : List[float]
    model_id: Optional[str] = None  # default: first available model

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        import math
        if not v:
            raise ValueError("Input data cannot be empty")
        if any(math.isnan(x) or math.isinf(x) for x in v):
            raise ValueError("Input contains NaN or Inf values")
        return v

class PredictResponse(BaseModel):
    prediction : int
    confidence : float
    model_id   : str
    latency_ms : float

class ModelStats(BaseModel):
    model_id   : str
    lambda_val : float
    accuracy   : float
    sparsity   : float
    input_dim  : int
    layer_wise : Dict

# Helper: async DB write
async def _log_inference_async(model_id: str, prediction: int, latency: float):
    """Write to DB in a thread so we don't block the async event loop."""
    def _write():
        db = SessionLocal()
        try:
            db.add(InferenceLog(model_id=model_id, prediction=prediction, latency_ms=latency))
            db.commit()
        finally:
            db.close()
    await asyncio.to_thread(_write)

# Endpoints

@app.get("/health")
async def health():
    """Liveness probe — always returns 200 if server is up."""
    return {
        "status"      : "ok",
        "models_loaded": len(_registry),
        "device"      : str(DEVICE),
    }

@app.get("/models")
async def list_models(_: str = Depends(require_api_key)):
    """List all loaded model checkpoints with their metadata."""
    return {
        mid: {
            "lambda"   : m["lambda"],
            "accuracy" : m["accuracy"],
            "sparsity" : m["sparsity"],
            "input_dim": m["input_dim"],
        }
        for mid, m in _registry.items()
    }

@app.get("/stats/{model_id}", response_model=ModelStats)
async def get_stats(model_id: str, _: str = Depends(require_api_key)):
    """Return sparsity stats for a specific loaded model."""
    if model_id not in _registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found. "
                            f"Available: {list(_registry.keys())}")
    m = _registry[model_id]
    return ModelStats(
        model_id   = model_id,
        lambda_val = m["lambda"],
        accuracy   = m["accuracy"],
        sparsity   = m["sparsity"],
        input_dim  = m["input_dim"],
        layer_wise = m["model"].compute_sparsity(),
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, _: str = Depends(require_api_key)):
    """Run inference. Logs result to DB asynchronously."""
    # Select model
    model_id = req.model_id
    if model_id is None:
        if not _registry:
            raise HTTPException(status_code=503, detail="No models loaded. Run training first.")
        model_id = next(iter(_registry))    # use first available
    if model_id not in _registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    m       = _registry[model_id]
    model   = m["model"]
    exp_dim = m["input_dim"]

    if len(req.data) != exp_dim:
        raise HTTPException(status_code=400,
            detail=f"Input dim mismatch: expected {exp_dim}, got {len(req.data)}")

    t0 = time.perf_counter()
    with torch.no_grad():
        x       = torch.tensor([req.data], dtype=torch.float32).to(DEVICE)
        logits  = model(x)
        probs   = torch.softmax(logits, dim=1)
        pred    = probs.argmax(1).item()
        conf    = probs.max().item()
    latency = (time.perf_counter() - t0) * 1000

    # Non-blocking DB log
    asyncio.create_task(_log_inference_async(model_id, pred, latency))

    return PredictResponse(
        prediction = pred,
        confidence = round(conf, 4),
        model_id   = model_id,
        latency_ms = round(latency, 3),
    )

@app.get("/logs")
async def get_logs(limit: int = 20, _: str = Depends(require_api_key)):
    """Return recent inference log entries."""
    def _read():
        db = SessionLocal()
        try:
            rows = (db.query(InferenceLog)
                      .order_by(InferenceLog.id.desc())
                      .limit(limit).all())
            return [
                {"id": r.id, "model_id": r.model_id,
                 "prediction": r.prediction, "latency_ms": round(r.latency_ms, 3),
                 "timestamp": str(r.timestamp)}
                for r in rows
            ]
        finally:
            db.close()
    return await asyncio.to_thread(_read)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
