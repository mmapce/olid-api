# app/app.py
# (TR) Tweet Foul-Language API
# - model.pkl: {'pipeline': <sklearn Pipeline>, 'threshold': float}
# - /           : sağlık kontrolü
# - /predict    : tek metin tahmini
# - /predict/batch : toplu metin tahmini (1-100)
# Not: artifacts/model.pkl yoksa 503 döner (açık hata mesajı).

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle, os
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

APP_TITLE = "Tweet Foul-Language Detector"
APP_VERSION = "1.2"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

# (Opsiyonel) CORS: gerekirse origin kısıtlayabilirsin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prod'da kendi domainini yaz
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pathlib import Path

# Proje kökü = bu dosyanın (app/app.py) iki üstü
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"

_pipeline = None
_threshold = None

def _load_model():
    """Pickle'dan modeli yükle. Dosya yoksa None bırak."""
    global _pipeline, _threshold
    if not os.path.exists(MODEL_PATH):
        _pipeline, _threshold = None, None
        return False
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    _pipeline = obj["pipeline"]
    _threshold = float(obj["threshold"])
    return True

# Uygulama başlarken bir kez dene
_load_model()

# ----- Şemalar -----
class TextIn(BaseModel):
    text: str = Field(..., min_length=1, description="Sınıflandırılacak metin")

class PredictOut(BaseModel):
    prediction: int   # 1=foul, 0=proper
    prob_foul: float  # [0,1]
    threshold: float

class BatchIn(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100, description="Sınıflandırılacak metinler listesi (1-100 arası)")


class PredItem(BaseModel):
    text: str
    prediction: int
    prob_foul: float

class BatchOut(BaseModel):
    results: List[PredItem]
    threshold: float

# ----- Yardımcı -----
def ensure_model():
    if _pipeline is None or _threshold is None:
        raise HTTPException(status_code=503, detail="Model yüklü değil. artifacts/model.pkl bulunamadı.")

# ----- Endpoint'ler -----
@app.get("/", summary="Sağlık kontrolü / sürüm")
def root():
    return {
        "ok": True,
        "about": "foul(1) vs proper(0)",
        "model_loaded": _pipeline is not None,
        "threshold": _threshold,
        "version": APP_VERSION,
    }

@app.post("/predict", response_model=PredictOut, summary="Tek metin tahmini")
def predict(inp: TextIn):
    ensure_model()
    proba = float(_pipeline.predict_proba([inp.text])[:, 1][0])
    pred = int(proba >= _threshold)
    return PredictOut(prediction=pred, prob_foul=proba, threshold=_threshold)

@app.post("/predict/batch", response_model=BatchOut, summary="Toplu metin tahmini (1-100)")
def predict_batch(inp: BatchIn):
    ensure_model()
    probs = _pipeline.predict_proba(inp.texts)[:, 1].tolist()
    preds = [int(p >= _threshold) for p in probs]
    results = [
        PredItem(text=t, prediction=preds[i], prob_foul=float(probs[i]))
        for i, t in enumerate(inp.texts)
    ]
    return BatchOut(results=results, threshold=float(_threshold))

@app.post("/admin/reload", summary="(Admin) modeli yeniden yükle")
def admin_reload():
    ok =_load_model()
    return {"reloaded": ok, "model_loaded": _pipeline is not None, "threshold": _threshold}

import os

@app.get("/debug/paths")
def debug_paths():
    return {
        "cwd": os.getcwd(),
        "base_dir": str(BASE_DIR),
        "expected_model_path": str(MODEL_PATH),
        "exists": MODEL_PATH.exists()
    }