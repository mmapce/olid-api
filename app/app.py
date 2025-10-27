# app/app.py
# ———————————————————————————————————————————————————————————
# FASTAPI: Tweet Foul-Language Detector (threshold override destekli)
# ———————————————————————————————————————————————————————————

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle, os, re
from typing import List, Optional
from fastapi.staticfiles import StaticFiles
from pathlib import Path

APP_TITLE = "Tweet Foul-Language Detector"
APP_VERSION = "1.2"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# UI mount (../static)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Model path (../artifacts/model.pkl)
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"

_pipeline = None
_threshold = None  # model artefact'tan okunan varsayılan eşik

def _load_model():
    """artifacts/model.pkl içindeki {'pipeline','threshold'} sözlüğünü yükler"""
    global _pipeline, _threshold
    if not os.path.exists(MODEL_PATH):
        _pipeline, _threshold = None, None
        return False
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    _pipeline = obj["pipeline"]
    # threshold key'i yoksa 0.50'ye düş
    _threshold = float(obj.get("threshold", 0.50))
    return True

_load_model()

# ——— Eğitimdeki temizleme ile aynı davranış ———
EMOJI_PATTERN = re.compile(
    "["                    # train_olid_V1.py ile uyumlu emoji aralığı
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE
)

def clean_tweet(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URL
    text = re.sub(r"@\w+", "", text)                    # Mention
    text = re.sub(r"#", "", text)                       # Hashtag sembolü
    text = EMOJI_PATTERN.sub("", text)                  # Emoji kaldır
    text = re.sub(r"\d+", "", text)                     # Sayılar
    # Not: train_olid_V1.py'de noktalama kaldırılmıyor; burada da kaldırmıyoruz.
    text = re.sub(r"\s+", " ", text).strip()            # Fazla boşluk
    return text

# ——— Şemalar ———
class TextIn(BaseModel):
    text: str = Field(..., min_length=1, description="Sınıflandırılacak metin")
    # UI'dan isteğe bağlı threshold override
    threshold: Optional[float] = Field(None, description="Opsiyonel karar eşiği (0..1)")

class PredictOut(BaseModel):
    prediction: int
    prob_foul: float
    threshold: float  # kullanılmış eşik (override edilmişse o)

class BatchIn(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100,
                             description="Sınıflandırılacak metinler listesi (1-100 arası)")
    threshold: Optional[float] = Field(None, description="Opsiyonel karar eşiği (0..1)")

class PredItem(BaseModel):
    text: str
    prediction: int
    prob_foul: float

class BatchOut(BaseModel):
    results: List[PredItem]
    threshold: float  # kullanılmış eşik (override edilmişse o)

def ensure_model():
    if _pipeline is None or _threshold is None:
        raise HTTPException(status_code=503, detail="Model yüklü değil. artifacts/model.pkl bulunamadı.")

# ——— Endpoints ———
@app.get("/", summary="Sağlık kontrolü / sürüm")
def root():
    return {
        "ok": True,
        "about": "foul(1) vs proper(0)",
        "model_loaded": _pipeline is not None,
        "threshold": float(_threshold) if _threshold is not None else None,
        "version": APP_VERSION,
    }

@app.post("/predict", response_model=PredictOut, summary="Tek metin tahmini")
def predict(inp: TextIn):
    ensure_model()
    # İstek eşik verdiyse kullan, yoksa modelin varsayılan eşiği
    thr = float(inp.threshold) if inp.threshold is not None else float(_threshold)
    proba = float(_pipeline.predict_proba([clean_tweet(inp.text)])[:, 1][0])
    pred = int(proba >= thr)
    return PredictOut(prediction=pred, prob_foul=proba, threshold=thr)

@app.post("/predict/batch", response_model=BatchOut, summary="Toplu metin tahmini (1-100)")
def predict_batch(inp: BatchIn):
    ensure_model()
    thr = float(inp.threshold) if inp.threshold is not None else float(_threshold)
    cleaned_texts = [clean_tweet(t) for t in inp.texts]
    probs = _pipeline.predict_proba(cleaned_texts)[:, 1].tolist()
    preds = [int(p >= thr) for p in probs]
    results = [
        PredItem(text=inp.texts[i], prediction=preds[i], prob_foul=float(probs[i]))
        for i in range(len(inp.texts))
    ]
    return BatchOut(results=results, threshold=thr)

@app.post("/admin/reload", summary="(Admin) modeli yeniden yükle")
def admin_reload():
    ok = _load_model()
    return {"reloaded": ok, "model_loaded": _pipeline is not None, "threshold": _threshold}

@app.get("/debug/paths")
def debug_paths():
    return {
        "cwd": os.getcwd(),
        "base_dir": str(BASE_DIR),
        "expected_model_path": str(MODEL_PATH),
        "exists": MODEL_PATH.exists()
    }