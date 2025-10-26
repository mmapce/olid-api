# app/app.py
# LÜTFEN DOSYANIZIN TAMAMINI BU İÇERİKLE DEĞİŞTİRİN

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle, os
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path

# --- EKLENMESİ GEREKEN KÜTÜPHANE ---
import re
# --- SON ---

APP_TITLE = "Tweet Foul-Language Detector"
APP_VERSION = "1.2"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"

_pipeline = None
_threshold = None

def _load_model():
    global _pipeline, _threshold
    if not os.path.exists(MODEL_PATH):
        _pipeline, _threshold = None, None
        return False
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    _pipeline = obj["pipeline"]
    _threshold = float(obj["threshold"])
    return True

_load_model()


# --- GÜNCELLEME 1: 'train_olid_V1.py' dosyasından clean_tweet fonksiyonunu kopyalayın ---
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE
)

def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URL
    text = re.sub(r"@\w+", "", text)  # Mention
    text = re.sub(r"#", "", text)  # Hashtag sembolü
    text = EMOJI_PATTERN.sub("", text)  # Emoji kaldır
    text = re.sub(r"\d+", "", text)  # Sayılar
    # Noktalama işaretlerini kaldırma (r"[^\w\s]") adımı
    # char_wb kullandığımız için eğitim script'inde kaldırılmıştı.
    # Burada da OLMAMASI gerekiyor.
    text = re.sub(r"\s+", " ", text).strip()  # Fazla boşluk
    return text
# --- GÜNCELLEME 1 SONU ---


# ----- Şemalar (Değişiklik yok) -----
class TextIn(BaseModel):
    text: str = Field(..., min_length=1, description="Sınıflandırılacak metin")

class PredictOut(BaseModel):
    prediction: int
    prob_foul: float
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

    # --- GÜNCELLEME 2: Tekli tahmin için temizleme ---
    clean_text = clean_tweet(inp.text)
    proba = float(_pipeline.predict_proba([clean_text])[:, 1][0])
    # --- GÜNCELLEME 2 SONU ---

    pred = int(proba >= _threshold)
    return PredictOut(prediction=pred, prob_foul=proba, threshold=_threshold)

@app.post("/predict/batch", response_model=BatchOut, summary="Toplu metin tahmini (1-100)")
def predict_batch(inp: BatchIn):
    ensure_model()

    # --- GÜNCELLEME 3: Toplu tahmin için temizleme ---
    cleaned_texts = [clean_tweet(t) for t in inp.texts]
    probs = _pipeline.predict_proba(cleaned_texts)[:, 1].tolist()
    # --- GÜNCELLEME 3 SONU ---

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

@app.get("/debug/paths")
def debug_paths():
    return {
        "cwd": os.getcwd(),
        "base_dir": str(BASE_DIR),
        "expected_model_path": str(MODEL_PATH),
        "exists": MODEL_PATH.exists()
    }