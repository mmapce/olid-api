# app/app.py
# ———————————————————————————————————————————————————————————
# FASTAPI: Tweet Foul-Language Detector (threshold override + model info)
# ———————————————————————————————————————————————————————————

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
import pickle, json, os, re
from typing import List, Optional

# ——— App meta ———
APP_TITLE = "Tweet Foul-Language Detector"
APP_VERSION = "1.3"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# ——— Static UI mount ———
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

# ——— CORS ———
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ——— Model paths ———
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"

_pipeline = None
_threshold = None  # model artefact'tan okunan varsayılan eşik


# ——— Model yükleme ———
def _load_model():
    """artifacts/model.pkl içindeki {'pipeline','threshold'} sözlüğünü yükler"""
    global _pipeline, _threshold
    if not os.path.exists(MODEL_PATH):
        _pipeline, _threshold = None, None
        return False
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    _pipeline = obj["pipeline"]
    _threshold = float(obj.get("threshold", 0.50))
    return True


def _load_metrics():
    """artifacts/metrics.json varsa skorları döndürür."""
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


_load_model()

# ——— Temizleme (train_olid_V1.py ile aynı) ———
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
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
    text = re.sub(r"\s+", " ", text).strip()            # Fazla boşluk
    return text


# ——— Şemalar ———
class TextIn(BaseModel):
    text: str = Field(..., min_length=1, description="Sınıflandırılacak metin")
    threshold: Optional[float] = Field(None, description="Opsiyonel karar eşiği (0..1)")


class PredictOut(BaseModel):
    prediction: int
    prob_foul: float
    threshold: float


class BatchIn(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100, description="Sınıflandırılacak metin listesi")
    threshold: Optional[float] = Field(None, description="Opsiyonel karar eşiği (0..1)")


class PredItem(BaseModel):
    text: str
    prediction: int
    prob_foul: float


class BatchOut(BaseModel):
    results: List[PredItem]
    threshold: float


# ——— Yardımcılar ———
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


@app.get("/model/info", summary="Eğitimden gelen skorlar ve seçilen model")
def model_info():
    ensure_model()
    m = _load_metrics() or {}
    selected = m.get("selected", {})
    comparison = m.get("comparison", {})
    return {
        "model_loaded": _pipeline is not None,
        "default_threshold": float(_threshold),
        "selected_model": selected.get("model"),
        "selected_threshold": selected.get("threshold", _threshold),
        "comparison": comparison,
    }


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
        "exists": MODEL_PATH.exists(),
    }