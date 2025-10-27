# train_olid_V1.py
# v3: Karakter N-Gram + Stop Words Temizliği + Sütun Adı Temizleme

import os, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import re

# --- NLTK Stop Words ---
# "love you" -> "love" dönüşümü için
import nltk

try:
    from nltk.corpus import stopwords

    # NLTK listesini genişletiyoruz, çünkü "you" kelimesi normalde stopword değil
    STOP_WORDS = set(stopwords.words('english'))
    # 'you' ve varyantları modelin kafasını en çok karıştıran kelimeler
    STOP_WORDS.update(['you', 'your', 'yours', 'yourself', 'u', 'ur'])
except LookupError:
    print("NLTK stopwords listesi bulunamadı. İndiriliyor...")
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    STOP_WORDS = set(stopwords.words('english'))
    STOP_WORDS.update(['you', 'your', 'yours', 'yourself', 'u', 'ur'])
# --- Bitiş ---

# Matplotlib sadece dosyaya kaydedecek
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    classification_report
)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

DATA_PATH = Path("data/labeled_data.csv")  # Orijinal dosya yolunuzu koruyoruz
ART_DIR = Path("artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ART_DIR / "model.pkl"
METRICS_PATH = ART_DIR / "metrics.json"
CM_PNG = ART_DIR / "confusion_matrix.png"

RANDOM_STATE = 42


def evaluate(proba, y_true, threshold=0.50, title="Validation"):
    y_pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Curves
    precision, recall, _ = precision_recall_curve(y_true, proba)
    pr_auc = auc(recall, precision)
    try:
        roc_auc = roc_auc_score(y_true, proba)
    except Exception:
        roc_auc = float("nan")

    # Confusion matrix image
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'Confusion Matrix — {title} @t={threshold:.2f}')
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ['proper(0)', 'foul(1)'], rotation=45)
    plt.yticks(ticks, ['proper(0)', 'foul(1)'])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center',
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label');
    plt.xlabel('Predicted label');
    plt.tight_layout()
    fig.savefig(CM_PNG, dpi=140);
    plt.close(fig)

    print(f"\n== {title} @ threshold={threshold:.2f} ==")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1(weighted): {f1w:.4f}")
    print(f"F1(macro): {f1m:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4, zero_division=0))

    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1_weighted": f1w, "f1_macro": f1m,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "threshold": threshold,
        "support": {"class_0": int((y_true == 0).sum()), "class_1": int((y_true == 1).sum())}
    }, cm


# Regex pattern to strip most Unicode emoji characters
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\u2600-\u26FF"  # misc symbols
    "\u2700-\u27BF"  # dingbats
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

    # Noktalama işaretlerini kaldırmıyoruz (char_wb analizörü bunları işler)
    text = re.sub(r"\s+", " ", text).strip()  # Fazla boşluk

    # --- Stop Words'leri Manuel Kaldırma ('love you' sorunu için) ---
    try:
        words = text.split()
        filtered_words = [word for word in words if word not in STOP_WORDS]
        text = " ".join(filtered_words)
    except NameError:
        pass
        # --- Bitiş ---

    return text


def main():
    assert DATA_PATH.exists(), f"Dataset not found at {DATA_PATH}."
    df = pd.read_csv(DATA_PATH)

    # --- YENİ DÜZELTME: Sütun adlarındaki gizli boşlukları temizle ('AssertionError' için) ---
    df.columns = [col.strip().lower() for col in df.columns]  # .lower() ekleyerek küçük harf garantisi
    # --- DÜZELTME SONU ---

    # --- Normalize columns for Davidson/Kaggle dataset variants ---
    if "class" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"class": "label"})
    if "tweet" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"tweet": "text"})

    # (Bir önceki çözümden gelen) OLID veri seti uyumluluğu
    if "subtask_a" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"subtask_a": "label"})

    # Basic checks (Artık bu 'assert' hata vermemeli)
    assert {"text", "label"}.issubset(
        set(df.columns)), f"CSV must contain 'text' and 'label'. Columns found: {list(df.columns)}"

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)

    # --- Etiket (Label) Normalleştirme (Tüm veri setleri için) ---
    if df["label"].dtype == object:
        _map = {
            "off": 1, "offensive": 1,
            "not": 0, "neither": 0,
            "hate_speech": 1, "hate": 1,
            "offensive_language": 1,
            "clean": 0, "normal": 0
        }
        df["label"] = df["label"].str.lower().map(_map)
    else:
        vals = set(pd.unique(df["label"].dropna()))
        if vals.issuperset({0, 1}) and 2 in vals:
            # 0(hate), 1(offensive) -> 1 (Foul); 2(neither) -> 0 (Proper)
            df["label"] = df["label"].astype(int).map({0: 1, 1: 1, 2: 0})

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print("Cleaning text data...")
    df["clean_text"] = df["text"].apply(clean_tweet)
    # ---------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE
    )

    # --- Simple POSITIVE oversampling ---
    pos_mask = (y_train == 1)
    if pos_mask.sum() == 0:
        print("Uyarı: Eğitim verisinde hiç pozitif (foul) etiket bulunamadı. Oversampling atlanıyor.")
        X_train_os, y_train_os = X_train, y_train
    else:
        X_train_os = pd.concat([X_train, X_train[pos_mask]], ignore_index=True)
        y_train_os = pd.concat([y_train, y_train[pos_mask]], ignore_index=True)

    # === Modeller (Karakter N-Gram) ===
    # Shared TF-IDF vectorizer
    tfidf_vec = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 5),
        strip_accents="unicode",
        lowercase=True,
        min_df=3,
        max_df=0.95,
        stop_words=None,  # clean_tweet içinde manuel halledildi
        sublinear_tf=True,
    )

    # 1) Logistic Regression
    pipe_lr = Pipeline([
        ("tfidf", tfidf_vec),
        ("logreg", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced"))
    ])

    # 2) Linear SVM
    svm_base = LinearSVC(class_weight="balanced", max_iter=2000, dual=True)
    pipe_svm = Pipeline([
        ("tfidf", tfidf_vec),
        ("svm", CalibratedClassifierCV(svm_base, method="sigmoid", cv=3))
    ])

    # 3) Multinomial Naive Bayes
    bow_vec = CountVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 5),
        strip_accents="unicode",
        lowercase=True,
        min_df=3,
        max_df=0.95,
        stop_words=None,  # clean_tweet içinde manuel halledildi
    )
    pipe_nb = Pipeline([
        ("bow", bow_vec),
        ("nb", MultinomialNB(alpha=0.5))
    ])

    MODELS = {
        "LR_TFIDF_char_stop": pipe_lr,
        "SVM_TFIDF_char_stop": pipe_svm,
        "NB_BOW_char_stop": pipe_nb,
    }

    comparisons = {}
    best_choice = None

    for name, model in MODELS.items():
        print(f"\nTraining model: {name}...")
        model.fit(X_train_os, y_train_os)
        proba_val = model.predict_proba(X_val)[:, 1]

        m_05, _ = evaluate(proba_val, y_val, threshold=0.50, title=f"Validation [{name}]")
        comparisons[name] = {"@0.50": m_05}

        # En iyiyi F1(weighted)'a göre seç
        key = (m_05["f1_weighted"], m_05["precision"], m_05["roc_auc"])
        if (best_choice is None) or (key > best_choice["key"]):
            best_choice = {"name": name, "model": model, "threshold": 0.50, "key": key}

    # Final best selection
    best_name = best_choice["name"]
    best_model = best_choice["model"]
    best_threshold = best_choice["threshold"]

    print(f"\n== Evaluating best model [{best_name}] for final artifacts ==")
    best_proba = best_model.predict_proba(X_val)[:, 1]
    final_metrics, _ = evaluate(best_proba, y_val, threshold=best_threshold, title=f"Final Validation [{best_name}]")

    print(f"\n== Selected best model: {best_name} @ t={best_threshold:.2f} ==")

    # Metrikleri kaydet
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"comparison": comparisons, "selected": {"model": best_name, "threshold": float(best_threshold),
                                                           "final_metrics": final_metrics}}, f, indent=2)
    print(f"\n✅ Saved metrics to {METRICS_PATH}")

    # Modeli kaydet
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": best_model, "threshold": float(best_threshold)}, f)
    print(f"✅ Saved model artifact to {MODEL_PATH}")
    print(f"🖼️ Confusion matrix image saved to {CM_PNG}")


if __name__ == "__main__":
    main()