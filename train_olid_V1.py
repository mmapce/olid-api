# train_olid_V1.py
# OLID verisi (data/olid.csv) ile TF-IDF(uni+bi) + LogisticRegression eğitir,
# recall>=0.80 için eşiği ayarlar, metrikleri ve modeli artifacts/ altına kaydeder.

import os, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd



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


DATA_PATH = Path("data/labeled_data.csv")
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
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'Confusion Matrix — {title} @t={threshold:.2f}')
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ['proper(0)','foul(1)'], rotation=45)
    plt.yticks(ticks, ['proper(0)','foul(1)'])
    thresh = cm.max()/2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha='center',
                     color="white" if cm[i,j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label'); plt.tight_layout()
    fig.savefig(CM_PNG, dpi=140); plt.close(fig)

    print(f"\n== {title} @ threshold={threshold:.2f} ==")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1(weighted): {f1w:.4f}")
    print(f"F1(macro): {f1m:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4, zero_division=0))

    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1_weighted": f1w, "f1_macro": f1m,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "threshold": threshold,
        "support": {"class_0": int((y_true==0).sum()), "class_1": int((y_true==1).sum())}
    }, cm

def main():
    assert DATA_PATH.exists(), f"Dataset not found at {DATA_PATH}. Expected columns: text,label"
    df = pd.read_csv(DATA_PATH)
    # --- Normalize columns for Davidson/Kaggle dataset variants ---
    # Rename common column names to our expected schema
    if "class" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"class": "label"})
    if "tweet" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"tweet": "text"})

    # Collapse 3-class schema to binary: hate/offensive -> 1, neither -> 0
    if "label" in df.columns:
        # String labels (e.g., 'hate_speech','offensive_language','neither')
        if df["label"].dtype == object:
            _map = {
                "hate_speech": 1, "hate": 1,
                "offensive_language": 1, "offensive": 1,
                "neither": 0, "clean": 0, "normal": 0
            }
            df["label"] = df["label"].str.lower().map(_map)
        # Numeric 0/1/2 -> 1/1/0
        vals = set(pd.unique(df["label"].dropna()))
        if vals.issuperset({0,1}) and 2 in vals:
            df["label"] = df["label"].astype(int).map({0:1, 1:1, 2:0})
    # Basic checks
    assert {"text","label"}.issubset(set(df.columns)), f"CSV must contain 'text' and 'label'. Columns: {list(df.columns)}"
    df = df.dropna(subset=["text","label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    import re
    # Regex pattern to strip most Unicode emoji characters (no external deps)
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\u2600-\u26FF"          # misc symbols
        "\u2700-\u27BF"          # dingbats
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
        text = re.sub(r"[^\w\s]", " ", text)  # Noktalama/özel karakter
        text = re.sub(r"\s+", " ", text).strip()  # Fazla boşluk
        return text

    df["clean_text"] = df["text"].apply(clean_tweet)
    # ---------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE
    )


    # --- Simple POSITIVE oversampling to lift recall at t=0.50 ---
    pos_mask = (y_train == 1)
    X_train_os = pd.concat([X_train, X_train[pos_mask]], ignore_index=True)  # +1x positives
    y_train_os = pd.concat([y_train, y_train[pos_mask]], ignore_index=True)


    # === Three models: LR (TFIDF), SVM (calibrated, TFIDF), NB (CountVec) ===
    # Shared TF-IDF vectorizer for LR/SVM
    tfidf_vec = TfidfVectorizer(
        ngram_range=(1,2),
        strip_accents="unicode",
        lowercase=True,
        min_df=1,
        max_df=0.99,
        stop_words="english",
        sublinear_tf=True,
    )

    # 1) Logistic Regression (stronger positive weight)
    pipe_lr = Pipeline([
        ("tfidf", tfidf_vec),
        ("logreg", LogisticRegression(max_iter=2000, solver="liblinear", class_weight={0:1.0, 1:2.0}))
    ])

    # 2) Linear SVM with probability via calibration (stronger positive weight)
    svm_base = LinearSVC(class_weight={0:1.0, 1:2.0})
    pipe_svm = Pipeline([
        ("tfidf", tfidf_vec),
        ("svm", CalibratedClassifierCV(svm_base, method="sigmoid", cv=3))
    ])

    # 3) Multinomial Naive Bayes (uses CountVectorizer)
    bow_vec = CountVectorizer(
        ngram_range=(1,2),
        strip_accents="unicode",
        lowercase=True,
        min_df=1,
        max_df=0.99,
        stop_words="english",
    )
    pipe_nb = Pipeline([
        ("bow", bow_vec),
        ("nb", MultinomialNB(alpha=0.5))
    ])




    MODELS = {
        "LR_TFIDF": pipe_lr,
        "SVM_TFIDF": pipe_svm,
        "NB_BOW": pipe_nb,
    }


    comparisons = {}
    best_choice = None

    for name, model in MODELS.items():
        model.fit(X_train_os, y_train_os)
        X_val_input = X_val

        if hasattr(model, "predict_proba"):
            proba_val = model.predict_proba(X_val_input)[:, 1]
        else:
            proba_val = model.decision_function(X_val_input)

        # Eval @ 0.50 only (no tuning)
        m_05, _ = evaluate(proba_val, y_val, threshold=0.50, title=f"Validation [{name}]")
        comparisons[name] = {"@0.50": m_05}

        # select best by F1(weighted), then Precision, then ROC-AUC
        key = (m_05["f1_weighted"], m_05["precision"], m_05["roc_auc"])
        if (best_choice is None) or (key > best_choice["key"]):
            best_choice = {"name": name, "model": model, "threshold": 0.50, "key": key}

    # Final best selection (no tuning)
    best_name = best_choice["name"]
    best_model = best_choice["model"]
    best_threshold = best_choice["threshold"]

    print(f"\n== Selected best model: {best_name} @ t={best_threshold:.2f} ==")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"comparison": comparisons, "selected": {"model": best_name, "threshold": float(best_threshold)}}, f, indent=2)
    print(f"\n✅ Saved metrics to {METRICS_PATH}")

    # Save artifact (selected pipeline + chosen threshold)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": best_model, "threshold": float(best_threshold)}, f)
    print(f"✅ Saved model artifact to {MODEL_PATH}")
    print(f"🖼️ Confusion matrix image saved to {CM_PNG}")

if __name__ == "__main__":
    main()