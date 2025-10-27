# train_olid_V1.py
# v3: Character N-Gram + Stop Words Cleaning + Column Name Cleaning

import re
import pickle
import json
from pathlib import Path
import nltk
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

# --- NLTK Stop Words Configuration ---
# We extend the default NLTK stopword list because words like 'you'
# are not stopwords by default but add noise to this specific task.
try:
    from nltk.corpus import stopwords

    STOP_WORDS = set(stopwords.words('english'))
    STOP_WORDS.update(['you', 'your', 'yours', 'yourself', 'u', 'ur'])
except LookupError:
    print("NLTK stopwords list not found. Downloading...")
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    STOP_WORDS = set(stopwords.words('english'))
    STOP_WORDS.update(['you', 'your', 'yours', 'yourself', 'u', 'ur'])
# --- End Stop Words ---

# --- Global Paths & Configuration ---
DATA_PATH = Path("data/labeled_data.csv")  # Path to the source dataset
ART_DIR = Path("artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ART_DIR / "model.pkl"
METRICS_PATH = ART_DIR / "metrics.json"
CM_PNG = ART_DIR / "confusion_matrix.png"

RANDOM_STATE = 42

# Regex pattern to strip most Unicode emoji characters
EMOJI_PATTERN = re.compile(
    "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF" "\U0001F1E0-\U0001F1FF"
    "\u2600-\u26FF" "\u2700-\u27BF" "]+", flags=re.UNICODE
)


def evaluate_probs(proba, y_true, threshold=0.50, title="Validation", save_cm_path=None):
    """
    Calculates metrics and optionally saves a confusion matrix plot.
    This function is designed to save the artifact plot headless (without showing it).
    """
    y_pred = (proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision, recall, _ = precision_recall_curve(y_true, proba)
    pr_auc = auc(recall, precision)
    try:
        roc_auc = roc_auc_score(y_true, proba)
    except Exception:
        roc_auc = float("nan")

    # Calculate the raw confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Plot and save the confusion matrix artifact if a path is provided
    if save_cm_path is not None:
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title(f'Confusion Matrix (Artifact) — {title} @t={threshold:.2f}')
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
        fig.savefig(save_cm_path, dpi=140);
        # Close the figure handle to free up memory
        plt.close(fig)

    print(f"\n== {title} @ threshold={threshold:.2f} ==")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1(weighted): {f1w:.4f}")
    print(f"F1(macro): {f1m:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4, zero_division=0))

    # Return metrics and the raw CM data for separate plotting
    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1_weighted": f1w, "f1_macro": f1m,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "threshold": float(threshold),
        "support": {"class_0": int((y_true == 0).sum()), "class_1": int((y_true == 1).sum())}
    }, cm


def clean_tweet(text: str,
                remove_punct: bool = False,  # Optional: True -> remove all punctuation
                collapse_punct: bool = True,  # Optional: True -> !!! -> !, ??? -> ?
                keep_apostrophe: bool = True  # Optional: If remove_punct=True, keep apostrophes?
                ) -> str:
    """
    Cleans a single text string by removing URLs, mentions, emojis,
    and normalizing punctuation.
    """
    if not isinstance(text, str):
        return ""
    t = text.lower()

    # Remove URLs, mentions, hashtag symbols, emojis, and digits
    t = re.sub(r"http\S+|www\S+|https\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#", "", t)
    t = EMOJI_PATTERN.sub("", t)
    t = re.sub(r"\d+", "", t)

    # Clean up "RT" (Retweet) artifacts
    t = re.sub(r"^\s*rt\b:?\s*", "", t)  # At the beginning of the string
    t = re.sub(r"\brt\b:?", "", t)  # As a standalone token in the body

    # Normalize repeated punctuation (e.g., !!! -> !, ..... -> .)
    if collapse_punct:
        t = re.sub(r"([!?.,])\1+", r"\1", t)  # Collapse repeats
        t = re.sub(r"^[!?.,]+", "", t)  # Trim from start
        t = re.sub(r"[!?.,]+$", "", t)  # Trim from end

    # Optional: Remove all punctuation
    if remove_punct:
        keep = "'" if keep_apostrophe else ""
        # Remove any character that is not a word, whitespace, or the kept character
        t = re.sub(fr"[^\w\s{keep}]", " ", t)

    # Consolidate multiple spaces into one and strip leading/trailing whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    """
    Main training and evaluation pipeline.
    """
    assert DATA_PATH.exists(), f"Dataset not found at {DATA_PATH}."
    df = pd.read_csv(DATA_PATH)

    # --- NEW FIX: Clean hidden whitespace from column names ---
    df.columns = [col.strip().lower() for col in df.columns]
    # --- END FIX ---

    # --- Normalize columns for Davidson/Kaggle dataset variants ---
    if "class" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"class": "label"})
    if "tweet" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"tweet": "text"})

    # Compatibility for OLID dataset structure
    if "subtask_a" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"subtask_a": "label"})

    # Basic checks
    assert {"text", "label"}.issubset(
        set(df.columns)), f"CSV must contain 'text' and 'label'. Columns found: {list(df.columns)}"

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)

    # --- Label Normalization (for all dataset types) ---
    if df["label"].dtype == object:
        # Handle string labels
        _map = {
            "off": 1, "offensive": 1,
            "not": 0, "neither": 0,
            "hate_speech": 1, "hate": 1,
            "offensive_language": 1,
            "clean": 0, "normal": 0
        }
        df["label"] = df["label"].str.lower().map(_map)
    else:
        # Handle 3-class numeric labels (e.g., Davidson dataset)
        vals = set(pd.unique(df["label"].dropna()))
        if vals.issuperset({0, 1}) and 2 in vals:
            # Map 0(hate), 1(offensive) -> 1 (Foul); 2(neither) -> 0 (Proper)
            df["label"] = df["label"].astype(int).map({0: 1, 1: 1, 2: 0})

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print("Cleaning text data...")
    # Apply the cleaning function to the text column
    df["clean_text"] = df["text"].apply(clean_tweet)
    # ---------------------------------------

    # --- Train/Validation Split ---
    X_train, X_val, y_train, y_val = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE
    )

    # --- Simple POSITIVE oversampling ---
    pos_mask = (y_train == 1)
    if pos_mask.sum() == 0:
        print("Warning: No positive (foul) labels found in training data. Skipping oversampling.")
        X_train_os, y_train_os = X_train, y_train
    else:
        X_train_os = pd.concat([X_train, X_train[pos_mask]], ignore_index=True)
        y_train_os = pd.concat([y_train, y_train[pos_mask]], ignore_index=True)

    # === Model Definitions (Using Character N-Grams) ===

    # Shared TF-IDF vectorizer (character n-grams)
    tfidf_vec = TfidfVectorizer(
        analyzer='char_wb',  # Analyze word boundaries
        ngram_range=(2, 5),  # Use n-grams from 2 to 5 chars
        strip_accents="unicode",
        lowercase=True,
        min_df=3,
        max_df=0.95,
        stop_words=None,  # Stop words are not used for char n-grams
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

    # 3) Multinomial Naive Bayes (Requires CountVectorizer)
    bow_vec = CountVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 5),
        strip_accents="unicode",
        lowercase=True,
        min_df=3,
        max_df=0.95,
        stop_words=None,
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

    # --- Training & Evaluation Loop ---
    comparisons = {}
    best_choice = None
    DEFAULT_THRESHOLD = 0.50

    for name, model in MODELS.items():
        print(f"\n[Training] {name}")
        model.fit(X_train_os, y_train_os)

        # Get probabilities
        if hasattr(model, "predict_proba"):
            proba_val = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            proba_val = model.decision_function(X_val)
        else:
            proba_val = model.predict(X_val).astype(float)

        # 1. Evaluate metrics and save the artifact image
        m_05, cm = evaluate_probs(
            proba_val, y_val,
            threshold=DEFAULT_THRESHOLD,
            title=f"Validation [{name}]",
            save_cm_path=str(CM_PNG)  # Saves the plot to file
        )

        # 2. Display the confusion matrix plot on screen
        # We use the 'cm' data returned from the function
        print(f"Displaying plot for {name}...")
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['proper(0)', 'foul(1)']
        )
        disp.plot(cmap='Blues', values_format='d', colorbar=False)
        plt.title(f'Confusion Matrix — Validation [{name}] @t={DEFAULT_THRESHOLD:.2f}')
        plt.tight_layout()

        # plt.show() will open a new window for each plot.
        # block=False keeps the script from halting.
        # plt.pause() allows the window to render before the loop continues.
        plt.show(block=False)
        plt.pause(0.001)  # 1 millisecond pause to render

        # store metrics for this model
        comparisons[name] = {"@0.50": m_05}

        # Selection policy: Find the best model based on F1, then Precision, then ROC-AUC
        key = (m_05["f1_weighted"], m_05["precision"], m_05["roc_auc"])
        if (best_choice is None) or (key > best_choice["key"]):
            best_choice = {"name": name, "model": model, "threshold": DEFAULT_THRESHOLD, "key": key}

    best_name = best_choice["name"]
    best_model = best_choice["model"]
    best_threshold = best_choice["threshold"]
    print(f"\n== Selected best model: {best_name} @ t={best_threshold:.2f} ==")

    # --- Save Final Artifacts ---
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        # We need a custom handler to serialize numpy types
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        json.dump({"comparison": comparisons, "selected": {"model": best_name, "threshold": float(best_threshold)}}, f,
                  indent=2, cls=NpEncoder)
    print(f"Saved metrics to {METRICS_PATH}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": best_model, "threshold": float(best_threshold)}, f)
    print(f"Saved model artifact to {MODEL_PATH}")

    print("\n--- Training pipeline finished. ---")
    print(f"The best confusion matrix was saved to {CM_PNG}")
    print("Close plot windows to exit.")

    # Keep all plot windows open until the user closes them
    plt.show()


if __name__ == "__main__":
    main()
