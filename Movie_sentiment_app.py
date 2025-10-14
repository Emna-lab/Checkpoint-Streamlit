# Movie_sentiment_app.py
# ------------------------------------------------------------
# 🎬 IMDb Sentiment Classification — Neural Network (TF-IDF)
# A playful, step-by-step lab that trains a simple neural net
# to classify movie reviews as positive or negative.
#
# Notes about robustness (to avoid deployment errors):
# - CSV loader auto-detects location (root or data/) and encodings.
# - No hard NLTK dependency at runtime: we *try* NLTK stopwords,
#   but safely fall back to scikit-learn's built-in list.
# - TF-IDF features are converted to dense arrays for Keras so
#   validation_split works (Keras cannot split sparse matrices).
# - EarlyStopping is included and configurable from the sidebar.
# ------------------------------------------------------------

from __future__ import annotations
import os, re, html, warnings, io
from pathlib import Path

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)

# Streamlit UI
import streamlit as st

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

# ML utils
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn import metrics

# Optional NLTK (safe, with fallback)
try:
    import nltk
    _NLTK_OK = True
except Exception:
    _NLTK_OK = False

# TensorFlow / Keras (CPU build recommended in requirements)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------
# Sidebar — Training controls (kept exactly, plus EarlyStopping)
# ------------------------------------------------------------
st.set_page_config(page_title="IMDb Sentiment — TF-IDF + NN", layout="wide")

with st.sidebar:
    st.header("⚙️ Training controls")
    max_features = st.slider(
        "TF-IDF max_features (vocabulary size)",
        min_value=5_000, max_value=50_000, value=30_000, step=5_000,
        help="Higher = more expressive features (but slower & more RAM)."
    )
    ngram_max = st.select_slider(
        "TF-IDF n-grams (max)",
        options=[1, 2, 3], value=1,
        help="Unigrams only (1) or include bigrams (2)/trigrams (3)."
    )
    test_size = st.slider(
        "Test split",
        min_value=0.1, max_value=0.3, value=0.2, step=0.05
    )
    validation_split = st.slider(
        "Validation split (used by Keras during training)",
        min_value=0.05, max_value=0.3, value=0.1, step=0.05
    )
    epochs = st.slider("Epochs", 1, 100, 10)
    batch_size = st.select_slider("Batch size", options=[32, 64, 128, 256], value=64)

    st.divider()
    st.subheader("🛑 Early stopping")
    use_es = st.checkbox("Enable EarlyStopping", value=True)
    patience = st.slider("Patience (epochs without val_loss improvement)", 1, 10, 3)
    min_delta = st.number_input("min_delta (minimum improvement in val_loss)", value=0.0, step=0.001, format="%.3f")

# ------------------------------------------------------------
# Little helpers
# ------------------------------------------------------------
def try_get_stopwords() -> set[str]:
    """Try NLTK stopwords; if unavailable, fall back to scikit-learn."""
    # We want it deploy-safe; never hard-fail on NLTK.
    if _NLTK_OK:
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            try:
                nltk.download("stopwords", quiet=True)
            except Exception:
                pass
        try:
            from nltk.corpus import stopwords as _sw
            words = set(_sw.words("english"))
            return words
        except Exception:
            pass
    # Fallback (always available)
    return set(sk_text.ENGLISH_STOP_WORDS)

STOPWORDS = try_get_stopwords()

CLEAN_HTML = re.compile(r"<.*?>")
CLEAN_URL  = re.compile(r"http\S+|www\.\S+", re.I)
TOKENIZER  = re.compile(r"[a-zA-Z']+")  # simple word tokens

def clean_text(s: str) -> str:
    """Very explicit cleaning for pedagogy:
    - lowercasing
    - unescape HTML entities
    - remove HTML tags
    - remove URLs
    - tokenize and remove stopwords
    """
    s = (s or "").strip().lower()
    s = html.unescape(s)
    s = CLEAN_HTML.sub(" ", s)
    s = CLEAN_URL.sub(" ", s)
    tokens = [w for w in TOKENIZER.findall(s) if w not in STOPWORDS]
    return " ".join(tokens)

@st.cache_data(show_spinner=False)
def load_csv() -> pd.DataFrame:
    """Load 'IMDB Dataset.csv' from repo root OR ./data/, robust to encoding."""
    candidates = [Path("IMDB Dataset.csv"), Path("data/IMDB Dataset.csv")]
    for p in candidates:
        if p.exists():
            csv_path = p
            break
    else:
        st.error(
            "❌ CSV not found.\n\n"
            "Put **IMDB Dataset.csv** at the repo root or in a **data/** folder."
        )
        st.stop()

    # Robust read: try utf-8, then latin-1; skip bad lines; keep only needed cols
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, on_bad_lines="skip")
            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]
            # The canonical Kaggle file uses 'review','sentiment'
            # Some variants use 'text','label' or 'content','sentiment'
            col_map = {
                "review": "review",
                "text": "review",
                "content": "review",
                "sentiment": "sentiment",
                "label": "sentiment",
                "target": "sentiment",
            }
            keep = {}
            for c in df.columns:
                if c in col_map and col_map[c] not in keep:
                    keep[col_map[c]] = c
            df = df[[keep["review"], keep["sentiment"]]].rename(
                columns={keep["review"]: "review", keep["sentiment"]: "sentiment"}
            )
            # Drop NA and strip
            df = df.dropna().copy()
            df["review"] = df["review"].astype(str).str.strip()
            df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
            # Filter to expected labels only
            df = df[df["sentiment"].isin(["positive", "negative"])]
            # Basic sanity
            if len(df) == 0:
                raise ValueError("No usable rows after cleaning columns/labels.")
            return df.reset_index(drop=True)
        except Exception as e:
            last_err = e
    st.error(f"❌ Failed to read 'IMDB Dataset.csv'.\n\nDetails: {last_err}")
    st.stop()

@st.cache_data(show_spinner=False)
def preprocess_df(df: pd.DataFrame, max_features: int, ngram_max: int):
    """Clean text + TF-IDF vectorization. Return (X_dense, y, vectorizer, clean_df)."""
    clean = df["review"].apply(clean_text)
    out = df.copy()
    out["clean"] = clean

    vec = TfidfVectorizer(
        max_features=int(max_features),
        ngram_range=(1, int(ngram_max)),
        dtype=np.float32
    )
    X_sparse = vec.fit_transform(clean)  # sparse CSR
    # Keras's validation_split requires dense arrays (or Tensors)
    X = X_sparse.toarray()
    y = (df["sentiment"].values == "positive").astype(np.int32)
    return X, y, vec, out

def build_model(input_dim: int) -> keras.Model:
    """Sequential Dense net per the spec (ReLU hidden, Sigmoid output)."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ============================================================
# 👋 Title & Intro
# ============================================================
st.markdown("# 🎬 IMDb Sentiment Classification — Neural Network (TF-IDF)")
st.write(
    "This playful lab trains a **simple neural network** to classify movie reviews "
    "from IMDb as **positive** or **negative**."
)

# ============================================================
# 1) Data Loading and Exploration
# ============================================================
st.header("1) Data Loading and Exploration")

df = load_csv()
st.success(f"Loaded {len(df):,} rows.")
st.dataframe(df.head(), use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Class distribution (sentiment):")
    fig, ax = plt.subplots()
    df["sentiment"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("")
    st.pyplot(fig)
with col_b:
    st.subheader("Review length (characters):")
    fig, ax = plt.subplots()
    df["review"].str.len().clip(0, 15000).plot(kind="hist", bins=40, ax=ax)
    ax.set_xlabel("Length (chars)")
    st.pyplot(fig)

# ============================================================
# 2) Data Preprocessing
# ============================================================
st.header("2) Data Preprocessing")
st.markdown("""
- Convert to **lowercase**  
- Remove **HTML** and **URLs**  
- Tokenize and remove **stop words**  
- Convert to numeric features via **TF-IDF** (this is what the neural net will consume)
""")

with st.spinner("Cleaning & vectorizing…"):
    X, y, vectorizer, df_preview = preprocess_df(df, max_features, ngram_max)

st.subheader("Preview of cleaned text:")
st.dataframe(df_preview[["review", "clean", "sentiment"]].head(), use_container_width=True)
st.markdown(f"**TF-IDF feature dimension:** `{X.shape[1]:,}`")

# ============================================================
# 3) Model Building
# ============================================================
st.header("3) Model Building")
model = build_model(X.shape[1])

# Show model summary nicely
st.code(
    model.summary(print_fn=lambda x: st.session_state.setdefault("_model_summary", []).append(x))
    or "\n".join(st.session_state["_model_summary"])
)

# ============================================================
# 4) Model Training
# ============================================================
st.header("4) Model Training")

# Split BEFORE Keras so test is held out
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=float(test_size), random_state=42, stratify=y
)

callbacks = []
if use_es:
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(patience),
            min_delta=float(min_delta),
            restore_best_weights=True
        )
    )

with st.spinner("Training the model…"):
    history = model.fit(
        X_train, y_train,
        epochs=int(epochs),
        batch_size=int(batch_size),
        validation_split=float(validation_split),  # works because X is dense
        callbacks=callbacks,
        verbose=0
    )
st.success("Training finished.")

# ============================================================
# 5) Evaluation
# ============================================================
st.header("5) Evaluation on Test Set")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
st.write(f"**Test Loss:** {test_loss:.4f} | **Test Accuracy:** {test_acc:.4f}")

# ============================================================
# 6) Visualization
# ============================================================
st.header("6) Visualization (Training Curves)")

hist = pd.DataFrame(history.history)
fig1, ax1 = plt.subplots()
ax1.plot(hist["loss"], label="train")
ax1.plot(hist["val_loss"], label="val")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(hist["accuracy"], label="train")
ax2.plot(hist["val_accuracy"], label="val")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend()
st.pyplot(fig2)

# ============================================================
# 📊 Confusion Matrix + Precision/Recall/F1 (on Test Set)
#    (Placed after evaluation and before the report)
# ============================================================
st.header("📊 Confusion Matrix & Precision/Recall/F1 (on Test Set)")

y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)
cm = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])

fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("True")
st.pyplot(fig_cm)

# Per-class metrics table
prec, rec, f1, support = metrics.precision_recall_fscore_support(y_test, y_pred, labels=[0,1], zero_division=0)
table = pd.DataFrame({
    "class": ["Negative (0)", "Positive (1)"],
    "precision": np.round(prec, 4),
    "recall":    np.round(rec, 4),
    "f1-score":  np.round(f1, 4),
    "support":   support
})
st.subheader("Per-class metrics:")
st.dataframe(table, use_container_width=True)

# Text report
st.subheader("Classification report (text):")
st.code(metrics.classification_report(y_test, y_pred, digits=2))

# ============================================================
# 7) Report
# ============================================================
st.header("7) Report — Insights & Next Steps")
st.markdown("""
**What we learned**
- **TF-IDF** transforms reviews into numeric features that work well with dense networks.
- **EarlyStopping** helps stop training when validation loss stops improving.
- **Confusion matrix** and **Precision/Recall/F1** reveal class-wise behavior beyond average accuracy.

**Challenges**
- Text cleaning choices (stopwords, n-grams, `max_features`) strongly affect performance and speed.
- Even mild class imbalance should be verified before training.
- Overfitting can happen quickly; regularization (dropout), EarlyStopping, and more data help.

**Potential improvements**
- Try more/larger hidden layers or different activations.
- Tune TF-IDF (bigrams/trigrams, `max_features`).
- Replace TF-IDF + Dense with pretrained embeddings (e.g., GloVe) or transformer models (e.g., DistilBERT) for stronger accuracy.
""")
