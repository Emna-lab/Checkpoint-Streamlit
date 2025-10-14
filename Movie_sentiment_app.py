# ================================ IMPORTS =====================================
# We keep imports explicit and beginner-friendly: each one has a clear role.
import os
import re
import io
import html
import string
from pathlib import Path

import numpy as np
import pandas as pd

import streamlit as st

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML & feature engineering
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
)
from sklearn.feature_extraction.text import TfidfVectorizer

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ==================== MAKE NLTK DOWNLOADS ROBUST ON STREAMLIT =================
# On Streamlit Cloud, the ~/nltk_data folder may already exist.
# We create it if needed and add it to nltk.data.path to avoid FileExistsError.
NLTK_DIR = Path.home() / "nltk_data"
NLTK_DIR.mkdir(parents=True, exist_ok=True)
nltk.data.path.append(str(NLTK_DIR))

def safe_nltk_download(resource, name):
    """Try to find an NLTK resource; download it quietly if missing."""
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(name, download_dir=str(NLTK_DIR), quiet=True)

safe_nltk_download('tokenizers/punkt', 'punkt')
safe_nltk_download('corpora/stopwords', 'stopwords')

EN_STOP = set(stopwords.words('english'))


# ============================= STREAMLIT HEADER ================================
st.set_page_config(page_title="IMDb Sentiment — Simple NN (TF-IDF + Keras)",
                   page_icon="🎬", layout="wide")
st.title("🎬 IMDb Movie Review Sentiment — Simple Neural Network (TF-IDF + Keras)")
st.write("""
**Goal (for beginners):** Train a small neural network to classify IMDb reviews as **positive** or **negative**.  
This app is deliberately simple and pedagogical, with very explicit comments and visuals.
""")

st.info("➡️ Place a file named **`IMDB Dataset.csv`** at the app root **or** in a `data/` folder. "
        "It must contain two columns: `review` (text) and `sentiment` (`positive`/`negative`).")


# ============================ HELPER: READ CSV SAFE ============================
@st.cache_data(show_spinner=False)
def read_imdb_csv(csv_path: Path) -> pd.DataFrame:
    """
    Robust CSV reader:
      - Autodetects a plausible separator among [',', ';', '\\t', '|'].
      - Normalizes column names to lower-case and strips spaces.
      - Maps 'review' and 'sentiment' columns even if small naming variations exist.

    Raises a clear error if format is not compatible.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    # Peek first kilobytes to guess delimiter in a simple way
    sample = csv_path.read_text(encoding="utf-8", errors="ignore")[:5000]
    seps = [',', ';', '\t', '|']
    sep_counts = {s: sample.count(s) for s in seps}
    sep = max(sep_counts, key=sep_counts.get) if sep_counts else ','

    try:
        df = pd.read_csv(csv_path, sep=sep, encoding="utf-8", on_bad_lines="skip")
    except Exception as e:
        raise ValueError(f"❌ Failed to read '{csv_path.name}'. Details: {e}")

    if df.empty or df.shape[1] < 2:
        raise ValueError("❌ Empty or invalid file: not enough columns to parse.")

    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]

    # Map columns flexibly
    col_map = {}
    for c in df.columns:
        if 'review' in c:
            col_map[c] = 'review'
        elif 'sentiment' in c or 'label' in c or 'target' in c:
            col_map[c] = 'sentiment'

    df = df.rename(columns=col_map)

    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("❌ CSV must have columns named (or mappable to) 'review' and 'sentiment'.")

    # Keep only what we need
    df = df[['review', 'sentiment']].dropna()

    # Normalize sentiment values to {0,1}
    df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
    mapping = {'positive': 1, 'pos': 1, '1': 1, 'negative': 0, 'neg': 0, '0': 0}
    if not set(df['sentiment']).issubset(set(mapping.keys())):
        raise ValueError("❌ Sentiment values must be 'positive'/'negative' (or equivalents like pos/neg/1/0).")
    df['sentiment'] = df['sentiment'].map(mapping).astype(int)

    # Small sanity cleanup
    df['review'] = df['review'].astype(str)
    df = df[df['review'].str.strip().ne('')]

    return df.reset_index(drop=True)


# ========================== TEXT PREPROCESSING PIPELINE ========================
def clean_text(s: str) -> str:
    """
    Very simple and explainable cleaning:
      1) HTML entity unescape (e.g., &amp; -> &)
      2) Lowercase the text
      3) Remove HTML tags
      4) Remove URLs
      5) Remove extra spaces
    """
    s = html.unescape(s)
    s = s.lower()
    s = re.sub(r"<[^>]+>", " ", s)                          # remove HTML tags
    s = re.sub(r"http\S+|www\.\S+", " ", s)                 # remove URLs
    s = re.sub(r"[^a-z0-9' ]+", " ", s)                     # keep letters/digits/space/simple apostrophes
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_and_remove_stopwords(s: str):
    """
    Tokenize with NLTK and remove English stop words.
    We keep it readable and easy to follow.
    """
    tokens = word_tokenize(s)
    tokens = [t for t in tokens if t not in EN_STOP and t not in string.punctuation]
    return " ".join(tokens)


@st.cache_data(show_spinner=False)
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to the DataFrame (review column).
    Returns a *new* DataFrame with a 'review_clean' column.
    """
    df = df.copy()
    df['review_clean'] = df['review'].apply(clean_text).apply(tokenize_and_remove_stopwords)
    return df


# =============================== LOAD THE DATA =================================
# We accept two possible locations: ./IMDB Dataset.csv or ./data/IMDB Dataset.csv
root_csv = Path(__file__).parent / "IMDB Dataset.csv"
data_csv = Path(__file__).parent / "data" / "IMDB Dataset.csv"

csv_path = root_csv if root_csv.exists() else data_csv

if not csv_path.exists():
    st.error("❌ Could not find `IMDB Dataset.csv` at the app root or in a `data/` folder.")
    st.stop()

try:
    df = read_imdb_csv(csv_path)
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(f"✅ Loaded {len(df):,} rows.")
with st.expander("👀 Peek at the first rows"):
    st.dataframe(df.head(), use_container_width=True)

# Class balance
col_a, col_b = st.columns(2)
with col_a:
    st.write("**Class distribution**")
    class_counts = df['sentiment'].value_counts().sort_index()
    fig, ax = plt.subplots()
    sns.barplot(x=['Negative (0)', 'Positive (1)'], y=class_counts.values, ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Sentiment counts")
    st.pyplot(fig, clear_figure=True)
with col_b:
    st.write("**Review length (chars)**")
    lens = df['review'].str.len()
    fig2, ax2 = plt.subplots()
    sns.histplot(lens, bins=40, ax=ax2)
    ax2.set_xlabel("Characters per review")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2, clear_figure=True)


# ================================ PREPROCESSING ================================
st.header("🧹 Step 1 — Preprocess Text")
with st.spinner("Cleaning and tokenizing…"):
    df_clean = preprocess_dataframe(df)

st.success("Text preprocessed → created column `review_clean`.")
with st.expander("🔎 Example cleaned text"):
    st.write(df_clean[['review', 'review_clean']].head(3))


# ================================ TF-IDF SETUP =================================
st.header("🧮 Step 2 — TF-IDF Vectorization")

# Simple, transparent TF-IDF configuration for beginners
max_features = st.slider(
    "TF-IDF max features (larger = more expressive but heavier)",
    min_value=5_000, max_value=50_000, value=20_000, step=5_000
)
ngram = st.selectbox("n-gram range", options=["(1,1) unigrams", "(1,2) uni+bi"], index=1)
ngram_range = (1, 2) if "1,2" in ngram else (1, 1)

vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

with st.spinner("Fitting TF-IDF and transforming…"):
    X = vectorizer.fit_transform(df_clean['review_clean'])
    y = df_clean['sentiment'].values

st.write(f"TF-IDF output shape: **{X.shape}** (sparse CSR matrix)")
st.caption("Note: Keras Dense layers expect dense input, so we will convert only the needed parts to NumPy arrays.")


# ================================ TRAIN/TEST SPLIT =============================
st.header("✂️ Step 3 — Split Data")
test_size = st.slider("Test size", 0.1, 0.3, 0.2, 0.05)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)
st.write(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# Create a REAL validation set (we must not use validation_split with CSR)
val_size = st.slider("Validation size (fraction of train)", 0.1, 0.3, 0.15, 0.05)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
)
st.write(f"Final: Train {X_train.shape[0]} | Val {X_val.shape[0]} | Test {X_test.shape[0]}")


# ================================ BUILD THE MODEL ==============================
st.header("🏗️ Step 4 — Build the Neural Network")

input_dim = X.shape[1]  # number of TF-IDF features

with st.expander("Model hyperparameters", expanded=True):
    hidden_units = st.slider("Hidden units (layer 1)", 64, 512, 256, 32)
    hidden_units2 = st.slider("Hidden units (layer 2)", 0, 512, 128, 32)
    dropout_rate = st.slider("Dropout (regularization)", 0.0, 0.7, 0.3, 0.05)
    learning_rate = st.selectbox("Learning rate", [1e-3, 5e-4, 1e-4], index=0)

# Build a simple, readable Sequential model
def build_model(input_dim: int) -> keras.Model:
    """
    A small Dense network for binary classification.
    We keep it simple so beginners can map code → concept.
    """
    model = keras.Sequential(name="tfidf_dense_classifier")
    model.add(layers.Input(shape=(input_dim,), name="tfidf_input"))
    model.add(layers.Dense(hidden_units, activation="relu", name="dense_1"))
    if hidden_units2 > 0:
        model.add(layers.Dense(hidden_units2, activation="relu", name="dense_2"))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="dropout"))
    model.add(layers.Dense(1, activation="sigmoid", name="output"))  # sigmoid for binary
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

model = build_model(input_dim)
st.code(model.summary(print_fn=lambda x: st.text(x)), language="text")


# ================================ TRAIN THE MODEL ==============================
st.header("🏃 Step 5 — Train")

# Controls required by your spec
epochs = st.slider("Max epochs", min_value=1, max_value=100, value=10, step=1)
batch_size = st.selectbox("Batch size", [32, 64, 128, 256], index=1)

# EarlyStopping (configurable)
st.subheader("Early stopping")
patience = st.slider("Patience (epochs without improvement)", 1, 10, 3, 1)
min_delta = st.number_input("Minimum loss improvement (min_delta)", min_value=0.0, value=0.0, step=0.001, format="%.3f")
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", mode="min", patience=patience, min_delta=min_delta, restore_best_weights=True, verbose=0
)

# Convert only the subsets we will use into dense NumPy arrays
with st.spinner("Converting sparse matrices to dense (only what's needed for Keras)…"):
    X_train_dense = X_train.toarray()
    X_val_dense = X_val.toarray()

with st.spinner("Training the model…"):
    history = model.fit(
        X_train_dense, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_dense, y_val),  # ✅ real validation set (no CSR issue)
        callbacks=[early_stop],
        verbose=0
    )

st.success("Training finished (best weights restored).")

# Visualize training curves
hist = history.history
col_l, col_r = st.columns(2)
with col_l:
    fig, ax = plt.subplots()
    ax.plot(hist["loss"], label="train")
    ax.plot(hist["val_loss"], label="val")
    ax.set_title("Loss over epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary crossentropy")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

with col_r:
    fig, ax = plt.subplots()
    ax.plot(hist["accuracy"], label="train")
    ax.plot(hist["val_accuracy"], label="val")
    ax.set_title("Accuracy over epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig, clear_figure=True)


# ================================== EVALUATION =================================
st.header("🧪 Step 6 — Evaluate on Test Set")

with st.spinner("Preparing test set…"):
    X_test_dense = X_test.toarray()

test_loss, test_acc = model.evaluate(X_test_dense, y_test, verbose=0)
st.success(f"Test accuracy: **{test_acc:.3f}** | Test loss: **{test_loss:.3f}**")


# =================== CONFUSION MATRIX + PRECISION/RECALL/F1 ====================
st.header("📊 Step 7 — Confusion Matrix & Precision/Recall/F1")

y_proba = model.predict(X_test_dense, verbose=0).ravel()
y_pred = (y_proba >= 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred: Neg (0)", "Pred: Pos (1)"],
            yticklabels=["True: Neg (0)", "True: Pos (1)"],
            ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig, clear_figure=True)

# Precision / Recall / F1
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)
st.write(f"**Precision (positive=1)**: {prec:.3f}  |  **Recall**: {rec:.3f}  |  **F1**: {f1:.3f}")

# Full classification report (per class)
st.text("Classification report:\n" + classification_report(y_test, y_pred, digits=3))


# ==================================== REPORT ===================================
st.header("📝 Step 8 — Report (Insights, Challenges, Improvements)")

st.markdown("""
**Insights gained:**
- A simple TF-IDF + Dense NN can already reach strong accuracy on IMDb reviews.
- Bigger `max_features` and `(1,2)` n-grams often help, at the cost of memory.
- Early stopping on `val_loss` prevents overfitting and speeds up training.

**Challenges faced and solutions:**
- *CSR matrices + Keras `validation_split`* → **Fix:** create an explicit validation set with `train_test_split` and pass `validation_data=(X_val, y_val)`.
- *Sparse input to Dense layers* → **Fix:** convert only the needed subsets to **dense** with `.toarray()`.
- *NLTK downloads failing on Streamlit Cloud* → **Fix:** create `~/nltk_data` if needed and pass `download_dir` + `quiet=True`.

**Potential improvements:**
- Try classical linear models (e.g., Logistic Regression, Linear SVM) — often competitive on TF-IDF.
- Add text cleaning extras (lemmatization, handling negations).
- Explore CNN/bi-LSTM/transformers with embeddings for richer representations.
- Use calibration for probabilities or threshold tuning for better precision/recall trade-offs.
""")
