# imdb_sentiment_app.py
# -----------------------------------------------------------------------------
# 🎯 Objective (EN): Train a neural network to classify IMDb reviews (pos/neg)
# 💡 This Streamlit app follows the requested, step-by-step educational flow.
# -----------------------------------------------------------------------------

import os
import io
import re
import unicodedata
import string
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple

# Reproducibility
import random
random.seed(42)
np.random.seed(42)

# NLTK for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================== NLTK safety download ==========================
def safe_nltk_download(resource, name):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(name)

safe_nltk_download('tokenizers/punkt', 'punkt')
safe_nltk_download('corpora/stopwords', 'stopwords')

EN_STOP = set(stopwords.words('english'))

# ============================== Streamlit Page Setup ==========================
st.set_page_config(page_title="IMDb Sentiment (DL, TF-IDF)", page_icon="🎬", layout="wide")
st.title("🎬 IMDb Sentiment Classification — Neural Network (TF-IDF)")

st.markdown("""
This playful lab trains a **simple neural network** to classify movie reviews from IMDb
as **positive** or **negative**.
""")

# ============================== Sidebar Controls =============================
st.sidebar.header("⚙️ Training controls")

# 1) Epochs: now allowed up to 100 (requested change)
epochs = st.sidebar.slider("Epochs (max 100)", min_value=1, max_value=100, value=10, step=1)

# 2) EarlyStopping controls (requested change)
use_es = st.sidebar.checkbox("Enable EarlyStopping (monitor = val_loss)", value=True)
es_patience = st.sidebar.slider("EarlyStopping patience (epochs without improvement)", 1, 20, 3)
es_min_delta = st.sidebar.number_input("EarlyStopping min_delta (loss improvement threshold)", min_value=0.0, value=0.0, step=0.001)

batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64, 128], index=1)
val_split = st.sidebar.slider("Validation split", 0.1, 0.3, 0.2, 0.05)
hidden_units = st.sidebar.slider("Hidden units", 64, 512, 128, 32)
hidden_dropout = st.sidebar.slider("Hidden dropout", 0.0, 0.6, 0.2, 0.05)
learning_rate = st.sidebar.selectbox("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], index=2)

st.sidebar.caption("""
**Tip:** EarlyStopping stops training when `val_loss` stops improving by at least `min_delta`
for `patience` epochs, and restores the best weights.
""")

# ============================== Helpers (preprocess) ==========================
def normalize_text(s: str) -> str:
    """Lowercase, strip accents, remove HTML/URLs, tokenize and remove stopwords & punctuation."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()

    # Remove HTML tags
    s = re.sub(r"<[^>]+>", " ", s)
    # Remove URLs
    s = re.sub(r"http\S+|www\.\S+", " ", s)

    # Normalize unicode (strip accents)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("M"))

    # Tokenize
    tokens = word_tokenize(s)

    # Remove stopwords & punctuation & keep simple alpha tokens
    tokens = [t for t in tokens
              if t not in EN_STOP and t not in string.punctuation and any(c.isalpha() for c in t)]

    return " ".join(tokens)

# ============================== 1) Data Loading & Exploration =================
st.header("1) Data Loading and Exploration")

csv_path = Path(__file__).parent / "IMDB Dataset.csv"
if not csv_path.exists():
    st.error("❌ Could not find `IMDB Dataset.csv` next to this script.")
    st.stop()

df = pd.read_csv(csv_path)

st.write("**Preview (first 5 rows):**")
st.dataframe(df.head())

# Basic exploration: class distribution and review length
col_a, col_b = st.columns(2)
with col_a:
    st.write("**Class distribution (sentiment):**")
    st.bar_chart(df['sentiment'].value_counts())

with col_b:
    st.write("**Review length (characters):**")
    lengths = df['review'].astype(str).apply(len)
    fig, ax = plt.subplots()
    sns.histplot(lengths, bins=50, ax=ax)
    ax.set_xlabel("Length (chars)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# ============================== 2) Data Preprocessing =========================
st.header("2) Data Preprocessing")

st.markdown("""
- Convert to lowercase  
- Remove **HTML** and **URLs**  
- Tokenize and remove **stop words**  
- Convert to numeric features via **TF-IDF** (this is what the neural net will consume)
""")

with st.spinner("Cleaning text (this can take a few seconds)…"):
    df['clean'] = df['review'].astype(str).apply(normalize_text)

st.write("**Preview of cleaned text:**")
st.dataframe(df[['review', 'clean', 'sentiment']].head(5))

# Train/Test split
X_text = df['clean'].values
y = (df['sentiment'].str.lower() == 'positive').astype(int).values  # positive=1, negative=0

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(X_train_text)
X_test  = vectorizer.transform(X_test_text)

st.write(f"**TF-IDF feature dimension:** `{X_train.shape[1]}`")

# ============================== 3) Model Building =============================
st.header("3) Model Building")

input_dim = X_train.shape[1]
model = keras.Sequential([
    # First layer (input_dim must match TF-IDF features)  ✅
    layers.Input(shape=(input_dim,), name="tfidf_input"),
    layers.Dense(hidden_units, activation="relu"),
    layers.Dropout(hidden_dropout),
    # You can experiment with more layers if you like (kept simple and readable)
    layers.Dense(1, activation="sigmoid")  # Output for binary classification
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

st.code(model.summary(print_fn=lambda x: st.text(x)), language="text")

# ============================== 4) Model Training =============================
st.header("4) Model Training")

callbacks = []
if use_es:
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=es_patience,
            min_delta=es_min_delta,
            mode="min",
            restore_best_weights=True,
            verbose=1
        )
    )

with st.spinner("Training the model…"):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,                       # ✅ now up to 100
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=0
    )

st.success("Training finished.")

# ============================== 5) Evaluation (Test set) ======================
st.header("5) Evaluation on Test Set")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
st.write(f"**Test Loss:** {test_loss:.4f}  |  **Test Accuracy:** {test_acc:.4f}")

# ---------- 6) Visualization: training curves ----------
st.header("6) Visualization (Training Curves)")

fig1, ax1 = plt.subplots()
ax1.plot(history.history['loss'], label='train_loss')
ax1.plot(history.history['val_loss'], label='val_loss')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Binary cross-entropy loss")
ax1.legend()
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(history.history['accuracy'], label='train_acc')
ax2.plot(history.history['val_accuracy'], label='val_acc')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
st.pyplot(fig2)

# =================== 3) Confusion Matrix + Precision/Recall/F1 =================
st.header("📊 Confusion Matrix & Precision/Recall/F1 (on Test Set)")

# Predictions -> labels
y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # 0=neg, 1=pos
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("True")
st.pyplot(fig_cm)

# Precision / Recall / F1
prec, rec, f1, sup = precision_recall_fscore_support(y_test, y_pred, labels=[0,1], zero_division=0)
report_df = pd.DataFrame({
    "class": ["Negative (0)", "Positive (1)"],
    "precision": np.round(prec, 4),
    "recall":    np.round(rec, 4),
    "f1-score":  np.round(f1, 4),
    "support":   sup
})
st.write("**Per-class metrics:**")
st.dataframe(report_df, use_container_width=True)

st.write("**Classification report (text):**")
st.code(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]), language="text")

# ============================== 7) Report (Insights) ==========================
st.header("7) Report — Insights & Next Steps")

st.markdown("""
**What we learned**
- TF-IDF transforms reviews into numeric features that work well with dense networks.
- EarlyStopping helps stop training when validation loss stops improving, often yielding better generalization.
- Confusion matrix and Precision/Recall/F1 reveal class-wise behavior beyond average accuracy.

**Challenges**
- Text cleaning choices (stopwords, n-grams, max_features) strongly affect performance and speed.
- Class imbalance is mild in IMDb, but always verify the distribution before training. 
- Overfitting can happen quickly; regularization (dropout), EarlyStopping, and more data help.

**Potential improvements**
- Try more/larger hidden layers or different activations.
- Tune TF-IDF (bigrams/trigrams, max_features).
- Replace TF-IDF + Dense with pretrained embeddings (e.g., GloVe) or modern text models (e.g., BERT/DistilBERT) for stronger accuracy.
""")
