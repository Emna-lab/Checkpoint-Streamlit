# app_churn.py
# Churn App — Explore & Prepare → Visualize → Profiling → Train (Pipeline) → Predict
# Simple, pédagogique et robuste.
# Nouveauté : gestion des outliers (winsorisation par quantiles) DANS LA PIPELINE.

from pathlib import Path
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

# Profiling
from ydata_profiling import ProfileReport

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
import joblib

# ---- Silence uniquement l’avertissement matplotlib “glyph missing”
warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")

# ============================== Helpers / Transformers ==============================

def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def make_ohe():
    """OneHotEncoder qui renvoie toujours un array dense suivant la version de sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def clean_for_plot(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s = s.str.replace(r"[\t\r\n]+", " ", regex=True).str.replace(r"\s{2,}", " ", regex=True).str.strip()
    return s

def drop_high_corr(df, target, thr=0.95):
    nums = df.select_dtypes(include=[np.number]).columns.drop([target], errors="ignore")
    if len(nums) < 2:
        return df, []
    corr = df[nums].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > thr)]
    return df.drop(columns=to_drop, errors="ignore"), to_drop

# ------- CountEncoder compatible scikit-learn (clone OK) -------
class CountEncoder(BaseEstimator, TransformerMixin):
    """
    Encode catégories en fréquences observées au fit.
    - columns: tuple immuable -> clone() OK
    - renvoie un np.ndarray 2D
    """
    def __init__(self, columns=()):
        self.columns = tuple(columns)

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cols = self.columns or tuple(X.columns)
        self._cols_ = cols
        self._maps_ = {c: X[c].astype(str).value_counts() for c in cols}
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        if not hasattr(self, "_cols_"):
            return np.zeros((len(X), 0))
        arrs = []
        for c in self._cols_:
            arrs.append(X[c].astype(str).map(self._maps_[c]).fillna(0).astype(float).to_numpy())
        return np.vstack(arrs).T if arrs else np.zeros((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        if hasattr(self, "_cols_"):
            return np.array([f"{c}_count" for c in self._cols_], dtype=object)
        cols = self.columns or (input_features if input_features is not None else [])
        return np.array([f"{c}_count" for c in cols], dtype=object)

# ------- QuantileClipper (winsorisation) pour gérer les outliers -------
class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Clippe chaque variable numérique entre [quantile_low, quantile_high].
    - Appris sur l'ensemble d'entraînement (pas de fuite).
    - Renvoie un np.ndarray 2D pour être compatible ColumnTransformer.
    """
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = float(lower)
        self.upper = float(upper)

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        # garde les noms pour la transform
        self.columns_ = tuple(df.columns)
        # bornes par colonne
        self.low_ = df.quantile(self.lower)
        self.high_ = df.quantile(self.upper)
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=getattr(self, "columns_", None))
        for c in df.columns:
            lo = self.low_.get(c, None)
            hi = self.high_.get(c, None)
            if lo is not None and hi is not None:
                df[c] = df[c].clip(lo, hi)
        return df.to_numpy()

# ============================== Page ==============================

st.set_page_config(page_title="Churn Analyzer (Pipeline)", layout="wide")
st.markdown("<h1 style='text-align:center;margin:.2rem 0;'>Churn Analyzer — Explore • Prepare • Viz • Profile • Train • Predict</h1>", unsafe_allow_html=True)
st.caption("Target fixée: **CHURN**. `TOP_PACK` encodé par **fréquence** (pas One-Hot). Outliers gérés par **winsorisation** optionnelle (quantiles).")

st.markdown("""
<style>
h2,h3 { margin-top:.4rem }
.card { padding:.6rem .8rem; border-radius:10px; background:#0b0e12; border:1px solid #232831; }
</style>
""", unsafe_allow_html=True)

# ============================== Data loading ==============================

st.sidebar.header("1) Data loading")
uploaded = st.sidebar.file_uploader("Upload CSV (.csv / .csv.gz)", type=["csv", "gz"])
local_path = st.sidebar.text_input(
    "Or local path",
    value=str(Path(r"C:\Users\Admin\OneDrive\Documents\Bootcamp DataScience\expresso_churn_sample_100k.csv.gz"))
)

@st.cache_data(show_spinner=False)
def read_csv_cached(src):
    return pd.read_csv(src, low_memory=False)

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
    st.session_state.loaded_name = None

if st.sidebar.button("Reset file"):
    st.session_state.df_raw = None
    st.session_state.loaded_name = None
    _rerun()

try:
    if uploaded is not None and st.session_state.loaded_name != getattr(uploaded, "name", None):
        with st.spinner("Loading uploaded file..."):
            st.session_state.df_raw = read_csv_cached(uploaded)
        st.session_state.loaded_name = getattr(uploaded, "name", None)
        st.success(f"Loaded upload: {st.session_state.loaded_name} ({len(st.session_state.df_raw):,} rows)")
except Exception as e:
    st.error(f"Upload load failed: {e}")

try:
    if local_path and Path(local_path).exists() and st.session_state.loaded_name != local_path:
        with st.spinner(f"Loading local file: {local_path}"):
            st.session_state.df_raw = read_csv_cached(local_path)
        st.session_state.loaded_name = local_path
        st.success(f"Loaded path: {local_path} ({len(st.session_state.df_raw):,} rows)")
except Exception as e:
    st.error(f"Local path load failed: {e}")

df_raw = st.session_state.df_raw
if df_raw is None:
    st.info("⬅️ Load a dataset (upload or path).")
    st.stop()

# ============================== Target coercion ==============================

TARGET = "CHURN"
if TARGET not in df_raw.columns:
    st.error("Column 'CHURN' not found.")
    st.stop()

if not pd.api.types.is_numeric_dtype(df_raw[TARGET]):
    mapping = {"yes":1,"no":0,"Yes":1,"No":0,"true":1,"false":0,"True":1,"False":0}
    df_raw[TARGET] = pd.to_numeric(df_raw[TARGET].replace(mapping), errors="coerce")

df_raw = df_raw.dropna(subset=[TARGET]).copy()
df_raw[TARGET] = df_raw[TARGET].astype(int)

# ============================== Tabs ==============================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔍 Explore & Prepare", "📈 Visualize", "🔎 Profiling", "🤖 Train (Pipeline)", "🔮 Predict"]
)

if "df_prepared" not in st.session_state:
    st.session_state.df_prepared = df_raw.copy()
    st.session_state.prep_note = "Initial data (no preprocessing applied yet)."

# -------------------------- Explore & Prepare --------------------------
with tab1:
    st.subheader("Overview")
    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", df_raw.shape[0]); c2.metric("Columns", df_raw.shape[1]); c3.metric("Target", TARGET)

    b1,b2,b3,b4 = st.columns(4)
    if b1.button("Head"): st.dataframe(df_raw.head(), use_container_width=True)
    if b2.button("Tail"): st.dataframe(df_raw.tail(), use_container_width=True)
    if b3.button("Describe (numeric)"): st.dataframe(df_raw.select_dtypes(include=[np.number]).describe().T)
    if b4.button("Dtypes"): st.dataframe(df_raw.dtypes.to_frame("dtype"))

    st.subheader("Missing values (%)")
    miss = df_raw.isna().mean().mul(100).sort_values(ascending=False)
    st.dataframe(miss.to_frame("missing_%"))

    st.subheader("Duplicates")
    dup_cnt = int(df_raw.duplicated().sum()); dup_pct = 100.0 * dup_cnt / len(df_raw)
    cA,cB = st.columns(2)
    cA.info(f"Found **{dup_cnt}** duplicate rows ({dup_pct:.2f}%).")
    drop_dups = cB.checkbox("Drop duplicates", value=(dup_cnt>0))

    st.subheader("Drop columns with too many missing values")
    thr_nan = st.slider("Max % missing allowed (columns above are dropped)", 50, 99, 90, 1)
    high_nan_cols = miss[miss > thr_nan].index.tolist()
    st.write(f"Columns > {thr_nan}% NaN: {high_nan_cols or 'None'}")
    drop_high_nan = st.checkbox("Drop high-missing columns", value=(len(high_nan_cols)>0))

    st.subheader("Constant columns")
    nunique = df_raw.nunique(dropna=False); const_cols = nunique[nunique<=1].index.tolist()
    st.write(f"Detected constants: {const_cols or 'None'}")
    drop_consts = st.checkbox("Drop constant columns", value=(len(const_cols)>0))

    st.subheader("TENURE as ordered feature (optional)")
    tenure_order = ["B < 1 month","C 1-3 month","D 3-6 month","E 6-9 month","F 9-12 month","G 12-15 month","H 15-18 month","I 18-21 month","J 21-24 month","K > 24 month"]
    use_tenure_order = st.checkbox("Convert TENURE to ordinal codes (if present)", value=True)

    st.subheader("Drop highly correlated numeric features (optional)")
    use_corr_prune = st.checkbox("Enable", value=False)
    corr_thr = st.slider("Correlation threshold", 0.80, 0.99, 0.95, 0.01, disabled=not use_corr_prune)

    st.subheader("Drop additional columns (IDs, etc.)")
    id_like = [c for c in df_raw.columns if "id" in c.lower() and c != TARGET]
    extra_drop = st.multiselect("Columns to drop", options=[c for c in df_raw.columns if c != TARGET], default=id_like)

    if st.button("✅ Apply preprocessing"):
        df_p = df_raw.copy()

        if drop_dups and dup_cnt>0:
            df_p = df_p.drop_duplicates()

        if use_tenure_order and "TENURE" in df_p.columns:
            df_p["TENURE"] = pd.Categorical(df_p["TENURE"], categories=tenure_order, ordered=True).codes

        if drop_high_nan and high_nan_cols:
            df_p = df_p.drop(columns=high_nan_cols, errors="ignore")

        if drop_consts and const_cols:
            df_p = df_p.drop(columns=const_cols, errors="ignore")

        if extra_drop:
            df_p = df_p.drop(columns=extra_drop, errors="ignore")

        dropped_corr = []
        if use_corr_prune:
            df_p, dropped_corr = drop_high_corr(df_p, TARGET, thr=corr_thr)

        st.session_state.df_prepared = df_p
        note = []
        if drop_dups and dup_cnt>0: note.append(f"dropped duplicates ({dup_cnt}, {dup_pct:.2f}%)")
        if drop_high_nan and high_nan_cols: note.append(f"dropped high-missing {high_nan_cols}")
        if drop_consts and const_cols: note.append(f"dropped constants {const_cols}")
        if extra_drop: note.append(f"dropped {extra_drop}")
        if use_corr_prune and dropped_corr: note.append(f"dropped high-corr {dropped_corr} (>{corr_thr})")
        st.session_state.prep_note = "; ".join(note) or "Applied (no change)."
        st.success("Preprocessing applied.")

    st.markdown(f"**Prepared dataset:** {st.session_state.df_prepared.shape} — {st.session_state.prep_note}")

# -------------------------- Visualize --------------------------
with tab2:
    df = st.session_state.df_prepared
    st.subheader("Quick visualizations (prepared data)")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != TARGET]
    cat_cols = [c for c in df.columns if c != TARGET and not pd.api.types.is_numeric_dtype(df[c])]

    if num_cols:
        col_num = st.selectbox("Numeric (hist)", options=num_cols)
        bins = st.slider("Bins", 5, 60, 30, 5)
        fig, ax = plt.subplots(); ax.hist(df[col_num].dropna(), bins=bins); ax.set_title(f"Histogram — {col_num}")
        st.pyplot(fig)

        col_num2 = st.selectbox("Numeric vs target (boxplot)", options=num_cols, index=min(1,len(num_cols)-1))
        fig, ax = plt.subplots(); sns.boxplot(data=df, x=TARGET, y=col_num2, ax=ax); ax.set_title(f"{col_num2} by {TARGET}")
        st.pyplot(fig)

    if cat_cols:
        col_cat = st.selectbox("Categorical (churn rate)", options=cat_cols)
        top_k = st.slider("Top K categories (display)", 5, 60, 20, 5)
        clean = clean_for_plot(df[col_cat])
        freq = clean.value_counts(dropna=False).head(top_k)
        st.write("**Top categories by frequency (display):**"); st.dataframe(freq.to_frame("count"))
        mask = clean.isin(freq.index)
        tmp = pd.DataFrame({col_cat: clean[mask], TARGET: df.loc[mask, TARGET]})
        churn_by_cat = tmp.groupby(col_cat)[TARGET].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7,4)); churn_by_cat.plot(kind="bar", ax=ax); ax.set_title(f"Mean {TARGET} by {col_cat} (top {top_k})"); ax.tick_params(axis="x", rotation=30); plt.tight_layout(); st.pyplot(fig)

    if len(num_cols)>=2:
        st.subheader("Correlation heatmap (numeric)")
        fig, ax = plt.subplots(figsize=(6,4)); sns.heatmap(df[num_cols].corr(), ax=ax, annot=False); st.pyplot(fig)

# -------------------------- Profiling --------------------------
with tab3:
    df = st.session_state.df_prepared
    st.subheader("YData Profiling (optional, on sample)")
    st.caption("Le profiling peut être long : utilisez un échantillon.")
    st.info(f"Prepared dataset → rows: {len(df):,}, duplicates: {int(df.duplicated().sum())} ({100*df.duplicated().mean():.2f}%).")
    sample_first = st.checkbox("Profile a sample", value=True)
    sample_n = st.number_input("Sample size", 1_000, 200_000, 50_000, 5_000)
    if st.button("Generate profiling report"):
        df_prof = df.sample(n=min(sample_n, len(df)), random_state=42) if sample_first else df
        profile = ProfileReport(df_prof, title="Churn Profiling", explorative=True)
        html(profile.to_html(), height=900, scrolling=True)

# -------------------------- Train (Pipeline) --------------------------
with tab4:
    df = st.session_state.df_prepared.copy()
    st.subheader("Train a model (with a Pipeline)")
    st.caption("Imputation : **numérique = médiane**, **catégoriel = mode (OHE)**.  "
               "**TOP_PACK** : **CountEncoding** (inconnus & NaN → 0).  "
               "**Outliers** : winsorisation par **quantiles** (optionnelle).")
    X = df.drop(columns=[TARGET]); y = df[TARGET].astype(int)

    num_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_features = [c for c in X.columns if c not in num_features]

    top_pack_present = "TOP_PACK" in X.columns
    cats_ohe = [c for c in cat_features if c != "TOP_PACK"]

    model_choice = st.radio("Model", ["RandomForest (recommended)", "Logistic Regression"], horizontal=True)
    use_scaler = (model_choice == "Logistic Regression")
    calibrate_rf = st.checkbox("Calibrate RF probabilities (better probas, a bit slower)", value=False, disabled=not model_choice.startswith("RandomForest"))

    # ---- NEW: gestion outliers (winsorisation)
    st.markdown("**Outliers (winsorisation quantiles)**")
    use_winsor = st.checkbox("Handle outliers (winsorize numeric features)", value=False)
    c1, c2 = st.columns(2)
    with c1:
        lower_q = st.slider("Lower quantile", 0.00, 0.10, 0.01, 0.01, disabled=not use_winsor)
    with c2:
        upper_q = st.slider("Upper quantile", 0.90, 1.00, 0.99, 0.01, disabled=not use_winsor)
    if use_winsor and lower_q >= upper_q:
        st.warning("Lower quantile must be < upper quantile. Values adjusted.")
        lower_q = min(lower_q, 0.49)
        upper_q = max(upper_q, lower_q + 0.01)

    # Numériques : impute → winsor (option) → scale (option)
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_winsor:
        num_steps.append(("winsor", QuantileClipper(lower=lower_q, upper=upper_q)))
    if use_scaler:
        num_steps.append(("scaler", StandardScaler()))

    # Catégorielles (HORS TOP_PACK)
    cat_steps = [("imputer", SimpleImputer(strategy="most_frequent")),
                 ("ohe", make_ohe())]

    # TOP_PACK — CountEncoder EN PREMIER (pas d’imputer avant)
    tp_steps = [("count", CountEncoder(columns=("TOP_PACK",)))]
    if use_scaler:
        tp_steps.append(("scaler", StandardScaler()))

    transformers = []
    if num_features: transformers.append(("num", Pipeline(num_steps), num_features))
    if cats_ohe: transformers.append(("cat_ohe", Pipeline(cat_steps), cats_ohe))
    if top_pack_present: transformers.append(("top_pack", Pipeline(tp_steps), ["TOP_PACK"]))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if model_choice.startswith("RandomForest"):
        n_trees = st.slider("Trees", 50, 500, 300, 50)
        base_rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1, class_weight="balanced")
        clf = CalibratedClassifierCV(base_rf, method="isotonic", cv=3) if calibrate_rf else base_rf
        st.caption("RF robuste pour données tabulaires hétérogènes ; pas besoin de scaling.")
    else:
        max_iter = st.slider("Max iterations (LogReg)", 200, 3000, 2000, 100)
        clf = LogisticRegression(max_iter=max_iter, class_weight='balanced')

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", clf)])

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    if st.button("Train model"):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
            with st.spinner("Training..."):
                pipe.fit(X_train, y_train)

            # Scores basés sur probas si dispo
            if hasattr(pipe.named_steps["model"], "predict_proba"):
                proba_test = pipe.predict_proba(X_test)[:, 1]
                y_pred = (proba_test >= 0.5).astype(int)
            else:
                y_pred = pipe.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1  = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1]) if hasattr(pipe.named_steps["model"], "predict_proba") else np.nan

            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Accuracy", f"{acc:.3f}"); m2.metric("Precision", f"{prec:.3f}"); m3.metric("Recall", f"{rec:.3f}")
            m4.metric("F1", f"{f1:.3f}"); m5.metric("ROC-AUC", f"{auc:.3f}" if not np.isnan(auc) else "n/a")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(); sns.heatmap(cm, annot=True, fmt="d", ax=ax); ax.set_title("Confusion matrix"); st.pyplot(fig)

            # Seuil qui maximise F1 (sur test)
            best_thr = 0.5
            if hasattr(pipe.named_steps["model"], "predict_proba"):
                grid = np.linspace(0.05, 0.95, 181)
                f1s = [f1_score(y_test, (proba_test >= t).astype(int)) for t in grid]
                best_thr = float(grid[int(np.argmax(f1s))])
                st.caption(f"Seuil conseillé (max F1 sur le test) ≈ **{best_thr:.2f}**")

            # Importance des features (si dispo)
            try:
                feat_names = pipe.named_steps["prep"].get_feature_names_out()
                model = pipe.named_steps["model"]
                importances = None

                if isinstance(model, RandomForestClassifier):
                    importances = model.feature_importances_
                elif isinstance(model, CalibratedClassifierCV):
                    try:
                        arrs = [cal.base_estimator.feature_importances_ for cal in model.calibrated_classifiers_]
                        importances = np.mean(np.vstack(arrs), axis=0)
                    except Exception:
                        importances = None
                elif isinstance(model, LogisticRegression) and hasattr(model, "coef_"):
                    importances = np.abs(model.coef_.ravel())

                if importances is not None and len(importances) == len(feat_names):
                    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
                    imp_df["feature"] = imp_df["feature"].str.replace(r"^(num|cat_ohe|top_pack)__", "", regex=True)
                    st.subheader("Top features")
                    st.dataframe(imp_df.sort_values("importance", ascending=False).head(15))
            except Exception:
                pass

            # Sauvegarde modèle + seuil conseillé
            st.session_state.trained_pipe = pipe
            st.session_state.best_thr = best_thr
            st.success("Model trained and stored. Go to Predict tab.")

            # Download bouton
            if st.button("Download trained model (.pkl)"):
                buf = io.BytesIO(); joblib.dump(pipe, buf)
                st.download_button("pipeline.pkl", data=buf.getvalue(), file_name="pipeline.pkl")

        except Exception as e:
            st.error("Training failed. See details below.")
            st.exception(e)

# -------------------------- Predict --------------------------
with tab5:
    if "trained_pipe" not in st.session_state:
        st.warning("Train a model first in the **Train (Pipeline)** tab.")
        st.stop()

    df = st.session_state.df_prepared
    pipe = st.session_state.trained_pipe

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != TARGET]
    cat_cols = [c for c in df.columns if c != TARGET and not pd.api.types.is_numeric_dtype(df[c])]

    st.subheader("Predict")
    st.caption("Réglez les features puis choisissez un seuil de décision.")

    user_vals = {}
    cols = st.columns(2)

    # Numériques
    for i, col in enumerate(num_cols):
        cmin = float(np.nan_to_num(df[col].min(), nan=0.0))
        cmax = float(np.nan_to_num(df[col].max(), nan=1000.0))
        default = float(np.nan_to_num(df[col].median(), nan=(cmin+cmax)/2))
        step = (cmax-cmin)/100 if cmax>cmin else 1.0
        with cols[i%2]:
            user_vals[col] = st.number_input(col, value=default, min_value=cmin, max_value=cmax, step=step)

    # Catégorielles — valeurs RAW pour matcher l’encodage
    for i, col in enumerate(cat_cols):
        levels = df[col].dropna().astype(str).value_counts().index.tolist()
        levels = levels[:50] if len(levels)>50 else levels
        with cols[(i+len(num_cols))%2]:
            if col=="TOP_PACK":
                pick = st.selectbox(f"{col} (top 50)", options=levels or [""])
                other = st.text_input("Or type another TOP_PACK (optional)", value="")
                user_vals[col] = other.strip() if other.strip() else pick
            else:
                user_vals[col] = st.selectbox(col, options=levels or [""])

    cA,cB = st.columns(2)
    thr_default = float(st.session_state.get("best_thr", 0.50))
    decision_thr = st.slider("Seuil de décision (probabilité ≥ seuil ⇒ classe 1 - churn)", 0.05, 0.95, thr_default, 0.01)

    if cA.button("Predict"):
        X_new = pd.DataFrame([user_vals])
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            p = float(pipe.predict_proba(X_new)[0, 1])
            y_hat = int(p >= decision_thr)
            risk_pct = 100.0 * p
            txt = (f"**Risque estimé de churn : {risk_pct:.1f}%**  \n"
                   f"Seuil utilisé : **{decision_thr:.2f}**  →  "
                   f"**Décision : {'CHURN (1)' if y_hat==1 else 'NO CHURN (0)'}**")
        else:
            y_hat = int(pipe.predict(X_new)[0])
            txt = f"Modèle sans probas. Décision : **{y_hat}** (0=no churn, 1=churn)"
        st.success(txt)

    if cB.button("Reset stored model"):
        st.session_state.pop("trained_pipe", None)
        st.info("Model removed.")
        _rerun()
