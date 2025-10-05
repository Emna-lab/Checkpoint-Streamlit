# Appli_Inclusion_checkpoint2.py
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix  # <— AJOUT
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Pour les visualisations
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Page & small theme tweak
# =========================
st.set_page_config(
    page_title="Bank Inclusion — Clean • Explore • Model",
    page_icon="💳",
    layout="wide"
)
st.markdown(
    """
    <style>
    .stApp { background-color: white; }
    .metric-box {padding: 0.5rem 0.75rem; border:1px solid #eee; border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ===============
# Helper functions
# ===============
TARGET_COL = "bank_account"

def drop_unique_id(df: pd.DataFrame) -> pd.DataFrame:
    """Drop 'uniqueid' (case insensitive) if present."""
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in ("uniqueid",):
        if candidate in cols_lower:
            df = df.drop(columns=[cols_lower[candidate]])
    return df

def split_features(df: pd.DataFrame):
    """Return feature lists (numeric, categorical) excluding target & uniqueid."""
    df = drop_unique_id(df)
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    return num_cols, cat_cols

def build_pipeline(num_cols, cat_cols, model_name, rf_params, logreg_params):
    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("oh", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if model_name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            min_samples_split=rf_params["min_samples_split"],
            random_state=42,
            n_jobs=-1,
        )
    else:  # Logistic Regression
        clf = LogisticRegression(
            C=logreg_params["C"],
            penalty="l2",
            solver="liblinear",
            max_iter=logreg_params["max_iter"],
            random_state=42,
        )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

@st.cache_resource(show_spinner=False)
def make_profile_html(df: pd.DataFrame, sample_size: int) -> bytes:
    """
    Crée le rapport YData Profiling et renvoie les bytes HTML.
    (Pas d'appel à to_file pour éviter l'erreur 'multiple values for output_file')
    """
    try:
        from ydata_profiling import ProfileReport
    except Exception:
        return b""
    df_small = df.sample(n=min(sample_size, len(df)), random_state=42) if sample_size else df
    pr = ProfileReport(df_small, title="YData Profiling Report", minimal=True)
    html_bytes = pr.to_html().encode("utf-8")
    return html_bytes

def metrics_block(y_true, y_pred, y_proba):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:  st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
    with col2:  st.metric("Precision", f"{precision_score(y_true, y_pred, pos_label='Yes'):.3f}")
    with col3:  st.metric("Recall", f"{recall_score(y_true, y_pred, pos_label='Yes'):.3f}")
    with col4:  st.metric("F1", f"{f1_score(y_true, y_pred, pos_label='Yes'):.3f}")
    try:
        auc = roc_auc_score((y_true == "Yes").astype(int), y_proba)
    except Exception:
        auc = float("nan")
    with col5:  st.metric("ROC-AUC", f"{auc:.3f}" if not np.isnan(auc) else "n/a")

# =================
# Data loading area
# =================
st.title("Financial Inclusion • End-to-end app")

with st.expander("1) Charger un fichier CSV (ou laisser vide si déjà en mémoire)", expanded=False):
    up = st.file_uploader("CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        st.session_state["df_raw"] = df.copy()
        st.success(f"Fichier chargé : {up.name} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
    elif "df_raw" not in st.session_state:
        st.info("Aucun fichier chargé. Merci d'uploader votre dataset.")
        st.stop()

df_raw = st.session_state["df_raw"].copy()

# =======================
# Tabs
# =======================
tab_overview, tab_viz, tab_profile, tab_train, tab_predict = st.tabs(
    ["Overview & Cleaning", "Visualisations", "Profiling (YData)", "Modelling (Train)", "Prediction"]
)

# -----------------------
# Overview & Cleaning
# -----------------------
with tab_overview:
    st.subheader("Aperçu & Nettoyage")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Aperçu brut (head)**")
        st.dataframe(df_raw.head(), use_container_width=True)
        st.caption(f"{df_raw.shape[0]} lignes • {df_raw.shape[1]} colonnes")
    with right:
        st.markdown("**Info colonnes**")
        st.write(pd.DataFrame({
            "col": df_raw.columns,
            "dtype": df_raw.dtypes.astype(str),
            "n_unique": [df_raw[c].nunique() for c in df_raw.columns],
            "n_null": [df_raw[c].isna().sum() for c in df_raw.columns]
        }))

    st.markdown("---")
    st.markdown("### Nettoyage (appliqué sur un *working dataset*)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        do_duplicates = st.checkbox("Drop duplicates", value=True)
    with c2:
        do_trim_spaces = st.checkbox("Trim strings", value=True)
    with c3:
        do_drop_na = st.checkbox("Drop rows with any NA", value=False)
    with c4:
        forced_drop_uniqueid = st.checkbox("Drop 'uniqueid' (forcé)", value=True)

    apply_clean = st.button("Valider / Appliquer le nettoyage", type="primary")

    if apply_clean:
        work = df_raw.copy()

        # 0) Harmoniser colonnes **avant** tests
        if do_trim_spaces:
            for c in work.select_dtypes(include="object"):
                work[c] = work[c].astype(str).str.strip()

        # 1) Dédoublonnage
        if do_duplicates:
            work = work.drop_duplicates(ignore_index=True)

        # 2) Drop NA si demandé
        if do_drop_na:
            work = work.dropna()

        # 3) Drop uniqueid **systématique** si demandé
        if forced_drop_uniqueid:
            work = drop_unique_id(work)

        # 4) Toujours garantir que uniqueid n'existe plus
        work = drop_unique_id(work)

        # 5) Stocker
        st.session_state["df_work"] = work.copy()
        st.success("Nettoyage appliqué.")

    # Afficher le working df si dispo
    if "df_work" in st.session_state:
        st.markdown("#### Working dataset (après validation)")
        _show = st.session_state["df_work"].copy()
        _show = drop_unique_id(_show)  # sécurité
        st.dataframe(_show.head(20), use_container_width=True)
        st.caption(f"{_show.shape[0]} lignes • {_show.shape[1]} colonnes")
    else:
        st.info("Appliquez le nettoyage pour créer le working dataset.")

# -----------------------
# Visualisations (RICHES)
# -----------------------
with tab_viz:
    st.subheader("Visualisations")
    if "df_work" not in st.session_state:
        st.warning("Créez d'abord le working dataset (onglet Overview & Cleaning).")
        st.stop()

    df_vis = drop_unique_id(st.session_state["df_work"].copy())
    if TARGET_COL not in df_vis.columns:
        st.error(f"Colonne cible '{TARGET_COL}' introuvable.")
        st.stop()

    # Récupération des colonnes
    num_cols, cat_cols = split_features(df_vis)
    target_values = df_vis[TARGET_COL].dropna().unique().tolist()

    # === 1) Histogramme d'une NUMÉRIQUE vs target ===
    st.markdown("### Histogramme (numérique vs target)")
    num_col_sel = st.selectbox("Numeric column", options=num_cols, index=0 if num_cols else None)

    if num_col_sel is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        # Histogrammes superposés par classe + courbes KDE pour lisibilité
        for val, color in zip(sorted(target_values), ["#1f77b4", "#ff7f0e"]):
            subset = df_vis[df_vis[TARGET_COL] == val][num_col_sel].dropna()
            ax.hist(subset, bins=30, alpha=0.35, label=str(val), color=color)
            if subset.size > 1:
                sns.kdeplot(x=subset, ax=ax, color=color, lw=2)
        ax.set_title(f"Histogram of {num_col_sel} by {TARGET_COL}")
        ax.set_xlabel(num_col_sel)
        ax.set_ylabel("Count")
        ax.legend(title=TARGET_COL)
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # === 2) Bar plot d'une CATÉGORIELLE vs target (proportions / comptes) ===
    st.markdown("### Bar plot (catégorielle vs target)")
    cat_col_sel = st.selectbox("Categorical column", options=cat_cols, index=0 if cat_cols else None)
    show_prop = st.checkbox("Show proportions (else counts)", value=True)

    if cat_col_sel is not None:
        tmp = (
            df_vis
            .groupby([cat_col_sel, TARGET_COL])
            .size()
            .reset_index(name="count")
        )
        if show_prop:
            # Proportions par catégorie
            tmp["prop"] = tmp.groupby(cat_col_sel)["count"].transform(lambda x: x / x.sum())
            ycol, ylabel = "prop", "Proportion"
        else:
            ycol, ylabel = "count", "Count"

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=tmp, x=cat_col_sel, y=ycol, hue=TARGET_COL, ax=ax2)
        ax2.set_xlabel(cat_col_sel)
        ax2.set_ylabel(ylabel)
        ax2.set_title(f"Bar plot of {cat_col_sel} vs {TARGET_COL} ({'proportions' if show_prop else 'counts'})")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2, use_container_width=True)

    st.markdown("---")

    # === 3) Violin plot (numérique vs target) ===
    st.markdown("### Violin (numeric vs target)")
    num_col_violin = st.selectbox("Numeric column (violin)", options=num_cols, index=0 if num_cols else None)

    if num_col_violin is not None:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.violinplot(
            data=df_vis.dropna(subset=[num_col_violin, TARGET_COL]),
            x=TARGET_COL, y=num_col_violin,
            cut=0, inner="quartile", ax=ax3
        )
        ax3.set_title(f"Distribution of {num_col_violin} by {TARGET_COL} (with quartiles)")
        st.pyplot(fig3, use_container_width=True)

# -----------------------
# YData Profiling (corrigé)
# -----------------------
with tab_profile:
    st.subheader("Profiling (YData)")
    if "df_work" not in st.session_state:
        st.warning("Créez d'abord le working dataset (onglet Overview & Cleaning).")
        st.stop()

    df_prof = drop_unique_id(st.session_state["df_work"].copy())
    sample_size = st.slider("Taille de l'échantillon pour le rapport", 1000, min(10000, len(df_prof)), 3000, step=500)
    if st.button("Générer le rapport"):
        html_bytes = make_profile_html(df_prof, sample_size)
        if not html_bytes:
            st.error("ydata-profiling n'est pas disponible dans cet environnement.")
        else:
            st.download_button(
                "Télécharger le rapport HTML",
                data=html_bytes,
                file_name="ydata_profile.html"
            )
            st.components.v1.html(html_bytes.decode("utf-8"), height=700, scrolling=True)

# -----------------------
# Training (avec matrice de confusion)
# -----------------------
with tab_train:
    st.subheader("Modelling (Train)")
    if "df_work" not in st.session_state:
        st.warning("Créez d'abord le working dataset (onglet Overview & Cleaning).")
        st.stop()

    df_train = drop_unique_id(st.session_state["df_work"].copy())

    if TARGET_COL not in df_train.columns:
        st.error(f"Colonne cible '{TARGET_COL}' introuvable.")
        st.stop()

    num_cols, cat_cols = split_features(df_train)
    feature_cols = num_cols + cat_cols

    left, right = st.columns([1, 1])
    with left:
        model_name = st.selectbox("Modèle", ["RandomForest", "LogisticRegression"])
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, step=0.05)

    with right:
        st.markdown("**Hyperparamètres**")
        if model_name == "RandomForest":
            rf_params = {
                "n_estimators": st.slider("n_estimators", 50, 1000, 300, step=50),
                "max_depth": st.slider("max_depth", 2, 50, 12),
                "min_samples_split": st.slider("min_samples_split", 2, 20, 4),
            }
            logreg_params = {"C": 1.0, "max_iter": 1000}
        else:
            logreg_params = {
                "C": st.number_input("C (inverse of regularization strength)", 0.001, 10.0, 1.0, step=0.1),
                "max_iter": st.slider("max_iter", 100, 5000, 2000, step=100),
            }
            rf_params = {"n_estimators": 300, "max_depth": 12, "min_samples_split": 4}

    if st.button("Entraîner", type="primary"):
        X = df_train[feature_cols].copy()
        y = df_train[TARGET_COL].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        pipe = build_pipeline(num_cols, cat_cols, model_name, rf_params, logreg_params)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = (y_pred == "Yes").astype(float)

        st.session_state["model_pipe"] = pipe
        st.session_state["train_feature_cols"] = feature_cols

        st.success("Modèle entraîné.")
        metrics_block(y_test, y_pred, y_proba)

        # ======== MATRICES DE CONFUSION (ajout) ========
        st.markdown("### Confusion matrix")
        labels = ["No", "Yes"]  # ordre explicite

        # Matrice brute (comptes)
        cm_counts = confusion_matrix(y_test, y_pred, labels=labels)
        fig_cm1, ax_cm1 = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm_counts, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax_cm1)
        ax_cm1.set_xlabel("Predicted")
        ax_cm1.set_ylabel("True")
        ax_cm1.set_title("Counts")
        # Matrice normalisée par ligne (rappel par classe)
        cm_norm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")
        fig_cm2, ax_cm2 = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                    xticklabels=labels, yticklabels=labels, ax=ax_cm2, vmin=0, vmax=1)
        ax_cm2.set_xlabel("Predicted")
        ax_cm2.set_ylabel("True")
        ax_cm2.set_title("Normalized (per true class)")

        c1, c2 = st.columns(2)
        with c1: st.pyplot(fig_cm1, use_container_width=True)
        with c2: st.pyplot(fig_cm2, use_container_width=True)
        # ===============================================

# -----------------------
# Prediction
# -----------------------
with tab_predict:
    st.subheader("Prediction")
    if "model_pipe" not in st.session_state:
        st.warning("Entraînez d'abord un modèle dans l'onglet Modelling (Train).")
        st.stop()

    pipe = st.session_state["model_pipe"]
    df_ref = drop_unique_id(st.session_state["df_work"].copy())
    num_cols, cat_cols = split_features(df_ref)
    feature_cols = num_cols + cat_cols  # ordre cohérent avec le training

    mode = st.radio("Mode de prédiction", ["Upload CSV", "Saisie manuelle"], horizontal=True)

    if mode == "Upload CSV":
        pred_file = st.file_uploader("Fichier à prédire (mêmes colonnes features que training)", type=["csv"])
        if pred_file:
            df_pred = pd.read_csv(pred_file)
            df_pred = drop_unique_id(df_pred)  # sécurité
            missing = [c for c in feature_cols if c not in df_pred.columns]
            if missing:
                st.error(f"Colonnes manquantes pour la prédiction : {missing}")
            else:
                proba = pipe.predict_proba(df_pred[feature_cols])[:, 1]
                pred = (proba >= 0.5).astype(int)
                out = df_pred.copy()
                out["pred_proba_churn(Yes)"] = proba
                out["pred_label"] = np.where(pred == 1, "Yes", "No")
                st.dataframe(out.head(30), use_container_width=True)
                st.download_button("Télécharger les prédictions", out.to_csv(index=False).encode("utf-8"),
                                   file_name="predictions.csv")
    else:
        st.markdown("**Saisie manuelle des features (1 ligne)**")

        # CATEGORICAL
        st.markdown("##### Categorical inputs")
        man_vals = {}
        for c in cat_cols:
            options = sorted(df_ref[c].dropna().astype(str).unique().tolist())
            man_vals[c] = st.selectbox(c, options=options, index=0 if options else None)

        # NUMERIC
        st.markdown("##### Numeric inputs")
        num_inputs = {}
        for c in num_cols:
            col_min = float(np.nanmin(df_ref[c])) if df_ref[c].notna().any() else 0.0
            col_max = float(np.nanmax(df_ref[c])) if df_ref[c].notna().any() else 1.0
            default = float(df_ref[c].median()) if df_ref[c].notna().any() else 0.0
            num_inputs[c] = st.number_input(c, value=default, min_value=col_min, max_value=col_max, step=1.0)

        if st.button("Prédire cette ligne", type="primary"):
            row = {**man_vals, **num_inputs}
            df_one = pd.DataFrame([row], columns=feature_cols)  # ensure order/cols
            proba = pipe.predict_proba(df_one)[:, 1][0]
            label = "Yes" if proba >= 0.5 else "No"
            st.success(f"**Prediction : {label}** (prob≈{proba:.2f})")
