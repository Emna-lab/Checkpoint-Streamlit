# My_face_verificator_V1.py
# -------------------------------------------------------------------
# BankID • Face Verification (PoC)
# Streamlit app avec flux "snapshot" (st.camera_input), UX Start/Stop,
# et métrique simple (cosine similarity) sur visages recadrés 128x128.
# -------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2                  # OpenCV : lecture/traitement d'images
import numpy as np          # NumPy  : tableaux, normalisation, cosinus
import streamlit as st      # Streamlit : UI web

# ================== Page & style minimal ==================
st.set_page_config(page_title="BankID • Face Verification (PoC)", page_icon="🏦", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #ffffff; }
      .app-header { text-align:center; margin-top:.4rem; }
      .brand { font-size: 1.85rem; font-weight: 800; letter-spacing:.3px; color:#0f172a;}
      .subtitle { color:#475569; margin-top:.15rem; }
      .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 14px; background:#fff; }
      .section-title { font-weight:700; color:#0f172a; margin:.35rem 0 .5rem 0; }
      .badge { display:inline-block; padding:.25rem .6rem; border-radius:999px; font-weight:700; border:1px solid #e5e7eb;}
      .ok { background:#ecfdf5; color:#065f46; border-color:#a7f3d0;}
      .ko { background:#fef2f2; color:#991b1b; border-color:#fecaca;}
      .metric { font-size:1.6rem; font-weight:800; }
      .muted { color:#64748b; }
      .soft { color:#6b7280; }
      .btn-row > div { display:flex; gap:.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== Entête ==================
st.markdown(
    """
    <div class="app-header">
      <div class="brand">BankID • Face Verification (PoC)</div>
      <div class="subtitle">Snapshot-based verification using your camera.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ================== Cascade de visage (OpenCV) ==================
# Option : on tente d'abord le fichier local, sinon le chemin par défaut d’OpenCV
CASCADE_FILE = "haarcascade_frontalface_default.xml"
local_cascade = Path(__file__).parent / CASCADE_FILE
opencv_cascade = Path(cv2.data.haarcascades) / CASCADE_FILE
CASCADE_PATH = local_cascade if local_cascade.exists() else opencv_cascade

face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
if face_cascade.empty():
    st.error(
        "❌ Haar cascade introuvable ou corrompue.\n"
        f"Copiez `{CASCADE_FILE}` à côté de ce fichier ou vérifiez le chemin OpenCV :\n{opencv_cascade}"
    )
    st.stop()

# ================== Helpers ==================
def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """#RRGGBB -> (B, G, R) pour OpenCV."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return (b, g, r)

def bgr_from_upload(up) -> Optional[np.ndarray]:
    """Convertit un UploadedFile (upload ou camera_input) -> image BGR (OpenCV)."""
    if up is None:
        return None
    data = up.getvalue() if hasattr(up, "getvalue") else up.read()
    if not data:
        return None
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def largest_face(gray: np.ndarray, scale: float, neigh: int):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neigh)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])  # plus grande bbox

def face_vector(bgr: np.ndarray, scale: float, neigh: int, size: int = 128) -> Optional[np.ndarray]:
    """Retourne un vecteur normalisé du visage recadré (128x128 grayscale)."""
    if bgr is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bbox = largest_face(gray, scale, neigh)
    if bbox is None:
        return None
    x, y, w, h = bbox
    crop = gray[y:y+h, x:x+w]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    vec = crop.astype(np.float32).ravel()
    n = np.linalg.norm(vec)
    if n < 1e-8:
        return None
    return vec / n

def draw_boxes(bgr: np.ndarray, color_bgr: Tuple[int, int, int], scale: float, neigh: int) -> Optional[np.ndarray]:
    if bgr is None:
        return None
    out = bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neigh)
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x+w, y+h), color_bgr, 2)
    return out

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Produit scalaire (les vecteurs sont normalisés) = cosinus."""
    return float(np.dot(a, b))

def reset_result():
    st.session_state.last_result = None

# ================== État par défaut ==================
defaults = dict(
    ref_vec=None, ref_img=None, proof_img=None, last_result=None,
    rect_hex="#2563eb", scale_factor=1.30, min_neighbors=5,
    threshold=0.86,                           # décision
    ref_cam_on=False, proof_cam_on=False,     # UX Start/Stop
    allow_downloads=False                     # pas d'écriture disque par défaut
)
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ================== Instructions (centre) ==================
st.markdown(
    """
<div class="card">
  <div class="section-title">How to use</div>
  <ol class="muted">
    <li><b>Reference</b> : démarrez la caméra (ou uploadez une photo) et enregistrez la référence.</li>
    <li><b>Proof</b> : démarrez la caméra et prenez une photo à vérifier.</li>
    <li>Cliquez sur <b>Verify</b> pour obtenir la similarité, le pourcentage et la décision.</li>
  </ol>
</div>
    """,
    unsafe_allow_html=True,
)

# ================== Sidebar : paramètres (avec explications) ==================
with st.sidebar:
    st.header("Parameters")

    # Le on_change met à jour la décision immédiatement (sinon « lag »)
    st.session_state.rect_hex = st.color_picker("Rectangle color", st.session_state.rect_hex)
    st.caption("Couleur des boîtes dessinées sur les visages (visuel uniquement).")

    def _on_param_change():
        # Toute modif de param invalide le dernier résultat :
        reset_result()

    st.session_state.scale_factor = st.slider(
        "scaleFactor", 1.05, 1.60, st.session_state.scale_factor, 0.01, on_change=_on_param_change
    )
    st.caption("Taille de l’échelle pour la détection (>1.0). Plus grand = balayage plus grossier/rapide (1.1–1.4 typiquement).")

    st.session_state.min_neighbors = st.slider(
        "minNeighbors", 1, 12, st.session_state.min_neighbors, 1, on_change=_on_param_change
    )
    st.caption("Filtrage des faux positifs. Plus grand = moins de faux positifs mais visage plus stable requis.")

    st.session_state.threshold = st.slider(
        "Decision threshold (cosine)", 0.50, 0.98, st.session_state.threshold, 0.01, on_change=_on_param_change
    )
    st.caption("Seuil de décision. Similarité ≥ seuil ⇒ PASS. Plus élevé = décision plus stricte.")

    st.divider()
    st.header("Downloads")
    st.session_state.allow_downloads = st.checkbox(
        "Allow download of snapshots (PNG)", value=st.session_state.allow_downloads
    )
    st.caption("Option PoC : propose un bouton de téléchargement des images affichées (aucune écriture disque côté serveur).")

# ================== Mise en page : gauche = actions, droite = résultat ==================
left, right = st.columns([6, 6])

# ================== Colonne droite : Résultat ==================
with right:
    st.markdown('<div class="section-title">Verification result</div>', unsafe_allow_html=True)
    res = st.session_state.last_result
    if res is None:
        st.caption("Capture a reference and a proof, then click **Verify**.")
    else:
        sim, percent, thr, passed = res["similarity"], res["percent"], res["threshold"], res["passed"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Similarity**")
            st.markdown(f"<div class='metric'>{sim:.3f}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("**Match (%)**")
            st.markdown(f"<div class='metric'>{percent:.1f}%</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("**Threshold**")
            st.markdown(f"<div class='metric'>{thr:.2f}</div>", unsafe_allow_html=True)

        st.markdown("**Decision**")
        if passed:
            st.markdown("<span class='badge ok'>✅ PASS</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge ko'>❌ FAIL</span>", unsafe_allow_html=True)

        st.caption("Cosine similarity on 128×128 grayscale face crops. % is clipped to [0–100].")

# ================== Colonne gauche : Référence / Preuve / Vérifier ==================
with left:
    # --------- Step 1: Reference ----------
    st.markdown('<div class="section-title">1) Reference</div>', unsafe_allow_html=True)

    btn_ref_col, _ = st.columns([1, 1])
    with btn_ref_col:
        if not st.session_state.ref_cam_on:
            if st.button("▶ Start camera (reference)"):
                st.session_state.ref_cam_on = True
        else:
            if st.button("■ Stop camera (reference)"):
                st.session_state.ref_cam_on = False

    cols_ref = st.columns([1, 1])
    with cols_ref[0]:
        ref_upload = st.file_uploader("Or upload a reference photo", type=["jpg", "jpeg", "png"])
        if ref_upload is not None:
            img = bgr_from_upload(ref_upload)
            vec = face_vector(img, st.session_state.scale_factor, st.session_state.min_neighbors)
            if vec is None:
                st.error("No face found in the uploaded image.")
            else:
                st.session_state.ref_vec = vec
                st.session_state.ref_img = draw_boxes(
                    img, hex_to_bgr(st.session_state.rect_hex),
                    st.session_state.scale_factor, st.session_state.min_neighbors
                )
                reset_result()
                st.success("Reference saved (from upload).")

    with cols_ref[1]:
        if st.session_state.ref_cam_on:
            ref_cam = st.camera_input("Reference snapshot", key="ref_cam_input")
            if ref_cam is not None:
                img = bgr_from_upload(ref_cam)
                vec = face_vector(img, st.session_state.scale_factor, st.session_state.min_neighbors)
                if vec is None:
                    st.error("No face detected in the reference snapshot.")
                else:
                    st.session_state.ref_vec = vec
                    st.session_state.ref_img = draw_boxes(
                        img, hex_to_bgr(st.session_state.rect_hex),
                        st.session_state.scale_factor, st.session_state.min_neighbors
                    )
                    reset_result()
                    st.success("Reference saved (from camera).")

    # Aperçu + clear
    if st.session_state.ref_img is not None:
        st.image(cv2.cvtColor(st.session_state.ref_img, cv2.COLOR_BGR2RGB),
                 caption="Reference", use_container_width=True)
        if st.button("🧹 Clear reference"):
            st.session_state.ref_vec = None
            st.session_state.ref_img = None
            reset_result()
            st.success("Reference cleared.")

    st.markdown('---')

    # --------- Step 2: Proof ----------
    st.markdown('<div class="section-title">2) Proof</div>', unsafe_allow_html=True)
    btn_proof_col, _ = st.columns([1, 1])
    with btn_proof_col:
        if not st.session_state.proof_cam_on:
            if st.button("▶ Start camera (proof)"):
                st.session_state.proof_cam_on = True
        else:
            if st.button("■ Stop camera (proof)"):
                st.session_state.proof_cam_on = False

    if st.session_state.proof_cam_on:
        proof_cam = st.camera_input("Proof snapshot", key="proof_cam_input")
        if proof_cam is not None:
            img = bgr_from_upload(proof_cam)
            st.session_state.proof_img = draw_boxes(
                img, hex_to_bgr(st.session_state.rect_hex),
                st.session_state.scale_factor, st.session_state.min_neighbors
            )
            reset_result()
            st.success("Proof snapshot captured.")

    if st.session_state.proof_img is not None:
        st.image(cv2.cvtColor(st.session_state.proof_img, cv2.COLOR_BGR2RGB),
                 caption="Proof", use_container_width=True)

    st.markdown('---')

    # --------- Step 3: Verify ----------
    st.markdown('<div class="section-title">3) Verify</div>', unsafe_allow_html=True)
    if st.button("✅ Verify"):
        if st.session_state.ref_vec is None:
            st.warning("Please set a reference first (upload or camera).")
        elif st.session_state.proof_img is None:
            st.warning("Please take a proof snapshot.")
        else:
            # Recalcule le vecteur de preuve à partir de l’image courante :
            proof_vec = face_vector(st.session_state.proof_img,
                                    st.session_state.scale_factor,
                                    st.session_state.min_neighbors)
            if proof_vec is None:
                st.error("No face detected in the proof snapshot.")
            else:
                sim = cosine(st.session_state.ref_vec, proof_vec)
                percent = float(np.clip(sim, 0.0, 1.0)) * 100.0
                passed = sim >= st.session_state.threshold
                st.session_state.last_result = dict(
                    similarity=sim, percent=percent, threshold=st.session_state.threshold, passed=passed
                )
                # Affiche tout de suite (pas besoin de cliquer ailleurs)
                st.success("Verification computed. See the result panel on the right.")

    # Téléchargements (pas d'écriture serveur)
    if st.session_state.allow_downloads:
        if st.session_state.ref_img is not None:
            ok, buf = cv2.imencode(".png", st.session_state.ref_img)
            st.download_button("⬇️ Download reference (PNG)", data=buf.tobytes(),
                               file_name="reference.png", mime="image/png")
        if st.session_state.proof_img is not None:
            ok, buf = cv2.imencode(".png", st.session_state.proof_img)
            st.download_button("⬇️ Download proof (PNG)", data=buf.tobytes(),
                               file_name="proof.png", mime="image/png")

# ================== Notes ==================
st.markdown(
    """
<div class="card">
  <div class="section-title">Notes</div>
  <ul class="muted">
    <li>L’app utilise la caméra du navigateur via des <i>snapshots</i> (st.camera_input).</li>
    <li>Aucun fichier n’est écrit côté serveur ; vous pouvez télécharger vos images localement si vous cochez l’option.</li>
    <li>Ce PoC illustre le flux et l’UX. Pour la production, on utiliserait des modèles d’empreintes faciales & contrôles anti-usurpation.</li>
  </ul>
</div>
    """,
    unsafe_allow_html=True,
)
