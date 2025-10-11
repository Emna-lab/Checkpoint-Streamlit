# bank_face_verify_app.py
# ----------------------------------------------------------
# BankID • Face Verification (Streamlit + OpenCV)
# ----------------------------------------------------------
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# ===================== PAGE & STYLE =====================
st.set_page_config(page_title="BankID • Face Verification", page_icon="🏦", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #ffffff; }
      .app-header { text-align:center; margin-top:.4rem; }
      .brand { font-size: 1.85rem; font-weight: 800; letter-spacing:.3px; color:#0f172a;}
      .subtitle { color:#475569; margin-top:.15rem; }
      .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 14px; background:#fff; }
      .section-title { font-weight:700; color:#0f172a; margin-bottom:.4rem; }
      .badge { display:inline-block; padding:.25rem .6rem; border-radius:999px; font-weight:700; border:1px solid #e5e7eb;}
      .ok { background:#ecfdf5; color:#065f46; border-color:#a7f3d0;}
      .ko { background:#fef2f2; color:#991b1b; border-color:#fecaca;}
      .muted{ color:#64748b; }
      .metric { font-size:1.6rem; font-weight:800; }
      .logo-wrap { display:flex; align-items:center; justify-content:center; gap:.6rem;}
      .logo { width:28px; height:28px; border-radius:6px; background:#0ea5e9; display:inline-block; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===================== EN-TÊTE =====================
st.markdown(
    """
    <div class="app-header">
      <div class="logo-wrap">
        <span class="logo"></span>
        <div class="brand">BankID • Face Verification</div>
      </div>
      <div class="subtitle">Secure customer verification — internal demo</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===================== CHARGER LA CASCADE =====================
CASCADE_FILE = "haarcascade_frontalface_default.xml"
local_path = Path(__file__).parent / CASCADE_FILE
opencv_path = Path(cv2.data.haarcascades) / CASCADE_FILE

if local_path.exists():
    CASCADE_PATH = local_path
elif opencv_path.exists():
    CASCADE_PATH = opencv_path
else:
    st.error(
        f"❌ Cascade file not found: {CASCADE_FILE}\n\n"
        f"Searched:\n- {local_path}\n- {opencv_path}\n\n"
        "Place the XML next to this .py or rely on OpenCV’s default path."
    )
    st.stop()

face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
if face_cascade.empty():
    st.error("❌ Haar cascade failed to load (file may be corrupted).")
    st.stop()

SNAP_DIR = (Path(__file__).parent / "snapshots")
SNAP_DIR.mkdir(exist_ok=True)

# ===================== FONCTIONS UTILES =====================
def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convertit '#RRGGBB' en (B,G,R) pour OpenCV."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return (b, g, r)

def detect_largest_face_bbox(gray: np.ndarray, scale_factor: float, min_neighbors: int):
    """Détecte tous les visages et retourne la bbox du plus grand (ou None)."""
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])  # bbox avec l'aire max

def face_vector_from_bgr(
    bgr: np.ndarray,
    scale_factor: float,
    min_neighbors: int,
    size: int = 128,
) -> Optional[np.ndarray]:
    """
    Extrait un vecteur facial simple :
      1) gris 2) recadrage visage 3) resize (128x128) 4) aplatissement 5) normalisation L2.
    Retourne None si aucun visage détecté.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bbox = detect_largest_face_bbox(gray, scale_factor, min_neighbors)
    if bbox is None:
        return None
    x, y, w, h = bbox
    crop = gray[y : y + h, x : x + w]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    vec = crop.astype(np.float32).ravel()
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return None
    return vec / norm  # <- vecteur normalisé (norme=1)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Similarité cosinus = np.dot(a,b) quand a et b sont normalisés (norme=1).
    Valeur entre -1 et 1 ; on la "clippe" à [0,1] pour l'affichage en %.
    """
    return float(np.dot(a, b))

def draw_rects(bgr: np.ndarray, color_bgr: Tuple[int, int, int], scale_factor: float, min_neighbors: int) -> np.ndarray:
    """Dessine des rectangles autour de tous les visages détectés."""
    out = bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), color_bgr, 2)
    return out

# ===================== ÉTAT (SESSION) =====================
defaults = {
    "running": False,          # caméra en cours ?
    "last_frame": None,        # dernier frame BGR
    "ref_vec": None,           # vecteur du visage de référence
    "ref_img": None,           # image de référence (pour affichage)
    "scale_factor": 1.30,      # paramètre detectMultiScale
    "min_neighbors": 5,        # paramètre detectMultiScale
    "rect_hex": "#2563eb",     # bleu
    "threshold": 0.86,         # seuil de décision cosine
    "last_verify": None,       # dernier résultat de vérification (dict)
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ===================== INSTRUCTIONS VISIBLES =====================
st.markdown(
    """
<div class="card">
  <div class="section-title">Operating procedure</div>
  <ul class="muted">
    <li><b>1)</b> Provide a <b>Reference Face</b> (upload a photo or capture from camera).</li>
    <li><b>2)</b> Start the camera. Faces are detected in real time (blue rectangles).</li>
    <li><b>3)</b> Click <b>Verify</b> to compare the current face with the reference.</li>
  </ul>
</div>
    """,
    unsafe_allow_html=True,
)

# ===================== SIDEBAR (PARAMÈTRES + EXPLICATIONS) =====================
with st.sidebar:
    st.header("Configuration")
    st.session_state.rect_hex = st.color_picker("Rectangle color", st.session_state.rect_hex)
    st.caption("Couleur utilisée pour encadrer les visages détectés.")

    st.session_state.scale_factor = st.slider("scaleFactor", 1.05, 1.60, st.session_state.scale_factor, 0.01)
    st.caption("> 1.0. Plus grand = balayage plus grossier (plus rapide, moins précis). 1.1–1.4 raisonnable.")

    st.session_state.min_neighbors = st.slider("minNeighbors", 1, 12, st.session_state.min_neighbors, 1)
    st.caption("Plus haut = moins de faux positifs, mais nécessite des visages plus nets/continus.")

    st.session_state.threshold = st.slider("Verification threshold (cosine)", 0.50, 0.98, st.session_state.threshold, 0.01)
    st.caption("Seuil de décision. Similarité ≥ seuil ⇒ PASS. Plus haut = plus strict.")

# ===================== LAYOUT =====================
left, right = st.columns([7, 5])

# ===== COLONNE GAUCHE : Live + Verify =====
with left:
    st.markdown('<div class="section-title">Live Detector</div>', unsafe_allow_html=True)
    live_area = st.empty()

    # Boutons principaux
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        if st.button("▶️ Start", use_container_width=True):
            st.session_state.running = True
    with c2:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.running = False
    with c3:
        # Sauvegarde du dernier frame (s'il existe)
        if st.button("💾 Save snapshot", use_container_width=True):
            frame = st.session_state.last_frame
            if frame is None:
                st.warning("No frame yet.")
            else:
                ts = time.strftime("%Y%m%d-%H%M%S")
                out = (SNAP_DIR / f"snapshot_{ts}.png")
                cv2.imwrite(str(out), frame)
                ok, buf = cv2.imencode(".png", frame)
                st.success(f"Saved `{out.name}`")
                if ok:
                    st.download_button("⬇️ Download", data=buf.tobytes(), file_name=out.name, mime="image/png")
    with c4:
        # Vérification IMMÉDIATE (pas de flag caché) → on calcule et on stocke le résultat
        if st.button("✅ Verify", use_container_width=True):
            # On calcule tout de suite (dans CE run) et on stocke un dict résultat
            if st.session_state.ref_vec is None:
                st.session_state.last_verify = {"status": "no_ref"}
            else:
                frame = st.session_state.last_frame
                if frame is None:
                    st.session_state.last_verify = {"status": "no_frame"}
                else:
                    curr_vec = face_vector_from_bgr(
                        frame, st.session_state.scale_factor, st.session_state.min_neighbors
                    )
                    if curr_vec is None:
                        st.session_state.last_verify = {"status": "no_face"}
                    else:
                        sim = cosine_similarity(st.session_state.ref_vec, curr_vec)
                        # on clippe à [0,1] pour un affichage % lisible
                        sim_clipped = float(np.clip(sim, 0.0, 1.0))
                        percent = sim_clipped * 100.0
                        passed = sim >= st.session_state.threshold  # décision NON clipée
                        st.session_state.last_verify = {
                            "status": "ok",
                            "similarity": sim,
                            "percent": percent,
                            "passed": passed,
                            "threshold": st.session_state.threshold,
                        }

# ===== COLONNE DROITE : Référence & Résultat =====
with right:
    st.markdown('<div class="section-title">Reference Face</div>', unsafe_allow_html=True)
    rcols = st.columns([1, 1])

    # Upload image de référence
    with rcols[0]:
        uploaded = st.file_uploader("Upload reference photo", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            buf = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            vec = face_vector_from_bgr(img_bgr, st.session_state.scale_factor, st.session_state.min_neighbors)
            if vec is None:
                st.error("No face found in the uploaded image.")
            else:
                st.session_state.ref_vec = vec
                st.session_state.ref_img = draw_rects(
                    img_bgr, hex_to_bgr(st.session_state.rect_hex),
                    st.session_state.scale_factor, st.session_state.min_neighbors
                )
                st.success("Reference saved (from upload).")

    # Capturer référence depuis la caméra
    with rcols[1]:
        if st.button("📸 Capture from camera", use_container_width=True):
            frame = st.session_state.last_frame
            if frame is None:
                st.warning("Start the camera first.")
            else:
                vec = face_vector_from_bgr(frame, st.session_state.scale_factor, st.session_state.min_neighbors)
                if vec is None:
                    st.error("No face detected in current frame.")
                else:
                    st.session_state.ref_vec = vec
                    st.session_state.ref_img = draw_rects(
                        frame, hex_to_bgr(st.session_state.rect_hex),
                        st.session_state.scale_factor, st.session_state.min_neighbors
                    )
                    st.success("Reference saved (from camera).")

    # --- Clear reference button (add this right after the rcols block, before the preview) ---
    if st.button("🧹 Clear reference", use_container_width=True):
        st.session_state.ref_vec = None
        st.session_state.ref_img = None
        st.session_state.last_verify = None  # on efface aussi le dernier résultat
        st.success("Reference cleared.")

    # Aperçu de la référence
    if st.session_state.ref_img is not None:
        st.image(cv2.cvtColor(st.session_state.ref_img, cv2.COLOR_BGR2RGB),
                 caption="Reference", use_container_width=True)
    else:
        st.info("Provide a reference face to enable verification.")

    st.markdown("---")
    st.markdown('<div class="section-title">Verification Result</div>', unsafe_allow_html=True)

    # Affichage PERSISTANT du dernier résultat
    res = st.session_state.last_verify
    if res is None:
        st.caption("Click **Verify** to compute a decision.")
    else:
        status = res.get("status")
        if status == "no_ref":
            st.warning("Please set a reference face first.")
        elif status == "no_frame":
            st.warning("Start the camera and ensure a face is visible.")
        elif status == "no_face":
            st.error("No face detected in the current frame.")
        elif status == "ok":
            sim = res["similarity"]
            percent = res["percent"]
            threshold = res["threshold"]
            passed = res["passed"]

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown("**Similarity**")
                st.markdown(f"<div class='metric'>{sim:.3f}</div>", unsafe_allow_html=True)
            with m2:
                st.markdown("**Match (%)**")
                st.markdown(f"<div class='metric'>{percent:.1f}%</div>", unsafe_allow_html=True)
            with m3:
                st.markdown("**Threshold**")
                st.markdown(f"<div class='metric'>{threshold:.2f}</div>", unsafe_allow_html=True)

            st.markdown("**Decision**")
            if passed:
                st.markdown("<span class='badge ok'>✅ PASS</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='badge ko'>❌ FAIL</span>", unsafe_allow_html=True)

            st.caption("Similarity = cosine on normalized 128×128 grayscale face crops. % is clipped to [0–100].")

# ===================== BOUCLE CAMÉRA =====================
def run_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        live_area.error("❌ Cannot open webcam.")
        return
    try:
        while st.session_state.running:
            ok, frame = cap.read()
            if not ok:
                live_area.error("⛔️ Cannot read from webcam.")
                break
            st.session_state.last_frame = frame.copy()
            frame_disp = draw_rects(
                frame,
                hex_to_bgr(st.session_state.rect_hex),
                st.session_state.scale_factor,
                st.session_state.min_neighbors,
            )
            live_area.image(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB),
                            channels="RGB", use_container_width=True)
            time.sleep(0.01)
    finally:
        cap.release()

if st.session_state.running:
    run_camera()
