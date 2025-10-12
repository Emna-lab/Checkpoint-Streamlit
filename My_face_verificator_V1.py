# My_face_verificator_V1.py
# ----------------------------------------------------------
# Face Verification (PoC) — Streamlit + OpenCV (Cloud-ready)
# - Caméra via st.camera_input (compatible Streamlit Cloud)
# - Un seul contrôle central pour ON/OFF de la caméra "Proof"
# - Référence : upload OU snapshot
# - Détection Haar (boîtes tracées après capture)
# - UI simple, pédagogique, résultats synchrones
# ----------------------------------------------------------

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# ===================== PAGE & STYLE =====================
st.set_page_config(page_title="BankID • Face Verification (PoC)", page_icon="🏦", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #ffffff; }
      .app-header { text-align:center; margin-top:.5rem; margin-bottom:.25rem; }
      .brand { font-size: 1.85rem; font-weight: 800; letter-spacing:.2px; color:#0f172a;}
      .subtitle { color:#475569; margin-top:.15rem; }
      .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 14px; background:#fff; }
      .section-title { font-weight:700; color:#0f172a; margin:.25rem 0 .5rem 0; }
      .badge { display:inline-block; padding:.25rem .6rem; border-radius:999px; font-weight:700; border:1px solid #e5e7eb;}
      .ok { background:#ecfdf5; color:#065f46; border-color:#a7f3d0;}
      .ko { background:#fef2f2; color:#991b1b; border-color:#fecaca;}
      .metric { font-size:1.6rem; font-weight:800; }
      .muted { color:#64748b; }
      .guide { border: 2px dashed #94a3b8; border-radius: 12px; padding: 8px; text-align:center; color:#64748b; }
      .centered-btns { display:flex; gap:.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===================== ENTÊTE =====================
st.markdown(
    """
    <div class="app-header">
      <div class="brand">BankID • Face Verification (PoC)</div>
      <div class="subtitle">Snapshot based verification with clear consent</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===================== HAAR CASCADE =====================
CASCADE_FILE = "haarcascade_frontalface_default.xml"
local_path = Path(__file__).parent / CASCADE_FILE
opencv_path = Path(cv2.data.haarcascades) / CASCADE_FILE

if local_path.exists():
    CASCADE_PATH = local_path
elif opencv_path.exists():
    CASCADE_PATH = opencv_path
else:
    st.error(
        "❌ Haar cascade not found.\n"
        f"Place {CASCADE_FILE} next to this .py or rely on OpenCV’s default path:\n{opencv_path}"
    )
    st.stop()

face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
if face_cascade.empty():
    st.error("❌ Haar cascade failed to load (file may be corrupted).")
    st.stop()

# ===================== HELPERS =====================
def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert '#RRGGBB' to OpenCV BGR tuple."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return (b, g, r)

def largest_face_bbox(gray: np.ndarray, scale_factor: float, min_neighbors: int):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])

def face_vector_from_bgr(
    bgr: np.ndarray,
    scale_factor: float,
    min_neighbors: int,
    size: int = 128,
) -> Optional[np.ndarray]:
    """Return normalized grayscale vector from the largest detected face (None if no face)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bbox = largest_face_bbox(gray, scale_factor, min_neighbors)
    if bbox is None:
        return None
    x, y, w, h = bbox
    crop = gray[y:y+h, x:x+w]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    vec = crop.astype(np.float32).ravel()
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return None
    return vec / norm

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. With L2-normalized vectors, it's just a dot product."""
    return float(np.dot(a, b))

def bgr_from_file(file) -> Optional[np.ndarray]:
    """Read uploaded/camera_input image to BGR np.ndarray."""
    if file is None:
        return None
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if data is None:
        return None
    buf = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img

def draw_faces(bgr: np.ndarray, color_bgr: Tuple[int, int, int],
               scale_factor: float, min_neighbors: int) -> np.ndarray:
    """Draw rectangles on all detected faces."""
    out = bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), color_bgr, 2)
    return out

def show_bgr_image(bgr: Optional[np.ndarray], caption: str):
    """Display BGR → RGB; compat `use_container_width`/`use_column_width`."""
    if bgr is None:
        return
    img_rgb = bgr[:, :, ::-1]
    try:
        st.image(img_rgb, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img_rgb, caption=caption, use_column_width=True)

# ===================== STATE (defaults) =====================
defaults = {
    # Images & vectors
    "ref_img": None,        # BGR with rectangles (reference)
    "ref_img_raw": None,    # raw BGR (reference, sans rectangles) pour extraction robuste
    "proof_img": None,      # BGR with rectangles (proof)
    "proof_img_raw": None,  # raw BGR (proof, sans rectangles)
    "ref_vec": None,        # reference face vector
    "last_sim": None,       # last similarity

    # Unique camera control for PROOF
    "proof_camera_on": False,

    # Detection params
    "scale_factor": 1.30,
    "min_neighbors": 5,
    "rect_hex": "#2563eb",
    "threshold": 0.86,

    # Consent for saving
    "allow_persist": False,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ===================== INSTRUCTIONS =====================
st.markdown(
    """
<div class="card">
  <div class="section-title">How it works (snapshot & consent)</div>
  <ul class="muted">
    <li>📌 <b>Reference</b>: upload a photo <i>or</i> take a snapshot with your camera.</li>
    <li>📌 <b>Proof</b>: turn the camera ON (single control below), capture a snapshot, then turn it OFF.</li>
    <li>🔒 Images stay in memory (no disk) unless you explicitly enable saving.</li>
  </ul>
</div>
    """,
    unsafe_allow_html=True,
)

# ===================== SIDEBAR (PARAMS) =====================
with st.sidebar:
    st.header("Verification parameters")

    st.session_state.rect_hex = st.color_picker("Rectangle color", st.session_state.rect_hex)
    st.caption("Visual only — color of the detection boxes.")

    st.session_state.scale_factor = st.slider("scaleFactor", 1.05, 1.60, st.session_state.scale_factor, 0.01)
    st.caption("Image scaling step used by the detector. Typical: 1.1–1.4.")

    st.session_state.min_neighbors = st.slider("minNeighbors", 1, 12, st.session_state.min_neighbors, 1)
    st.caption("Higher → fewer false detections, requires a stabler face.")

    st.session_state.threshold = st.slider("Decision threshold (cosine)", 0.50, 0.98, st.session_state.threshold, 0.01)
    st.caption("Similarity ≥ threshold ⇒ PASS. Higher threshold = stricter.")

    st.divider()
    st.header("Security (PoC)")
    st.session_state.allow_persist = st.checkbox("Allow saving images to disk (snapshots/)", value=False)
    st.caption("If unchecked (default), images stay in memory and are discarded on refresh.")

# ===================== LAYOUT =====================
left, right = st.columns([7, 5])

# ---- RIGHT: RESULT ----
with right:
    st.markdown('<div class="section-title">Verification result</div>', unsafe_allow_html=True)
    if st.session_state.last_sim is None:
        st.caption("Capture a reference and a proof, then click **Verify**.")
    else:
        sim = float(st.session_state.last_sim)
        sim_clip = float(np.clip(sim, 0.0, 1.0))
        percent = sim_clip * 100.0
        passed = sim >= st.session_state.threshold

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Similarity**")
            st.markdown(f"<div class='metric'>{sim:.3f}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("**Match (%)**")
            st.markdown(f"<div class='metric'>{percent:.1f}%</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("**Threshold**")
            st.markdown(f"<div class='metric'>{st.session_state.threshold:.2f}</div>", unsafe_allow_html=True)

        st.markdown("**Decision**")
        if passed:
            st.markdown("<span class='badge ok'>✅ PASS</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge ko'>❌ FAIL</span>", unsafe_allow_html=True)

        st.caption("Cosine similarity on 128×128 grayscale face crops. % is clipped to [0–100].")

# ---- LEFT: Reference & Proof capture ----
with left:
    # ===== 1) REFERENCE =====
    st.markdown('<div class="section-title">1) Reference capture</div>', unsafe_allow_html=True)

    # Upload
    ref_upload = st.file_uploader("Upload reference photo", type=["jpg", "jpeg", "png"])
    if ref_upload is not None:
        img_bgr = bgr_from_file(ref_upload)
        vec = face_vector_from_bgr(img_bgr, st.session_state.scale_factor, st.session_state.min_neighbors)
        if vec is None:
            st.error("No face found in the uploaded image.")
        else:
            st.session_state.ref_vec = vec
            st.session_state.ref_img_raw = img_bgr
            st.session_state.ref_img = draw_faces(
                img_bgr, hex_to_bgr(st.session_state.rect_hex),
                st.session_state.scale_factor, st.session_state.min_neighbors
            )
            st.session_state.last_sim = None  # reset
            st.success("Reference saved (from upload).")

    # Camera (simple snapshot)
    cam_ref = st.camera_input("Or take a reference snapshot")
    if cam_ref is not None:
        img_bgr = bgr_from_file(cam_ref)
        vec = face_vector_from_bgr(img_bgr, st.session_state.scale_factor, st.session_state.min_neighbors)
        if vec is None:
            st.error("No face detected in the reference snapshot.")
        else:
            st.session_state.ref_vec = vec
            st.session_state.ref_img_raw = img_bgr
            st.session_state.ref_img = draw_faces(
                img_bgr, hex_to_bgr(st.session_state.rect_hex),
                st.session_state.scale_factor, st.session_state.min_neighbors
            )
            st.session_state.last_sim = None  # reset
            st.success("Reference saved (from camera).")

    # Preview + clear
    show_bgr_image(st.session_state.ref_img, caption="Reference (detected)")
    if st.session_state.ref_img is not None:
        if st.button("🧹 Clear reference"):
            st.session_state.ref_img = None
            st.session_state.ref_img_raw = None
            st.session_state.ref_vec = None
            st.session_state.last_sim = None
            st.success("Reference cleared.")

    st.markdown('---')

    # ===== 2) PROOF =====
    st.markdown('<div class="section-title">2) Proof (single camera control)</div>', unsafe_allow_html=True)

    # Un seul contrôle central pour la caméra Proof : ON/OFF
    cols_ctrl = st.columns(2)
    with cols_ctrl[0]:
        if not st.session_state.proof_camera_on and st.button("▶️ Start proof camera"):
            st.session_state.proof_camera_on = True
    with cols_ctrl[1]:
        if st.session_state.proof_camera_on and st.button("⏹ Stop proof camera"):
            st.session_state.proof_camera_on = False
            st.info("Proof camera turned OFF.")

    # Guide visuel (cadran) — info utilisateur
    st.markdown("<div class='guide'>Center your face, look straight, good lighting. Then capture.</div>", unsafe_allow_html=True)

    proof_snap = None
    if st.session_state.proof_camera_on:
        proof_snap = st.camera_input("Capture a proof snapshot")

    # Si on a capturé : stocker la raw et la version avec rectangles
    if proof_snap is not None:
        img_bgr = bgr_from_file(proof_snap)
        st.session_state.proof_img_raw = img_bgr
        st.session_state.proof_img = draw_faces(
            img_bgr, hex_to_bgr(st.session_state.rect_hex),
            st.session_state.scale_factor, st.session_state.min_neighbors
        )
        st.session_state.last_sim = None  # reset jusqu’à vérification
        st.success("Proof snapshot captured.")

    # Aperçu + Clear (pour retirer l’aperçu proof restant)
    show_bgr_image(st.session_state.proof_img, caption="Proof (detected)")
    if st.session_state.proof_img is not None:
        if st.button("🧹 Clear proof"):
            st.session_state.proof_img = None
            st.session_state.proof_img_raw = None
            st.session_state.last_sim = None
            st.success("Proof cleared.")

    st.markdown('---')

    # ===== 3) VERIFY =====
    st.markdown('<div class="section-title">3) Verify</div>', unsafe_allow_html=True)
    if st.button("✅ Verify identity"):
        if st.session_state.ref_vec is None:
            st.warning("Please set a reference first (upload or camera).")
        elif st.session_state.proof_img_raw is None:
            st.warning("Please take a proof snapshot.")
        else:
            # IMPORTANT : on extrait le vecteur depuis la version "raw" (sans rectangles)
            vec = face_vector_from_bgr(
                st.session_state.proof_img_raw,
                st.session_state.scale_factor,
                st.session_state.min_neighbors,
            )
            if vec is None:
                st.error("No face detected in the proof snapshot.")
            else:
                st.session_state.last_sim = cosine_similarity(st.session_state.ref_vec, vec)
                st.success("Verification computed. See the result panel on the right.")

    # Sauvegarde optionnelle (consentement explicite)
    if st.session_state.allow_persist:
        from datetime import datetime
        save_dir = Path("snapshots"); save_dir.mkdir(exist_ok=True)
        if st.session_state.ref_img is not None:
            cv2.imwrite(str(save_dir / f"reference_{datetime.now():%Y%m%d_%H%M%S}.png"), st.session_state.ref_img)
        if st.session_state.proof_img is not None:
            cv2.imwrite(str(save_dir / f"proof_{datetime.now():%Y%m%d_%H%M%S}.png"), st.session_state.proof_img)

# ===================== NOTES =====================
st.markdown(
    """
<div class="card">
  <div class="section-title">Notes (PoC)</div>
  <ul class="muted">
    <li>This is a didactic baseline (cosine on grayscale crops) to illustrate a verification flow.</li>
    <li>Images are kept in memory only, unless you enable saving in the sidebar.</li>
  </ul>
</div>
    """,
    unsafe_allow_html=True,
)
