# My_face_verificator_V1.py
# ----------------------------------------------------------
# Face Verification (PoC) — Streamlit + OpenCV (Cloud & Local)
# - Deux caméras indépendantes (Référence & Proof) avec ON/OFF
# - Bouton global ON/OFF pour les deux
# - Cloud: st.camera_input (cadran-guide statique à côté de la caméra)
# - Local (optionnel): petite fenêtre OpenCV de guide (preview) si désiré
# - Snapshot affiché à droite (pas sous la caméra)
# - Actions Verify / Clear sans double-clic (MAJ immédiate via session_state)
# ----------------------------------------------------------

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# ============ PAGE & STYLE ============
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
      .guide { border: 2px dashed #94a3b8; border-radius: 12px; padding: 10px; text-align:center; color:#64748b; height: 100%; }
      .centered-btns { display:flex; gap:.5rem; }
      .cam-box {border:1px solid #e5e7eb; border-radius:10px; padding:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============ ENTÊTE ============
st.markdown(
    """
    <div class="app-header">
      <div class="brand">BankID • Face Verification (PoC)</div>
      <div class="subtitle">Snapshot based verification with clear consent</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============ HAAR CASCADE ============
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

# ============ HELPERS ============
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
    except TypeError:  # older Streamlit
        st.image(img_rgb, caption=caption, use_column_width=True)

# ============ STATE (défauts) ============
defaults = {
    # images & vecteurs
    "ref_img": None,
    "ref_img_raw": None,
    "ref_vec": None,

    "proof_img": None,
    "proof_img_raw": None,

    # ON/OFF caméras (cloud)
    "ref_camera_on": False,
    "proof_camera_on": False,

    # paramètres détection & décision
    "scale_factor": 1.30,
    "min_neighbors": 5,
    "rect_hex": "#2563eb",
    "threshold": 0.86,

    # consentement pour enregistrer (optionnel)
    "allow_persist": False,

    # dernier résultat
    "last_sim": None,

    # mode local (preview OpenCV facultatif)
    "local_preview": False,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ============ INSTRUCTIONS ============
st.markdown(
    """
<div class="card">
  <div class="section-title">How it works</div>
  <ul class="muted">
    <li>📌 <b>Reference</b>: start the camera or upload a photo, then capture the snapshot.</li>
    <li>📌 <b>Proof</b>: start the camera, capture a snapshot, then you can stop the camera.</li>
    <li>🎯 The guidance frame helps you center your face before taking the photo.</li>
    <li>🔒 Images stay in memory (no disk) unless you enable saving (sidebar).</li>
  </ul>
</div>
    """,
    unsafe_allow_html=True,
)

# ============ SIDEBAR (PARAMS) ============
with st.sidebar:
    st.header("Verification parameters")

    st.session_state.rect_hex = st.color_picker("Rectangle color", st.session_state.rect_hex)
    st.caption("Visual only — color of detection boxes.")

    st.session_state.scale_factor = st.slider("scaleFactor", 1.05, 1.60, st.session_state.scale_factor, 0.01)
    st.caption("Image scaling step used by the detector. Typical: 1.1–1.4.")

    st.session_state.min_neighbors = st.slider("minNeighbors", 1, 12, st.session_state.min_neighbors, 1)
    st.caption("Higher → fewer false detections, requires a stabler face.")

    st.session_state.threshold = st.slider("Decision threshold (cosine)", 0.50, 0.98, st.session_state.threshold, 0.01)
    st.caption("Similarity ≥ threshold ⇒ PASS. Higher threshold = stricter.")

    st.divider()
    st.header("Capture controls")
    # Bouton global : ON/OFF les 2 caméras
    colg1, colg2 = st.columns(2)
    with colg1:
        if st.button("Start BOTH"):
            st.session_state.ref_camera_on = True
            st.session_state.proof_camera_on = True
    with colg2:
        if st.button("Stop BOTH"):
            st.session_state.ref_camera_on = False
            st.session_state.proof_camera_on = False

    st.divider()
    st.header("Security (PoC)")
    st.session_state.allow_persist = st.checkbox("Allow saving images to disk (snapshots/)", value=False)
    st.caption("If unchecked, images stay in memory and are discarded on refresh.")

    st.divider()
    st.header("Local preview (Optional)")
    st.session_state.local_preview = st.checkbox("OpenCV guide window (local run only)", value=False)
    st.caption("Works on local PyCharm only (not on Streamlit Cloud).")

# ============ RESULT PANEL (droite) ============
right_col, = st.columns([1])
with right_col:
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

st.markdown("---")

# ============ REFERENCE (caméra à gauche, snapshot à droite) ============
st.markdown('<div class="section-title">1) Reference</div>', unsafe_allow_html=True)
ref_cam_col, ref_snap_col = st.columns([2, 1])

with ref_cam_col:
    st.markdown("**Camera control (Reference)**")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        if not st.session_state.ref_camera_on and st.button("▶️ Start REF"):
            st.session_state.ref_camera_on = True
    with cc2:
        if st.session_state.ref_camera_on and st.button("⏹ Stop REF"):
            st.session_state.ref_camera_on = False
    with cc3:
        if st.button("🧹 Clear REF"):
            st.session_state.ref_img = None
            st.session_state.ref_img_raw = None
            st.session_state.ref_vec = None
            st.session_state.last_sim = None  # reset immédiat
            st.success("Reference cleared.")

    # Cloud-safe camera_input
    cam_ref_file = None
    guide_col1, guide_col2 = st.columns([3, 2])
    with guide_col1:
        st.markdown('<div class="cam-box">', unsafe_allow_html=True)
        if st.session_state.ref_camera_on:
            cam_ref_file = st.camera_input("Reference camera")
        else:
            st.info("Reference camera is OFF")
        st.markdown('</div>', unsafe_allow_html=True)
    with guide_col2:
        st.markdown("<div class='guide'>Center your face in this frame, then capture.</div>", unsafe_allow_html=True)

    # Si snapshot pris, on enregistre immédiatement
    if cam_ref_file is not None:
        img_bgr = bgr_from_file(cam_ref_file)
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
            st.session_state.last_sim = None  # reset immédiat
            st.success("Reference saved.")

with ref_snap_col:
    show_bgr_image(st.session_state.ref_img, caption="Reference (detected)")

# ============ PROOF (caméra à gauche, snapshot à droite) ============
st.markdown('<div class="section-title">2) Proof</div>', unsafe_allow_html=True)
proof_cam_col, proof_snap_col = st.columns([2, 1])

with proof_cam_col:
    st.markdown("**Camera control (Proof)**")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        if not st.session_state.proof_camera_on and st.button("▶️ Start PROOF"):
            st.session_state.proof_camera_on = True
    with pc2:
        if st.session_state.proof_camera_on and st.button("⏹ Stop PROOF"):
            st.session_state.proof_camera_on = False
    with pc3:
        if st.button("🧹 Clear PROOF"):
            st.session_state.proof_img = None
            st.session_state.proof_img_raw = None
            st.session_state.last_sim = None
            st.success("Proof cleared.")

    cam_proof_file = None
    guide2_col1, guide2_col2 = st.columns([3, 2])
    with guide2_col1:
        st.markdown('<div class="cam-box">', unsafe_allow_html=True)
        if st.session_state.proof_camera_on:
            cam_proof_file = st.camera_input("Proof camera")
        else:
            st.info("Proof camera is OFF")
        st.markdown('</div>', unsafe_allow_html=True)
    with guide2_col2:
        st.markdown("<div class='guide'>Align your face inside this frame, then capture.</div>", unsafe_allow_html=True)

    if cam_proof_file is not None:
        img_bgr = bgr_from_file(cam_proof_file)
        st.session_state.proof_img_raw = img_bgr
        st.session_state.proof_img = draw_faces(
            img_bgr, hex_to_bgr(st.session_state.rect_hex),
            st.session_state.scale_factor, st.session_state.min_neighbors
        )
        st.session_state.last_sim = None
        st.success("Proof snapshot captured.")

with proof_snap_col:
    show_bgr_image(st.session_state.proof_img, caption="Proof (detected)")

# ============ VERIFY ============
st.markdown('<div class="section-title">3) Verify</div>', unsafe_allow_html=True)
if st.button("✅ Verify identity"):
    if st.session_state.ref_vec is None:
        st.warning("Please set a reference first (upload or camera).")
    elif st.session_state.proof_img_raw is None:
        st.warning("Please take a proof snapshot.")
    else:
        vec = face_vector_from_bgr(
            st.session_state.proof_img_raw,
            st.session_state.scale_factor,
            st.session_state.min_neighbors,
        )
        if vec is None:
            st.error("No face detected in the proof snapshot.")
        else:
            st.session_state.last_sim = cosine_similarity(st.session_state.ref_vec, vec)
            st.success("Verification computed. See the result panel above.")

# ============ LOCAL PREVIEW (OPTIONNEL, PyCharm uniquement) ============
# Petite fenêtre OpenCV de guide (non utilisée sur Streamlit Cloud)
if st.session_state.local_preview:
    if st.button("Open local guide window (ESC to close)"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Unable to open local webcam.")
        else:
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    h, w = frame.shape[:2]
                    # simple guide rectangle centré
                    gh, gw = int(h * 0.6), int(w * 0.6)
                    y1 = (h - gh) // 2; y2 = y1 + gh
                    x1 = (w - gw) // 2; x2 = x1 + gw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 200, 255), 2)
                    cv2.imshow("Local guide — press ESC to close", frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()

# ============ NOTES ============
st.markdown(
    """
<div class="card">
  <div class="section-title">Notes (PoC)</div>
  <ul class="muted">
    <li>This is a didactic baseline to illustrate a verification flow.</li>
    <li>Images are kept in memory only, unless you enable saving in the sidebar.</li>
  </ul>
</div>
    """,
    unsafe_allow_html=True,
)
