# bank_face_verify_poc.py
# ----------------------------------------------------------
# Face Verification (PoC) — Streamlit + OpenCV (secure snapshot flow)
# ----------------------------------------------------------
from __future__ import annotations

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
      .app-header { text-align:center; margin-top:.4rem; }
      .brand { font-size: 1.85rem; font-weight: 800; letter-spacing:.3px; color:#0f172a;}
      .subtitle { color:#475569; margin-top:.15rem; }
      .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 14px; background:#fff; }
      .section-title { font-weight:700; color:#0f172a; margin-bottom:.5rem; }
      .badge { display:inline-block; padding:.25rem .6rem; border-radius:999px; font-weight:700; border:1px solid #e5e7eb;}
      .ok { background:#ecfdf5; color:#065f46; border-color:#a7f3d0;}
      .ko { background:#fef2f2; color:#991b1b; border-color:#fecaca;}
      .metric { font-size:1.6rem; font-weight:800; }
      .muted { color:#64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===================== ENTÊTE =====================
st.markdown(
    """
    <div class="app-header">
      <div class="brand">BankID • Face Verification (PoC)</div>
      <div class="subtitle">Secure snapshot-based verification — no continuous streaming</div>
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
    crop = gray[y : y + h, x : x + w]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    vec = crop.astype(np.float32).ravel()
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return None
    return vec / norm

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity (a and b must be L2-normalized)."""
    return float(np.dot(a, b))

def bgr_from_file(file) -> Optional[np.ndarray]:
    """Read uploaded/camera-input image to BGR np.ndarray."""
    if file is None:
        return None
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if data is None:
        return None
    buf = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img

def draw_faces(bgr: np.ndarray, color_bgr: Tuple[int, int, int], scale_factor: float, min_neighbors: int) -> np.ndarray:
    out = bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), color_bgr, 2)
    return out

# ===================== STATE =====================
defaults = {
    "ref_vec": None,
    "ref_img": None,
    "last_result": None,   # dict: similarity, percent, threshold, passed
    "scale_factor": 1.30,
    "min_neighbors": 5,
    "rect_hex": "#2563eb",
    "threshold": 0.86,
    "allow_persist": False,   # PoC: off by default (no disk write)
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
    <li>📌 <b>Proof</b>: take a snapshot with your camera for verification.</li>
    <li>🔒 PoC stores images in memory only (no disk), unless you explicitly enable saving.</li>
  </ul>
</div>
    """,
    unsafe_allow_html=True,
)

# ===================== SIDEBAR (PARAMS & SECURITY NOTES) =====================
with st.sidebar:
    st.header("Verification parameters")
    st.session_state.rect_hex = st.color_picker("Rectangle color", st.session_state.rect_hex)
    st.caption("Purely visual — color of the detection boxes.")

    st.session_state.scale_factor = st.slider("scaleFactor", 1.05, 1.60, st.session_state.scale_factor, 0.01)
    st.caption(">1.0. Larger = faster/rougher scanning; typical 1.1–1.4.")

    st.session_state.min_neighbors = st.slider("minNeighbors", 1, 12, st.session_state.min_neighbors, 1)
    st.caption("Higher = fewer false positives, needs more stable face.")

    st.session_state.threshold = st.slider("Decision threshold (cosine)", 0.50, 0.98, st.session_state.threshold, 0.01)
    st.caption("Similarity ≥ threshold ⇒ PASS. Higher threshold = stricter decision.")

    st.divider()
    st.header("Security (PoC)")
    st.session_state.allow_persist = st.checkbox("Allow saving images to disk (snapshots/)", value=False)
    st.caption("If disabled (default), images stay in memory and are discarded on refresh.")

# ===================== LAYOUT =====================
left, right = st.columns([6, 6])

# ---- RIGHT: RESULT (persistent) ----
with right:
    st.markdown('<div class="section-title">Verification result</div>', unsafe_allow_html=True)
    res = st.session_state.last_result
    if res is None:
        st.caption("Capture a reference and a proof, then click **Verify**.")
    else:
        sim = res["similarity"]
        percent = res["percent"]
        threshold = res["threshold"]
        passed = res["passed"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Similarity**")
            st.markdown(f"<div class='metric'>{sim:.3f}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("**Match (%)**")
            st.markdown(f"<div class='metric'>{percent:.1f}%</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("**Threshold**")
            st.markdown(f"<div class='metric'>{threshold:.2f}</div>", unsafe_allow_html=True)

        st.markdown("**Decision**")
        if passed:
            st.markdown("<span class='badge ok'>✅ PASS</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge ko'>❌ FAIL</span>", unsafe_allow_html=True)

        st.caption("Cosine similarity on 128×128 grayscale face crops. % is clipped to [0–100].")

# ---- LEFT: Reference & Proof capture ----
with left:
    st.markdown('<div class="section-title">1) Reference capture</div>', unsafe_allow_html=True)
    col_ref = st.columns([1, 1])

    # Upload reference
    with col_ref[0]:
        ref_upload = st.file_uploader("Upload reference photo", type=["jpg", "jpeg", "png"])
        if ref_upload is not None:
            img_bgr = bgr_from_file(ref_upload)
            vec = face_vector_from_bgr(img_bgr, st.session_state.scale_factor, st.session_state.min_neighbors)
            if vec is None:
                st.error("No face found in the uploaded image.")
            else:
                st.session_state.ref_vec = vec
                st.session_state.ref_img = draw_faces(
                    img_bgr, hex_to_bgr(st.session_state.rect_hex),
                    st.session_state.scale_factor, st.session_state.min_neighbors
                )
                st.success("Reference saved (from upload).")

    # Camera reference
    with col_ref[1]:
        ref_cam = st.camera_input("Or take a reference snapshot")
        if ref_cam is not None:
            img_bgr = bgr_from_file(ref_cam)
            vec = face_vector_from_bgr(img_bgr, st.session_state.scale_factor, st.session_state.min_neighbors)
            if vec is None:
                st.error("No face detected in the reference snapshot.")
            else:
                st.session_state.ref_vec = vec
                st.session_state.ref_img = draw_faces(
                    img_bgr, hex_to_bgr(st.session_state.rect_hex),
                    st.session_state.scale_factor, st.session_state.min_neighbors
                )
                st.success("Reference saved (from camera).")

    # Reference preview + clear
    if st.session_state.ref_img is not None:
        st.image(cv2.cvtColor(st.session_state.ref_img, cv2.COLOR_BGR2RGB),
                 caption="Reference", use_container_width=True)
        if st.button("🧹 Clear reference"):
            st.session_state.ref_vec = None
            st.session_state.ref_img = None
            st.session_state.last_result = None
            st.success("Reference cleared.")

    st.markdown('---')
    st.markdown('<div class="section-title">2) Proof snapshot</div>', unsafe_allow_html=True)
    proof_cam = st.camera_input("Take a proof snapshot for verification")

    st.markdown('---')
    st.markdown('<div class="section-title">3) Verify</div>', unsafe_allow_html=True)
    if st.button("✅ Verify identity"):
        if st.session_state.ref_vec is None:
            st.warning("Please set a reference first (upload or camera).")
        elif proof_cam is None:
            st.warning("Please take a proof snapshot.")
        else:
            proof_bgr = bgr_from_file(proof_cam)
            curr_vec = face_vector_from_bgr(proof_bgr, st.session_state.scale_factor, st.session_state.min_neighbors)
            if curr_vec is None:
                st.error("No face detected in the proof snapshot.")
            else:
                sim = cosine_similarity(st.session_state.ref_vec, curr_vec)
                sim_clip = float(np.clip(sim, 0.0, 1.0))
                percent = sim_clip * 100.0
                passed = sim >= st.session_state.threshold

                st.session_state.last_result = {
                    "similarity": sim,
                    "percent": percent,
                    "threshold": st.session_state.threshold,
                    "passed": passed,
                }
                st.success("Verification computed. See the result panel on the right.")

    # Optional disk save (explicit consent)
    if st.session_state.allow_persist:
        if st.session_state.ref_img is not None:
            ok, buf = cv2.imencode(".png", st.session_state.ref_img)
            st.download_button("⬇️ Download reference (PNG)", data=buf.tobytes(),
                               file_name="reference.png", mime="image/png")
        if proof_cam is not None:
            proof_bgr = bgr_from_file(proof_cam)
            ok, buf = cv2.imencode(".png", proof_bgr)
            st.download_button("⬇️ Download proof (PNG)", data=buf.tobytes(),
                               file_name="proof.png", mime="image/png")

# ===================== NOTES =====================
st.markdown(
    """
<div class="card">
  <div class="section-title">Security & PoC scope</div>
  <ul class="muted">
    <li>No continuous streaming — browser prompts for camera permission per snapshot.</li>
    <li>No disk write by default (toggle in sidebar if you need downloads).</li>
    <li>For production KYC/IDV: use robust face embeddings (e.g., InsightFace) and liveness detection.</li>
  </ul>
</div>
    """,
    unsafe_allow_html=True,
)
