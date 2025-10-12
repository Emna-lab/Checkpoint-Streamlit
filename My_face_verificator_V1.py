# My_face_verificator_V1.py
# -----------------------------------------------------------
# Face Verification (PoC) — Streamlit
# - Activation/Désactivation (avec libération mémoire)
# - Capture par caméra OU upload de fichier
# - Détection visage (Haar), crop du plus grand visage
# - Descripteur simple (histogramme), Similarité cosinus
# - Helpers robustes pour éviter les crashs cv2.cvtColor
# -----------------------------------------------------------

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st


# =========================
# Helpers robustes d'images
# =========================
def bgr_from_file(file):
    """
    Lit un fichier Streamlit (uploader ou camera_input) et renvoie une image BGR (np.ndarray).
    Tente cv2.imdecode, sinon fallback via PIL -> RGB -> BGR.
    Retourne None si échec.
    """
    if file is None:
        return None

    # UploadedFile a .getvalue()
    data = None
    if hasattr(file, "getvalue"):
        data = file.getvalue()
    else:
        try:
            data = file.read()
        except Exception:
            data = None

    if not data:
        return None

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR si succès
    if img is not None:
        return img

    # Fallback PIL (quelques cas de metadata caméra côté cloud)
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        img_rgb = np.array(pil)              # RGB
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def show_bgr_image(bgr: np.ndarray, *, caption: str = "", use_container_width: bool = True):
    """
    Affiche une image BGR dans Streamlit sans planter si elle est None ou malformée.
    """
    if bgr is None:
        st.warning("Image non disponible.")
        return
    if not isinstance(bgr, np.ndarray):
        st.warning("Format d’image inattendu (non-numpy).")
        return

    if bgr.ndim == 2:  # niveaux de gris
        st.image(bgr, caption=caption, use_container_width=use_container_width)
    elif bgr.ndim == 3 and bgr.shape[2] == 3:
        st.image(bgr[:, :, ::-1], caption=caption, use_container_width=use_container_width)  # BGR→RGB
    else:
        st.warning("Dimensions d’image non supportées.")


# ==================================
# Détection visage & description (PoC)
# ==================================
def get_haar_cascade():
    """
    Récupère le classifieur Haar frontal face depuis cv2.data.haarcascades.
    (pas besoin de fichier local)
    """
    xml = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(xml)
    if cascade.empty():
        raise RuntimeError("Impossible de charger le Haar cascade.")
    return cascade


def crop_largest_face(bgr, cascade, scaleFactor=1.1, minNeighbors=5):
    """
    Détecte les visages, retourne le crop du plus grand visage, ou None si aucun visage.
    """
    if bgr is None:
        return None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    if len(faces) == 0:
        return None

    # Garder le plus grand visage
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return bgr[y:y + h, x:x + w]


def make_descriptor(bgr_face, bins=64):
    """
    Crée un descripteur simple à partir d’un visage BGR :
    - Conversion en HSV
    - Histogramme H et S concaténés (normalisé)
    Retourne un vecteur 1D (np.float32) ou None si image invalide.
    """
    if bgr_face is None:
        return None

    hsv = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])   # H ∈ [0,180]
    s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])   # S ∈ [0,256]
    feat = np.concatenate([h.flatten(), s.flatten()]).astype(np.float32)
    norm = np.linalg.norm(feat) + 1e-8
    return feat / norm


def cosine_similarity(a, b):
    """
    Similarité cosinus entre deux vecteurs (a·b / ||a|| ||b||).
    Retourne un float ∈ [-1, 1]. Ici on s’attend à [0,1] car descripteurs normalisés.
    """
    if a is None or b is None:
        return None
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


# ========================
# Init de l'état applicatif
# ========================
def init_state():
    st.session_state.setdefault("active", False)       # app activée/désactivée
    st.session_state.setdefault("ref_img", None)       # BGR
    st.session_state.setdefault("ref_vec", None)       # descriptor
    st.session_state.setdefault("proof_img", None)     # BGR
    st.session_state.setdefault("proof_vec", None)     # descriptor


# =========
# Interface
# =========
st.set_page_config(page_title="Face Verification (PoC)", page_icon="🛡️", layout="wide")
init_state()

st.markdown(
    """
    <h1 style="text-align:center; margin-top:0">🛡️ Face Verification — PoC</h1>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Cette démonstration vérifie si **la personne capturée** correspond à une **référence**. "
    "Elle utilise un détecteur Haar (crop du plus grand visage) et un descripteur simple (histogrammes HSV) "
    "puis compare avec une **similarité cosinus**."
)

# -------------
# Sidebar: réglages
# -------------
with st.sidebar:
    st.header("⚙️ Réglages")

    # Activation / Désactivation (revient à libérer l'état)
    col_on, col_off = st.columns(2)
    with col_on:
        if st.button("✅ Activer", use_container_width=True):
            st.session_state.active = True
            st.toast("Module activé", icon="✅")
    with col_off:
        if st.button("🛑 Désactiver", use_container_width=True):
            # remise à zéro "soft"
            st.session_state.active = False
            st.session_state.ref_img = None
            st.session_state.ref_vec = None
            st.session_state.proof_img = None
            st.session_state.proof_vec = None
            st.toast("Module désactivé", icon="🛑")

    st.divider()
    st.caption("Détection visage (Haar)")
    scaleFactor = st.slider("scaleFactor", 1.05, 1.5, 1.15, 0.01)
    minNeighbors = st.slider("minNeighbors", 3, 12, 5, 1)

    st.divider()
    st.caption("Descripteur")
    bins = st.slider("Histogram bins (H & S)", 16, 128, 64, 8)

    st.divider()
    st.caption("Décision")
    threshold = st.slider("Seuil d'acceptation (%)", 50, 95, 80, 1)

    st.divider()
    # Clear reference
    if st.button("🧹 Clear reference", use_container_width=True):
        st.session_state.ref_img = None
        st.session_state.ref_vec = None
        st.toast("Référence supprimée.", icon="🧹")

# Stop net si désactivée
if not st.session_state.active:
    st.info("Le module est **désactivé**. Cliquez sur **Activer** dans la sidebar.")
    st.stop()

# Cascade prêt
try:
    CASCADE = get_haar_cascade()
except Exception as e:
    st.error(f"Erreur cascade Haar : {e}")
    st.stop()

# =========================
# Zone Référence et Preuve
# =========================
c_ref, c_proof = st.columns(2, gap="large")

# ----- Référence -----
with c_ref:
    st.subheader("🧾 Référence (photo connue)")
    st.write("Charge une image ou prends une photo de la **personne de référence**.")

    ref_upl = st.file_uploader("Upload image (référence)", type=["png", "jpg", "jpeg"], key="ref_upl")
    ref_cam = st.camera_input("Ou capture caméra (référence)", key="ref_cam")

    new_ref = ref_cam if ref_cam is not None else ref_upl
    if new_ref is not None:
        img_bgr = bgr_from_file(new_ref)
        face = crop_largest_face(img_bgr, CASCADE, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        if face is None:
            st.warning("Aucun visage détecté dans la référence.")
        else:
            st.session_state.ref_img = face
            st.session_state.ref_vec = make_descriptor(face, bins=bins)

    show_bgr_image(st.session_state.ref_img, caption="Référence (visage recadré)")

# ----- Preuve -----
with c_proof:
    st.subheader("🧑‍💻 Preuve (vérification)")
    st.write("Charge une image ou prends une photo à **vérifier**.")

    proof_upl = st.file_uploader("Upload image (preuve)", type=["png", "jpg", "jpeg"], key="proof_upl")
    proof_cam = st.camera_input("Ou capture caméra (preuve)", key="proof_cam")

    new_proof = proof_cam if proof_cam is not None else proof_upl
    if new_proof is not None:
        img_bgr = bgr_from_file(new_proof)
        face = crop_largest_face(img_bgr, CASCADE, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        if face is None:
            st.warning("Aucun visage détecté dans la preuve.")
        else:
            st.session_state.proof_img = face
            st.session_state.proof_vec = make_descriptor(face, bins=bins)

    show_bgr_image(st.session_state.proof_img, caption="Preuve (visage recadré)")

st.divider()

# ===========
# Vérification
# ===========
left, mid, right = st.columns([1, 1, 2])
with mid:
    verify_clicked = st.button("🔐 Vérifier l'identité", use_container_width=True)

if verify_clicked:
    ref_vec = st.session_state.ref_vec
    proof_vec = st.session_state.proof_vec

    if ref_vec is None:
        st.error("Référence absente ou invalide.")
    elif proof_vec is None:
        st.error("Preuve absente ou invalide.")
    else:
        sim = cosine_similarity(ref_vec, proof_vec)  # [0..1]
        percent = max(0.0, min(1.0, sim)) * 100.0
        st.metric("Similarité cosinus", f"{percent:.1f} %")

        if percent >= threshold:
            st.success(f"✅ Identité validée (≥ {threshold}%).")
        else:
            st.error(f"❌ Identité rejetée (< {threshold}%).")

# ===========================
# Notes pédagogiques (footer)
# ===========================
with st.expander("ℹ️ Explications (cliquer pour ouvrir)"):
    st.markdown(
        """
        **Pipeline pédagogique :**
        1. **Détection de visage** : Haar cascade (OpenCV) → on garde le **plus grand visage** de l'image.
        2. **Descripteur** : on convertit le visage en **HSV** puis on calcule deux histogrammes (**H** et **S**) que l’on **concatène et normalise**.
        3. **Similarité** : on calcule la **similarité cosinus** entre la référence et la preuve :
           \n
           \t\\( \\text{cos}(\\theta) = \\frac{\\mathbf{a}\\cdot\\mathbf{b}}{\\lVert\\mathbf{a}\\rVert\\,\\lVert\\mathbf{b}\\rVert} \\)
           \n
           Ici, plus la valeur est proche de **1** (donc 100%), plus les vecteurs sont “proches”.
        4. **Décision** : si la similarité ≥ **seuil** (réglable), on valide, sinon on rejette.

        **Réglages utiles (sidebar) :**
        - `scaleFactor` / `minNeighbors` : stabilité et sensibilité de la détection Haar.
        - `Histogram bins` : granularité du descripteur (plus grand = plus fin mais plus bruité).
        - `Seuil d'acceptation` : niveau d’exigence (en %).
        """
    )
