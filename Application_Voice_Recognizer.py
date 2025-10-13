# app_speech_stt.py
# -------------------------------------------------------
# Démo STT simple pour débutants — enregistrement local,
# upload WAV, transcription via Google ou Deepgram.
# -------------------------------------------------------

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import sounddevice as sd       # pour enregistrer au micro (local)
import soundfile as sf         # pour écrire du WAV en mémoire
import streamlit as st
import requests                # pour REST Deepgram
import speech_recognition as sr # pour Google (SpeechRecognition)


# ========= Helpers (I/O audio) =========
def record_microphone(seconds: float, samplerate: int = 16_000) -> bytes:
    """Enregistre `seconds` au micro (mono, 16 kHz) et renvoie des octets WAV."""
    # Enregistrement
    st.info("🎙️ Parlez maintenant… (bruit ambiant calibré)")
    sd.default.samplerate = samplerate
    sd.default.channels = 1
    data = sd.rec(int(seconds * samplerate), dtype="int16")
    sd.wait()
    st.success("✅ Enregistrement terminé.")

    # Écriture WAV en mémoire
    buf = io.BytesIO()
    sf.write(buf, data, samplerate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def play_wav(wav_bytes: bytes, label: str = "Aperçu audio"):
    """Affiche un lecteur audio Streamlit pour des octets WAV."""
    st.audio(wav_bytes, format="audio/wav")
    st.caption(label)


# ========= STT backends =========
def transcribe_google_from_wav(wav_bytes: bytes, language: str = "en-US") -> str:
    """Transcrit via Google (lib SpeechRecognition) — offline upload/bytes."""
    r = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        return "❓ Aucun texte compréhensible (Google)."
    except Exception as e:
        return f"⚠️ Erreur Google: {e}"


def transcribe_deepgram_from_wav(
    wav_bytes: bytes,
    api_key: str,
    model: str = "nova-2",
    language: str = "en",
) -> str:
    """Transcrit via Deepgram REST (/v1/listen)."""
    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {api_key}",  # ⚠️ pas "Bearer"
        "Content-Type": "audio/wav",
    }
    params = {"model": model, "language": language}

    r = requests.post(url, headers=headers, params=params, data=wav_bytes, timeout=30)
    if r.status_code == 401:
        return "❌ Deepgram 401 (Unauthorized). Vérifie ta clé API dans .streamlit/secrets.toml."
    if r.status_code >= 400:
        return f"❌ Deepgram {r.status_code}: {r.text}"

    data = r.json()
    # Structure standard : data["results"]["channels"][0]["alternatives"][0]["transcript"]
    try:
        return data["results"]["channels"][0]["alternatives"][0]["transcript"].strip() or "❓ (vide)"
    except Exception:
        return f"⚠️ Réponse inattendue Deepgram: {data}"


# ========= App =========
st.set_page_config(page_title="Speech-to-Text • Démo", page_icon="🎤", layout="centered")
st.title("🎤 Speech-to-Text (démo simple et pédagogique)")

# 1) État persistant — on ne réinitialise pas si déjà défini
st.session_state.setdefault("wav_bytes", None)
st.session_state.setdefault("transcript", "")
st.session_state.setdefault("engine", "Google")

# 2) Choix moteur
engine = st.selectbox(
    "Choisissez une API de reconnaissance vocale :",
    ["Google (SpeechRecognition)", "Deepgram (REST)"],
    index=0 if st.session_state["engine"] == "Google" else 1,
)
st.session_state["engine"] = "Google" if engine.startswith("Google") else "Deepgram"

# 3) Options (simples)
with st.expander("Options d’écoute (facultatif)", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        seconds = st.slider("Durée d'enregistrement (secondes)", 2, 15, 5)
        lang_google = st.text_input("Langue (Google)", value="en-US")
    with col2:
        lang_dg = st.text_input("Langue (Deepgram)", value="en")
        model_dg = st.text_input("Modèle (Deepgram)", value="nova-2")

# 4) Source audio : micro ou upload
st.header("1) Choisir une source audio")

left, right = st.columns(2)

with left:
    st.subheader("🎙️ Micro (local)")
    if st.button("Start Recording"):
        try:
            wav = record_microphone(seconds, 16_000)
            st.session_state["wav_bytes"] = wav
            play_wav(wav, "Enregistrement (micro)")
        except Exception as e:
            st.error(f"Impossible d'enregistrer : {e}")

with right:
    st.subheader("📂 Fichier WAV")
    up = st.file_uploader("Déposez un WAV (mono 16 kHz idéalement)", type=["wav"])
    if up is not None:
        st.session_state["wav_bytes"] = up.read()
        play_wav(st.session_state["wav_bytes"], "WAV importé")

# 5) Transcrire
st.header("2) Transcrire")
col_t1, col_t2 = st.columns([1, 1])
with col_t1:
    if st.button("📝 Transcrire maintenant"):
        wav = st.session_state.get("wav_bytes")
        if not wav:
            st.warning("Veuillez d'abord **enregistrer** ou **uploader** un WAV.")
        else:
            if st.session_state["engine"] == "Google":
                text = transcribe_google_from_wav(wav, language=lang_google)
            else:
                # Récup clé depuis secrets
                api_key = st.secrets.get("deepgram", {}).get("api_key", "")
                if not api_key:
                    st.error("Clé Deepgram manquante. Ajoute-la dans .streamlit/secrets.toml.")
                    text = ""
                else:
                    text = transcribe_deepgram_from_wav(
                        wav, api_key=api_key, model=model_dg, language=lang_dg
                    )
            st.session_state["transcript"] = text

with col_t2:
    if st.button("🧹 Effacer audio & texte"):
        st.session_state["wav_bytes"] = None
        st.session_state["transcript"] = ""
        st.success("État remis à zéro.")

# 6) Affichage du résultat
st.subheader("Transcription :")
if st.session_state["transcript"]:
    st.success(st.session_state["transcript"])
else:
    st.info("Aucune transcription pour l’instant.")

# 7) Infos déploiement / dépendances
st.markdown("---")
st.caption(
    "Dépendances : `sounddevice`, `soundfile` (micro), `SpeechRecognition` (Google), `requests` (Deepgram). "
    "Pour Deepgram, crée un fichier `.streamlit/secrets.toml` avec :\n\n"
    "```toml\n[deepgram]\napi_key = \"DG_API_KEY_...\"\n```"
)
