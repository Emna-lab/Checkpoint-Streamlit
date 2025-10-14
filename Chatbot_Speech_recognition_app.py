#############################################################
# 📚 Imports
#############################################################
import io
import re
import string
import unicodedata
from pathlib import Path

import nltk
import numpy as np
import sounddevice as sd        # enregistrement local (pas nécessaire en déploiement Cloud)
import soundfile as sf          # écriture WAV en mémoire (BytesIO)
import speech_recognition as sr # transcription Google via SpeechRecognition
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


#############################################################
# 🔒 NLTK: télécharger si besoin (silencieux)
#############################################################
def safe_nltk_download(resource, name):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(name, quiet=True)

safe_nltk_download("tokenizers/punkt", "punkt")
safe_nltk_download("corpora/stopwords", "stopwords")
safe_nltk_download("corpora/wordnet", "wordnet")

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


#############################################################
# 🔤 Normalisation & prétraitement
#############################################################
def normalize_question(q: str) -> str:
    """
    Rendre les chaînes comparables (pour match exact “robuste”) :
    - lower + trims
    - enlever 'Q:' / 'Question:'
    - normaliser unicode (accents) + enlever ponctuation
    - compacter les espaces
    """
    q = (q or "").strip().lower()
    q = re.sub(r"^(q|question)\s*[:=\-]\s*", "", q, flags=re.I)
    q = unicodedata.normalize("NFKC", q)
    # supprimer diacritiques
    q = "".join(c for c in q if not unicodedata.category(c).startswith("M"))
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def preprocess_text(sentence: str):
    """Tokenisation + minuscules + stopwords/ponctuation + lemmatisation."""
    words = word_tokenize(sentence or "")
    words = [
        LEMMATIZER.lemmatize(w.lower())
        for w in words
        if w.lower() not in STOPWORDS and w not in string.punctuation
    ]
    return words


#############################################################
# 🎧 I/O audio local (facultatif en déploiement Cloud)
#############################################################
def record_microphone(seconds: float, samplerate: int = 16_000) -> bytes:
    """
    Enregistre `seconds` secondes au micro (mono, 16 kHz) et renvoie
    un WAV en octets (bytes). Fonctionne en local.
    """
    st.info("🎙️ Parlez maintenant…")
    sd.default.samplerate = samplerate
    sd.default.channels = 1
    data = sd.rec(int(seconds * samplerate), dtype="int16")
    sd.wait()
    st.success("✅ Enregistrement terminé.")

    # écrire en WAV dans un buffer mémoire
    buf = io.BytesIO()
    sf.write(buf, data, samplerate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def play_wav(wav_bytes: bytes, label: str = "Aperçu audio"):
    st.audio(wav_bytes, format="audio/wav")
    st.caption(label)


#############################################################
# 🗣️ Transcription (SpeechRecognition → Google)
#############################################################
def transcribe_google_from_wav(wav_bytes: bytes, language: str = "en-US") -> str:
    """Transcrit un WAV (bytes) via Google Web Speech API (clé non requise)."""
    r = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        return "❓ I could not understand any speech."
    except Exception as e:
        return f"⚠️ Google Speech error: {e}"


@st.cache_resource
def load_qa_file(file_path: Path):
    """
    Parse un fichier texte contenant :
      - intro (avant la 1ère Q)
      - paires Q/A (sur 1 ou 2 lignes)
      - outro (après la dernière A ou marqué dans la dernière A)
    Retourne : intro_lines, questions, answers, outro_lines, q_tokens
    """

    def _norm_ws(s: str) -> str:
        # Normalise Unicode + espaces + retours Windows
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("\u00A0", " ")           # NBSP -> espace
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[ \t]+", " ", s)
        return s

    # 1) Lecture et normalisation
    raw = Path(file_path).read_text(encoding="utf-8", errors="replace")
    if raw and raw[0] == "\ufeff":            # retire BOM éventuel
        raw = raw[1:]
    raw = _norm_ws(raw)

    # 2) Regex “plein texte” pour paires Q/A
    pair_re = re.compile(
        r'(?:^|\n)\s*(?:[-•]\s*)?(?:Q|Question)\s*[:=\-]?\s*(.*?)\s*'
        r'(?:\n|\s)+(?:[-•]\s*)?(?:A|Answer)\s*[:=\-]?\s*(.*?)(?=\n\s*(?:[-•]\s*)?(?:Q|Question)\s*[:=\-]?|\Z)',
        flags=re.IGNORECASE | re.DOTALL,
    )
    matches = list(pair_re.finditer(raw))

    questions, answers = [], []
    intro_lines, outro_lines = [], []

    # Marqueurs d’outro insérés dans la dernière réponse (ligne seule)
    OUTRO_SPLITTER = re.compile(r'(?im)^\s*(?:---\s*OUTRO\s*---|OUTRO:|FIN:)\s*$')

    if matches:
        # 3) Intro = avant la 1re paire
        intro_text = raw[:matches[0].start()]
        intro_lines = [ln for ln in intro_text.split("\n") if ln.strip()]

        # 4) Paires Q/A
        for m in matches:
            q = m.group(1).strip()
            a = m.group(2).strip()
            if q and a:
                questions.append(q)
                answers.append(a)

        # 5) OUTRO : priorité au “splitter” dans la dernière A:
        if answers:
            last_answer = answers[-1]
            parts = OUTRO_SPLITTER.split(last_answer, maxsplit=1)
            if len(parts) == 2:
                # a) Marqueur trouvé : on sépare
                answers[-1] = parts[0].strip()
                embedded_outro = parts[1].strip()
                if embedded_outro:
                    outro_lines = [ln for ln in embedded_outro.split("\n") if ln.strip()]
            else:
                # b) Sinon : on prend ce qu’il y a après la dernière paire
                after_pairs = raw[matches[-1].end():]
                outro_lines = [ln for ln in after_pairs.split("\n") if ln.strip()]

    else:
        # 6) Fallback ligne-à-ligne si la regex n’a rien trouvé
        found_qa, current_q = False, None
        PAT_Q = re.compile(r"^\s*(?:[-•]\s*)?(?:Q|Question)\s*[:=\-]?\s*(.*)$", re.I)
        PAT_A = re.compile(r"^\s*(?:[-•]\s*)?(?:A|Answer)\s*[:=\-]?\s*(.*)$", re.I)

        for line in raw.split("\n"):
            s = line.strip()
            if not s:
                continue
            mq = PAT_Q.match(s)
            if mq:
                found_qa = True
                current_q = mq.group(1).strip()
                continue
            ma = PAT_A.match(s)
            if ma and current_q:
                questions.append(current_q)
                answers.append(ma.group(1).strip())
                current_q = None
                continue

            if not found_qa:
                intro_lines.append(s)
            else:
                outro_lines.append(s)

    # 7) Contrôles + tokenisation des questions
    if not questions or len(questions) != len(answers):
        raise ValueError("❌ Invalid format: file must contain paired Q:/A: entries.")

    q_tokens = [preprocess_text(q) for q in questions]
    return intro_lines, questions, answers, outro_lines, q_tokens

#############################################################
# 🔎 Recherche de réponse (match exact ⇒ similarité Jaccard)
#############################################################
def find_best_answer(user_question: str, questions, answers, q_tokens) -> str:
    # 1) match exact (normalisé)
    norm = normalize_question(user_question)
    table = {normalize_question(q): i for i, q in enumerate(questions)}
    if norm in table:
        return answers[table[norm]]

    # 2) similarité Jaccard très simple (pédagogique)
    tokens = preprocess_text(user_question)
    if not tokens:
        return "⚠️ Please ask a clear question."

    qset = set(tokens)
    best_idx, best_sim = -1, 0.0
    for i, qt in enumerate(q_tokens):
        sset = set(qt)
        denom = (qset | sset)
        if not denom:
            continue
        sim = len(qset & sset) / float(len(denom))
        if sim > best_sim:
            best_sim, best_idx = sim, i

    if best_idx == -1:
        return "😕 I don't have an answer. Try rephrasing."
    return answers[best_idx]


#############################################################
# 🎨 Interface Streamlit
#############################################################
def main():
    st.set_page_config(page_title="VisitTunis • Chatbot", page_icon="🧭")
    st.title("🧭 VisitTunis  —  Your Friendly Guide")
    st.caption("Ask your chatbot by **text** or **voice**; get quick tips to enjoy Tunis!")

    # --- état persistant ---
    for k, v in {
        "wav_bytes": None,
        "transcript": "",
        "last_answer": "",
        "input_mode": "Text",
    }.items():
        st.session_state.setdefault(k, v)

    # --- charge le fichier Q&A ---
    qa_path = Path(__file__).parent / "VisitTunis_chatbot_QandA.txt"
    intro, questions, answers, outro, q_tokens = load_qa_file(qa_path)

    # --- joli encart d'accueil ---
    if intro:
        #st.markdown("### ☀️ Why Tunis?")
        #st.info("\n".join(intro))
        st.markdown("## 🌍 Discover Tunis")
        st.markdown(
            "<div style='font-size:20px; line-height:1.6; margin-bottom:20px;'>"
            + "<br>".join(intro) +
            "</div>",
            unsafe_allow_html=True
        )

    st.divider()

    # ====== Sidebar: options simples ======
    with st.sidebar:
        st.header("⚙️ Options")
        rec_seconds = st.slider("Record duration (sec)", 2, 15, 5)
        lang_google = st.text_input("Speech language (Google)", "en-US")
        st.caption("Examples: **en-US**, **fr-FR**, **ar-TN**…")

    # ====== Choix du mode d’entrée ======
    st.subheader("1) Choose your input")
    mode = st.radio("How do you want to ask?", ["Text", "Speech (microphone)"], index=0)
    st.session_state["input_mode"] = mode

    # ====== Zone de question ======
    user_question = ""

    if mode == "Text":
        user_question = st.text_input(
            "💬 Your question",
            placeholder="Example: What are the must-see places in Tunis?",
        )
    else:
        col_rec, col_trans = st.columns([1, 1])
        with col_rec:
            if st.button("🎙️ Record"):
                try:
                    st.session_state["wav_bytes"] = record_microphone(rec_seconds)
                except Exception as e:
                    st.error(f"Microphone error: {e}")

            if st.session_state["wav_bytes"]:
                play_wav(st.session_state["wav_bytes"], "Recorded preview")

        with col_trans:
            if st.button("📝 Transcribe"):
                wav = st.session_state.get("wav_bytes")
                if not wav:
                    st.warning("Record audio first.")
                else:
                    st.session_state["transcript"] = transcribe_google_from_wav(
                        wav, language=lang_google
                    )

            if st.session_state["transcript"]:
                st.success(f"Transcript: {st.session_state['transcript']}")
                user_question = st.session_state["transcript"]

        if st.button("🧹 Clear audio/transcript"):
            st.session_state["wav_bytes"] = None
            st.session_state["transcript"] = ""
            st.session_state["last_answer"] = ""

    st.divider()

    # ====== Lancer le chatbot ======
    st.subheader("2) Ask the chatbot")
    ask = st.button("🚀 Get answer")
    if ask:
        question = user_question.strip()
        if not question:
            st.warning("Type a question or transcribe one from speech first.")
        else:
            st.session_state["last_answer"] = find_best_answer(
                question, questions, answers, q_tokens
            )

    if st.session_state["last_answer"]:
        st.success(st.session_state["last_answer"])



    # ====== Aide : exemples ======
    with st.expander("💡 Example questions"):
        for q in questions[:]:
            st.write(f"- {q}")

    # ===== Outro =====
    if outro:
        st.markdown("---")
        st.markdown("## 🌟 Before you go")  # Titre plus grand
        st.markdown(
            f"<div style='font-size:18px; line-height:1.6; color:#333;'>"
            + "<br>".join(outro) +
            "</div>",
            unsafe_allow_html=True
        )

#############################################################
# 🚀 Main - Lancement
#############################################################
if __name__ == "__main__":
    main()
