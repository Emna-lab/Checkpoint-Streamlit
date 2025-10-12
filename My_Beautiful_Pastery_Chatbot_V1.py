#############################################################
# 📚 Imports
#############################################################
import unicodedata
import re
import string
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path

#############################################################
# 🔒 NLTK: télécharger si besoin
#############################################################
def safe_nltk_download(resource, name):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(name)

safe_nltk_download('tokenizers/punkt', 'punkt')
safe_nltk_download('corpora/stopwords', 'stopwords')
safe_nltk_download('corpora/wordnet', 'wordnet')

# --- Helpers de normalisation (mets-les au-dessus du parseur)
def normalize_question(q: str) -> str:
    """Normalise une question pour l'appariement exact."""
    q = q.strip().lower()
    # retirer préfixe q:/question:
    q = re.sub(r'^(q|question)\s*[:=\-]\s*', '', q, flags=re.I)
    # normaliser unicode et enlever accents
    q = unicodedata.normalize('NFKC', q)
    q = ''.join(c for c in q if not unicodedata.category(c).startswith('M'))
    # remplacer ponctuation par espaces
    q = re.sub(r'[^\w\s]', ' ', q)
    # compacter espaces
    q = re.sub(r'\s+', ' ', q).strip()
    return q

#############################################################
# ⚙️ Prétraitement
#############################################################
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def preprocess_text(sentence: str):
    """Tokenisation + minuscules + stopwords/ponctuation + lemmatisation."""
    words = word_tokenize(sentence)
    words = [w.lower() for w in words if w.lower() not in STOPWORDS and w not in string.punctuation]
    words = [LEMMATIZER.lemmatize(w) for w in words]
    return words

#############################################################
# 📂 Lecture du fichier Q&A (robuste : Q/A sur une ou deux lignes)
#############################################################

@st.cache_resource
def load_qa_file(file_path: Path):
    """
    Retourne:
      - intro_lines : texte avant la section Q&A (list[str])
      - questions   : liste des questions (list[str])
      - answers     : liste des réponses (list[str])
      - outro_lines : texte après la section Q&A (list[str])
      - q_tokens    : questions tokenisées (pour la similarité)
    Accepte :
      • Q: ... A: ... sur la même ligne
      • Q: ... \n A: ... sur deux lignes
      • Puces éventuelles ('-', '•') devant Q:/A:
      • Espaces/tirets unicode (NBSP, –, —)
      • Retours Windows \r\n
    """

    def _norm_whitespace(s: str) -> str:
        # normalise unicode + remplace NBSP/tirets longs + compacte espaces
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("\u00A0", " ")   # NBSP -> espace
        s = s.replace("\u2013", "-")   # – -> -
        s = s.replace("\u2014", "-")   # — -> -
        s = s.replace("\r\n", "\n")    # CRLF -> LF
        s = s.replace("\r", "\n")
        s = re.sub(r"[ \t]+", " ", s)
        return s

    raw = Path(file_path).read_text(encoding="utf-8", errors="replace")
    # retirer un éventuel BOM
    if raw and raw[0] == "\ufeff":
        raw = raw[1:]
    raw = _norm_whitespace(raw)

    # --------- 1) REGEX pleine page : Q … A … (jusqu’à prochaine Q ou fin) ----------
    pair_re = re.compile(
        r'(?:^|\n)\s*(?:[-•]\s*)?(?:Q|Question)\s*[:=\-]?\s*(.*?)\s*'
        r'(?:\n|\s)+(?:[-•]\s*)?(?:A|Answer)\s*[:=\-]?\s*(.*?)(?=\n\s*(?:[-•]\s*)?(?:Q|Question)\s*[:=\-]?|\Z)',
        flags=re.IGNORECASE | re.DOTALL
    )

    matches = list(pair_re.finditer(raw))
    questions, answers = [], []

    if matches:
        for m in matches:
            q = m.group(1).strip()
            a = m.group(2).strip()
            if q and a:
                questions.append(q)
                answers.append(a)

        # calcul d’intro/outro via la zone couverte par la 1ère et dernière paire
        intro_text = raw[:matches[0].start()]
        outro_text = raw[matches[-1].end():]

        intro_lines = [ln for ln in intro_text.split("\n") if ln.strip()]
        outro_lines = [ln for ln in outro_text.split("\n") if ln.strip()]

    else:
        # --------- 2) FALLBACK ligne à ligne (Q seule puis A seule) ----------
        intro_lines, outro_lines = [], []
        found_qa_section = False
        current_q = None

        PAT_Q = re.compile(r'^\s*(?:[-•]\s*)?(?:Q|Question)\s*[:=\-]?\s*(.*)$', re.I)
        PAT_A = re.compile(r'^\s*(?:[-•]\s*)?(?:A|Answer)\s*[:=\-]?\s*(.*)$', re.I)
        PAT_QA_SAME = re.compile(
            r'^\s*(?:[-•]\s*)?(?:Q|Question)\s*[:=\-]?\s*(.*?)\s+(?:[-•]\s*)?(?:A|Answer)\s*[:=\-]?\s*(.*?)\s*$',
            re.I
        )

        for raw_line in raw.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            both = PAT_QA_SAME.match(line)
            if both:
                found_qa_section = True
                questions.append(both.group(1).strip())
                answers.append(both.group(2).strip())
                current_q = None
                continue

            mq = PAT_Q.match(line)
            if mq:
                found_qa_section = True
                current_q = mq.group(1).strip()
                continue

            ma = PAT_A.match(line)
            if ma and current_q:
                questions.append(current_q)
                answers.append(ma.group(1).strip())
                current_q = None
                continue

            if not found_qa_section:
                intro_lines.append(line)
            else:
                outro_lines.append(line)

    # ---------- Sécurité / message utile ----------
    if not questions or len(questions) != len(answers):
        # petit extrait pour diagnostiquer rapidement au besoin
        excerpt = "\n".join([ln for ln in raw.split("\n")[:20]])
        raise ValueError(
            "❌ Format invalide : il faut des paires Q:/A: (sur une ou deux lignes).\n"
            "Extrait du début du fichier pour diagnostic :\n"
            f"{excerpt}"
        )

    # tokenisation des questions pour la similarité
    q_tokens = [preprocess_text(q) for q in questions]
    return intro_lines, questions, answers, outro_lines, q_tokens

#############################################################
# 🔎 Recherche de la meilleure réponse (similarité de Jaccard)
#############################################################
def find_best_answer(user_question: str, questions, answers, q_tokens):
    # 1) Essayez d'abord un match exact sur une version normalisée
    norm_user = normalize_question(user_question)
    norm_map = {normalize_question(q): i for i, q in enumerate(questions)}
    if norm_user in norm_map:
        return answers[norm_map[norm_user]]

    # 2) Sinon, fallback sur la similarité (ton code existant)
    tokens = preprocess_text(user_question)
    if not tokens:
        return "⚠️Thanks for asking a valid question."

    qset = set(tokens)
    best_idx, best_sim = -1, 0.0
    for i, qt in enumerate(q_tokens):
        sset = set(qt)
        union = qset | sset
        if not union:
            continue
        sim = len(qset & sset) / float(len(union))
        if sim > best_sim:
            best_sim, best_idx = sim, i

    if best_idx == -1:
        return ("😕 Oups..., I do not have an answer. "
                "Please ask another one or contact us during opening hours.")
    return answers[best_idx]


#############################################################
# 🎨 Interface Streamlit
#############################################################
def main():
    st.title("💗🥐 My Beautiful Pastery — Chatbot")
    st.write("Posez vos questions : horaires, boutique en ligne, produits healthy, gâteaux sur mesure, etc.")

    target_file = Path(__file__).parent / "My_pastery_chatbot_QandA.txt"
    intro, questions, answers, outro, q_tokens = load_qa_file(target_file)

    with st.expander("🔎 Debug (questions normalisées)"):
        st.write([normalize_question(q) for q in questions])
        st.write("User normalized:", normalize_question(st.session_state.get("last_q", "")))

    if intro:
        st.markdown("### 🌸 About us")
        st.write("\n".join(intro))

    user_q = st.text_input("💬 Your question :", placeholder="Ex. : Is it easy to access to your boutique ?")
    if st.button("Envoyer"):
        st.session_state["last_q"] = user_q
        if not user_q.strip():
            st.warning("Thanks for asking me a question 😉")
        else:
            answer = find_best_answer(user_q, questions, answers, q_tokens)
            st.markdown(f"**🤖 Chatbot :** {answer}")

    if outro:
        st.markdown("---")
        st.markdown("### 🌸 Other informations")
        st.write("\n".join(outro))

#############################################################
# 🚀 Lancement
#############################################################
if __name__ == "__main__":
    main()
