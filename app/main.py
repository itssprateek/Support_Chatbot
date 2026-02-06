import os
import requests
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------- LOAD ENV ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FAQ_URL = os.getenv("FAQ_URL")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Put it in .env and restart the terminal.")
if not FAQ_URL:
    raise ValueError("FAQ_URL is missing. Put it in .env and restart the terminal.")

# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = """
You are an e-commerce customer support chatbot.

Hard rules:
1) Answer ONLY using the FAQ knowledge provided in CONTEXT.
2) Do NOT invent anything.
3) If the answer is not in CONTEXT, ask ONE short clarifying question.
4) Keep answers short, professional, and helpful.
5) Reply in the SAME language as the user.
"""

# ---------------- HELPERS: LANGUAGE ----------------
def detect_language_simple(text: str) -> str:
    """
    Cheap fallback detection (no LLM).
    Returns "de" for some German signals, else "en".
    """
    t = text.lower()
    de_markers = ["ä", "ö", "ü", "ß", "bestellung", "lieferung", "rückgabe", "erstattung", "zahlung", "konto", "passwort", "wo ist"]
    if any(m in t for m in de_markers):
        return "de"
    return "en"

def detect_language(llm, text: str) -> str:
    """
    LLM detection. Falls back to simple detection if API fails.
    """
    try:
        resp = llm.invoke([
            SystemMessage(content="Detect the language of the user's text. Return ONLY the ISO 639-1 code (e.g., en, de, hi, fr, es)."),
            HumanMessage(content=text)
        ])
        code = (resp.content or "").strip().lower()
        return code if len(code) == 2 else detect_language_simple(text)
    except Exception as e:
        print(f"[Language detection error] {e}")
        return detect_language_simple(text)

def translate(llm, text: str, target_lang: str) -> str:
    """
    Translate text to target_lang. If translation fails, returns original text.
    """
    try:
        resp = llm.invoke([
            SystemMessage(content=f"Translate the text to {target_lang}. Return only the translation."),
            HumanMessage(content=text)
        ])
        out = (resp.content or "").strip()
        return out if out else text
    except Exception as e:
        print(f"[Translation error] {e}")
        return text

# ---------------- LOAD FAQ FROM GITHUB ----------------
def load_faq():
    res = requests.get(FAQ_URL, timeout=20)
    res.raise_for_status()

    lines = [l for l in res.text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        raise ValueError("FAQ markdown table not found or too short.")

    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    data_lines = lines[2:]  # skip separator row

    faq = []
    for line in data_lines:
        values = [v.strip() for v in line.strip("|").split("|")]
        if len(values) != len(headers):
            continue
        faq.append(dict(zip(headers, values)))

    if not faq:
        raise ValueError("Parsed FAQ is empty. Check your markdown table formatting.")
    return faq

# ---------------- NLP MATCHER ----------------
class FAQMatcher:
    def __init__(self, faq):
        self.faq = faq
        self.questions = [f.get("User Question", "") for f in faq]
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.questions)

    def search(self, query: str):
        vec = self.vectorizer.transform([query])
        scores = cosine_similarity(vec, self.matrix)[0]
        idx = int(scores.argmax())
        return self.faq[idx], float(scores[idx])

# ---------------- CHATBOT ----------------
def main():
    faq = load_faq()
    matcher = FAQMatcher(faq)

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=250,
        api_key=GROQ_API_KEY,  # explicit to avoid env issues
    )

    print("🛒 Multilingual E-Commerce Support Bot (type 'exit' to quit)\n")

    exit_words = [
        "exit", "quit", "bye", "goodbye",
        "ende", "beenden",
        "salir", "quitter",
        "बंद", "बाहर"
    ]

    HIGH_CONF = 0.55

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() in exit_words:
            print("Bot: Bye! 👋")
            break

        # detect language and translate query to English for matching (FAQ is English)
        user_lang = detect_language(llm, user_input)
        query_en = user_input if user_lang == "en" else translate(llm, user_input, "en")

        match, score = matcher.search(query_en)

        # If match confident: answer directly from FAQ (then translate back)
        if score >= HIGH_CONF:
            bot_text_en = match.get("Bot Response", "")
            bot_text = bot_text_en if user_lang == "en" else translate(llm, bot_text_en, user_lang)
            print(f"Bot: {bot_text}\n")
            continue

        # If not confident: still grounded answer using top FAQ match
        context = f"Q: {match.get('User Question','')}\nA: {match.get('Bot Response','')}"

        try:
            resp = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"""
CONTEXT (FAQ):
{context}

User language: {user_lang}
User question (English for grounding):
{query_en}

Answer using ONLY the CONTEXT.
If not clearly present, ask ONE short clarifying question.
""".strip())
            ])
            bot_text_en = (resp.content or "").strip()
        except Exception as e:
            print(f"[LLM error] {e}")
            bot_text_en = "I’m having trouble accessing the assistant right now. Please try again."

        bot_text = bot_text_en if user_lang == "en" else translate(llm, bot_text_en, user_lang)
        print(f"Bot: {bot_text}\n")

if __name__ == "__main__":
    main()