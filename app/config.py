import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
FAQ_URL = os.getenv("FAQ_URL", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")

if not FAQ_URL:
    raise ValueError("Missing FAQ_URL in .env")