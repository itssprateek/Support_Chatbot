import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ---------------- ENV ----------------
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FAQ_URL = os.getenv("FAQ_URL")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = """
You are an e-commerce customer support chatbot.

Rules:
1) Answer ONLY using the FAQ data provided.
2) Do NOT invent information.
3) If the answer is not in the FAQ, say you don’t have that info and ask a follow-up question.
4) Keep responses short, professional, and helpful.
"""

# ---- FAQ cache (avoid downloading on every message) ----
_FAQ_CACHE = {"text": None}

def load_faq_text() -> str:
    if _FAQ_CACHE["text"] is not None:
        return _FAQ_CACHE["text"]

    if not FAQ_URL:
        raise ValueError("FAQ_URL missing in .env")

    res = requests.get(FAQ_URL, timeout=20)
    res.raise_for_status()
    _FAQ_CACHE["text"] = res.text
    return _FAQ_CACHE["text"]

def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY missing in .env")
    return ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! 👋 I’m your e-commerce support bot.\nAsk me anything about orders, delivery, returns, payments, etc."
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send your support question as a message.\nType /start to begin."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = (update.message.text or "").strip()
    if not user_text:
        await update.message.reply_text("Please send a text message.")
        return

    # Optional: show “typing…”
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        faq_text = load_faq_text()
        llm = get_llm()

        messages = [
            SystemMessage(content=SYSTEM_PROMPT + "\n\nFAQ DATA:\n" + faq_text),
            HumanMessage(content=user_text),
        ]
        resp = llm.invoke(messages)

        answer = (resp.content or "").strip()
        if not answer:
            answer = "Sorry — I couldn’t generate a response. Please rephrase your question."

        await update.message.reply_text(answer)

    except requests.HTTPError as e:
        await update.message.reply_text(f"FAQ link error: {e}")
    except Exception as e:
        await update.message.reply_text(f"Bot error: {e}")

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN missing in .env")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Telegram bot running (polling). Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()