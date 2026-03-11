"""
Central configuration for the Support Chatbot.

All paths, thresholds, and DB settings in one place.
Uses environment variables with sensible defaults.
"""

import os


# ── Paths ────────────────────────────────────────────────────────
ARTIFACTS_DIR  = os.getenv("ARTIFACTS_DIR", "artifacts/intent_model")
PROCESSED_DIR  = os.getenv("PROCESSED_DIR", "data/processed")
FAQ_PATH       = os.getenv("FAQ_PATH", "data/faq/faq_table.md")
EVAL_OUTPUT    = os.getenv("EVAL_OUTPUT", "artifacts/eval_results")

# ── Intent Confidence Thresholds ─────────────────────────────────
CONFIDENCE_HIGH  = float(os.getenv("CONFIDENCE_HIGH", "0.65"))   # above → return response
CONFIDENCE_LOW   = float(os.getenv("CONFIDENCE_LOW", "0.65"))    # below → ask to rephrase

# ── PostgreSQL (for online pipeline — not needed for offline) ────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_NAME     = os.getenv("DB_NAME", "supportbot")
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ── Retraining Triggers ─────────────────────────────────────────
RETRAIN_CONFIDENCE_THRESHOLD = float(os.getenv("RETRAIN_CONF_THRESHOLD", "0.70"))
PSEUDO_LABEL_THRESHOLD       = float(os.getenv("PSEUDO_LABEL_THRESHOLD", "0.85"))
