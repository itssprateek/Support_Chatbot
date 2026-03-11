"""
Chat Service — core routing logic for the Support Chatbot.

Flow:
    User message
        → IntentBiGRU predicts (intent, confidence)
        → confidence ≥ threshold?
            YES → query intent_responses table → return response
            NO  → return rephrase prompt
        → log everything to chat_logs table
        → return response + log_id (for feedback)

This is the single entry point the chat interface calls.
"""

from src.supportbot.ml.inference import IntentPredictor
from src.supportbot.retrieval.intent_router import get_response_for_intent, REPHRASE_RESPONSE
from src.supportbot.db.logger import log_interaction
from src.supportbot.core.config import ARTIFACTS_DIR, CONFIDENCE_HIGH


class ChatService:
    """Stateless chat service — one method handles the full pipeline."""

    def __init__(self):
        self.predictor = IntentPredictor(ARTIFACTS_DIR)
        print(f"✅ ChatService ready | Model loaded from {ARTIFACTS_DIR}")
        print(f"   Confidence threshold: {CONFIDENCE_HIGH}")

    def handle_message(self, customer_id: str, message: str) -> dict:
        """
        Process a user message end-to-end.

        Args:
            customer_id: identifies the user
            message:     raw user input text

        Returns:
            dict with keys:
                - response:   the bot's reply text
                - intent:     predicted intent label
                - confidence: model confidence (0.0 - 1.0)
                - route:      'high_conf' or 'rephrase'
                - log_id:     DB row ID (used for feedback)
        """
        # ── Step 1: Predict intent ───────────────────────────────
        intent, confidence = self.predictor.predict(message)

        # ── Step 2: Confidence gate ──────────────────────────────
        if confidence >= CONFIDENCE_HIGH:
            response = get_response_for_intent(intent)
            route = "high_conf"
        else:
            response = REPHRASE_RESPONSE
            route = "rephrase"

        # ── Step 3: Log to PostgreSQL ────────────────────────────
        log_id = log_interaction(
            customer_id=customer_id,
            user_message=message,
            predicted_intent=intent,
            confidence_score=confidence,
            response_sent=response,
            route=route,
        )

        return {
            "response": response,
            "intent": intent,
            "confidence": round(confidence, 4),
            "route": route,
            "log_id": log_id,
        }
