"""
Logger for chat interactions.

Every user message → model prediction → response is logged to PostgreSQL.
Feedback (👍/👎) is updated after the user responds.

This table powers:
- analytics.py  → weekly model health report
- retrain.py    → pull low-feedback logs for retraining
"""

import psycopg2
from src.supportbot.db.schema import get_connection


def log_interaction(
    customer_id: str,
    user_message: str,
    predicted_intent: str,
    confidence_score: float,
    response_sent: str,
    route: str = "high_conf",
) -> int:
    """
    Log a single chat interaction to the database.

    Args:
        customer_id:      who sent the message
        user_message:     the raw user input
        predicted_intent: model's predicted intent label
        confidence_score: softmax confidence (0.0 - 1.0)
        response_sent:    the response returned to the user
        route:            'high_conf' or 'rephrase'

    Returns:
        log_id of the inserted row
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_logs
                    (customer_id, user_message, predicted_intent,
                     confidence_score, response_sent, route)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING log_id
                """,
                (customer_id, user_message, predicted_intent,
                 confidence_score, response_sent, route),
            )
            log_id = cur.fetchone()[0]
        conn.commit()
        return log_id
    finally:
        conn.close()


def update_feedback(log_id: int, feedback: int) -> None:
    """
    Update user feedback for a specific chat log entry.

    Args:
        log_id:   the chat_logs row to update
        feedback: 1 = helpful (👍), 0 = not helpful (👎)
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_logs SET user_feedback = %s WHERE log_id = %s",
                (feedback, log_id),
            )
        conn.commit()
    finally:
        conn.close()
