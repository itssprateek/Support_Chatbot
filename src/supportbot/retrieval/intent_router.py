"""
Intent Router — maps a predicted intent to a response template.

Queries the intent_responses table in PostgreSQL.
Falls back to a generic message if the intent isn't found in the DB.
"""

from src.supportbot.db.schema import get_connection

FALLBACK_RESPONSE = (
    "I understand your question, but I don't have a specific answer for that yet. "
    "Please contact our support team for further assistance."
)

REPHRASE_RESPONSE = (
    "I'm not fully confident I understood your question. "
    "Could you rephrase it or provide more details?"
)


def get_response_for_intent(intent: str) -> str:
    """
    Look up the response template for a given intent.

    Args:
        intent: the predicted intent label (e.g. 'track_order')

    Returns:
        The response template string, or a fallback if not found.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT response_template FROM intent_responses WHERE intent = %s",
                (intent,),
            )
            row = cur.fetchone()
            if row:
                return row[0]
            return FALLBACK_RESPONSE
    finally:
        conn.close()
