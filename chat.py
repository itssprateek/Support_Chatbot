"""
Minimal terminal chat interface for the Support Chatbot.

Features:
- Type a message → get intent prediction + response
- After each response, rate with 👍 (1) or 👎 (0)
- Everything logged to PostgreSQL
- Type 'quit' or 'exit' to stop

Usage:
    python chat.py
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from src.supportbot.application.chat_service import ChatService
from src.supportbot.db.logger import update_feedback


BANNER = """
╔══════════════════════════════════════════════════╗
║         E-Commerce Support Chatbot               ║
║     Powered by IntentBiGRU + PostgreSQL           ║
║                                                   ║
║  Type your question and press Enter.              ║
║  After each response, rate: 1=👍  0=👎  Enter=skip ║
║  Type 'quit' to exit.                             ║
╚══════════════════════════════════════════════════╝
"""

CUSTOMER_ID = "demo_user"


def collect_feedback(log_id: int) -> None:
    """Ask user for feedback on the response."""
    feedback_input = input("  Rate this response [1=👍 / 0=👎 / Enter=skip]: ").strip()

    if feedback_input == "1":
        update_feedback(log_id, 1)
        print("  ✅ Thanks for the positive feedback!")
    elif feedback_input == "0":
        update_feedback(log_id, 0)
        print("  📝 Thanks — we'll use this to improve.")
    else:
        pass  # skipped — no feedback recorded


def main():
    print(BANNER)

    # Initialize chat service (loads model)
    service = ChatService()
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Process message
        result = service.handle_message(CUSTOMER_ID, user_input)

        # Display response
        print(f"\n  Bot: {result['response']}")
        print(f"  [{result['intent']} | conf: {result['confidence']} | {result['route']}]")
        print()

        # Collect feedback
        collect_feedback(result["log_id"])
        print()


if __name__ == "__main__":
    main()
