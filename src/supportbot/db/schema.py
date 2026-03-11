"""
PostgreSQL schema and connection manager for the Support Chatbot.

Tables:
- customers:         registered users
- chat_logs:         every conversation logged with confidence + feedback
- intent_responses:  intent → response template (replaces FAQ markdown)

Usage:
    python -m src.supportbot.db.schema          # creates all tables
    python -m src.supportbot.db.schema --seed    # creates tables + seeds intent_responses
"""

import sys, os, argparse

import psycopg2
from psycopg2.extras import execute_values

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.supportbot.core.config import DATABASE_URL


# ── Connection Helper ────────────────────────────────────────────

def get_connection():
    """Return a new PostgreSQL connection."""
    return psycopg2.connect(DATABASE_URL)


# ── Table Definitions ────────────────────────────────────────────

CREATE_CUSTOMERS = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id     VARCHAR(50) PRIMARY KEY,
    name            VARCHAR(100),
    email           VARCHAR(150),
    created_at      TIMESTAMP DEFAULT NOW()
);
"""

CREATE_CHAT_LOGS = """
CREATE TABLE IF NOT EXISTS chat_logs (
    log_id              SERIAL PRIMARY KEY,
    customer_id         VARCHAR(50) REFERENCES customers(customer_id) ON DELETE SET NULL,
    timestamp           TIMESTAMP DEFAULT NOW(),
    user_message        TEXT NOT NULL,
    predicted_intent    VARCHAR(100),
    confidence_score    REAL,
    response_sent       TEXT,
    route               VARCHAR(20) DEFAULT 'high_conf',
    user_feedback       SMALLINT DEFAULT NULL
);
"""
# route:         'high_conf' | 'rephrase'
# user_feedback: 1 = helpful, 0 = not helpful, NULL = no response

CREATE_INTENT_RESPONSES = """
CREATE TABLE IF NOT EXISTS intent_responses (
    intent              VARCHAR(100) PRIMARY KEY,
    response_template   TEXT NOT NULL,
    updated_at          TIMESTAMP DEFAULT NOW()
);
"""

# ── Index for analytics queries ──────────────────────────────────

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_chat_logs_timestamp ON chat_logs (timestamp);
CREATE INDEX IF NOT EXISTS idx_chat_logs_intent ON chat_logs (predicted_intent);
CREATE INDEX IF NOT EXISTS idx_chat_logs_confidence ON chat_logs (confidence_score);
"""


# ── Seed Data: 27 intent → response mappings ────────────────────

INTENT_RESPONSES_SEED = [
    ("cancel_order", "To cancel your order, please provide your order number and we'll process the cancellation. Note that orders already shipped may not be eligible for cancellation."),
    ("change_order", "To modify your order, share your order number and the changes you'd like to make. We can update items, quantities, or shipping details if the order hasn't been dispatched yet."),
    ("change_shipping_address", "To update your shipping address, please provide your order number and the new address. Changes are possible only before the order is shipped."),
    ("check_cancellation_fee", "Cancellation fees depend on the order status. Orders cancelled before dispatch are free. Once shipped, a restocking fee of up to 15% may apply. Share your order number for exact details."),
    ("check_invoice", "To view your invoice, please provide your order number or account email. You can also find invoices under 'Order History' in your account dashboard."),
    ("check_payment_methods", "We accept Visa, Mastercard, American Express, PayPal, Apple Pay, and Google Pay. All transactions are securely processed and encrypted."),
    ("check_refund_policy", "Our refund policy allows returns within 30 days of delivery. Items must be unused and in original packaging. Refunds are processed within 5-7 business days after we receive the return."),
    ("complaint", "We're sorry to hear about your experience. Please describe the issue in detail and include your order number so we can investigate and resolve it promptly."),
    ("contact_customer_service", "You can reach our customer service team via email at support@example.com or call us at 1-800-555-0199. Our hours are Mon-Fri, 9 AM to 6 PM EST."),
    ("contact_human_agent", "Let me connect you with a human agent. Please hold while we transfer you. Estimated wait time is under 5 minutes during business hours."),
    ("create_account", "To create an account, visit our website and click 'Sign Up'. You'll need your email address and a password. You can also sign up using your Google or Apple account."),
    ("delete_account", "To delete your account, go to Settings > Account > Delete Account, or contact support. Please note this action is permanent and all data will be removed within 30 days."),
    ("delivery_options", "We offer Standard (5-7 business days), Express (2-3 business days), and Next-Day delivery. Shipping costs vary by option and location. Free standard shipping on orders over $50."),
    ("delivery_period", "Standard delivery takes 5-7 business days. Express is 2-3 business days. Next-day delivery is available for orders placed before 2 PM. Provide your order number for a specific estimate."),
    ("edit_account", "To edit your account details, log in and go to Settings > Profile. You can update your name, email, phone number, and password from there."),
    ("get_invoice", "To receive a copy of your invoice, provide your order number and we'll email it to you. You can also download invoices directly from your account under 'Order History'."),
    ("get_refund", "To request a refund, please share your order number and reason for the return. Once approved, refunds are processed within 5-7 business days to your original payment method."),
    ("newsletter_subscription", "To manage your newsletter subscription, go to Settings > Notifications. You can subscribe or unsubscribe at any time. We send updates about new products and exclusive offers."),
    ("payment_issue", "We're sorry about the payment issue. Please verify your card details and ensure sufficient funds. If the problem persists, try a different payment method or contact your bank. Share your order number for further assistance."),
    ("place_order", "To place an order, browse our catalog, add items to your cart, and proceed to checkout. You'll need to provide shipping and payment details. Need help with a specific step?"),
    ("recover_password", "To reset your password, click 'Forgot Password' on the login page. We'll send a reset link to your registered email. The link expires in 24 hours."),
    ("registration_problems", "If you're having trouble registering, ensure your email isn't already linked to an existing account. Try clearing your browser cache or using a different browser. Contact support if the issue persists."),
    ("review", "We appreciate your feedback! You can leave a review on the product page under 'Customer Reviews'. Your review helps other shoppers and helps us improve."),
    ("set_up_shipping_address", "To add a shipping address, go to Settings > Addresses > Add New. You can save multiple addresses and set a default one for faster checkout."),
    ("switch_account", "To switch between accounts, log out of your current account and log in with the other account credentials. We don't support multiple simultaneous sessions for security reasons."),
    ("track_order", "To track your order, please provide your order number. You can also track it in real-time under 'My Orders' in your account dashboard."),
    ("track_refund", "To check your refund status, provide your order number. Refunds typically take 5-7 business days after approval. You'll receive an email confirmation once processed."),
]


# ── Setup Functions ──────────────────────────────────────────────

def create_tables(conn):
    """Create all tables and indexes."""
    with conn.cursor() as cur:
        cur.execute(CREATE_CUSTOMERS)
        cur.execute(CREATE_CHAT_LOGS)
        cur.execute(CREATE_INTENT_RESPONSES)
        cur.execute(CREATE_INDEXES)
    conn.commit()
    print("✅ Tables created successfully.")


def seed_intent_responses(conn):
    """Insert or update all 27 intent response templates."""
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO intent_responses (intent, response_template, updated_at)
            VALUES %s
            ON CONFLICT (intent) DO UPDATE SET
                response_template = EXCLUDED.response_template,
                updated_at = NOW()
            """,
            [(intent, response, "NOW()") for intent, response in INTENT_RESPONSES_SEED],
            template="(%s, %s, NOW())",
        )
    conn.commit()
    print(f"✅ Seeded {len(INTENT_RESPONSES_SEED)} intent responses.")


def seed_demo_customer(conn):
    """Insert a demo customer for testing."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO customers (customer_id, name, email)
            VALUES ('demo_user', 'Demo User', 'demo@example.com')
            ON CONFLICT (customer_id) DO NOTHING
        """)
    conn.commit()
    print("✅ Demo customer created.")


# ── CLI Entry Point ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Setup PostgreSQL schema for Support Chatbot")
    parser.add_argument("--seed", action="store_true", help="Seed intent responses + demo customer")
    args = parser.parse_args()

    conn = get_connection()
    try:
        create_tables(conn)
        if args.seed:
            seed_intent_responses(conn)
            seed_demo_customer(conn)
        print("\nDone. Database is ready.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
