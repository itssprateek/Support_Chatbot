"""
Weekly Model Health Report — queries chat_logs in PostgreSQL.

Prints:
- Total conversations
- High confidence vs rephrase rate
- Average confidence score + 7-day trend
- Top intents this week
- Low-confidence intents (candidates for retraining)
- CSAT score from user feedback

Usage:
    python scripts/analytics.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.supportbot.db.schema import get_connection


def run_report():
    conn = get_connection()
    cur = conn.cursor()

    print("=" * 55)
    print("   WEEKLY MODEL HEALTH REPORT")
    print("=" * 55)

    # ── Total conversations ──────────────────────────────────
    cur.execute("SELECT COUNT(*) FROM chat_logs")
    total = cur.fetchone()[0]
    print(f"\n  Total conversations:          {total}")

    if total == 0:
        print("\n  No data yet. Start chatting to generate logs!")
        conn.close()
        return

    # ── High confidence vs rephrase ──────────────────────────
    cur.execute("SELECT COUNT(*) FROM chat_logs WHERE route = 'high_conf'")
    high_conf = cur.fetchone()[0]
    rephrase = total - high_conf
    high_pct = (high_conf / total) * 100

    print(f"  High confidence responses:    {high_conf:>5}  ({high_pct:.1f}%)")
    print(f"  Low confidence (rephrase):    {rephrase:>5}  ({100-high_pct:.1f}%)")

    # ── Confidence scores ────────────────────────────────────
    cur.execute("SELECT AVG(confidence_score), STDDEV(confidence_score) FROM chat_logs")
    avg_conf, std_conf = cur.fetchone()
    avg_conf = avg_conf or 0
    std_conf = std_conf or 0
    print(f"\n  Avg confidence score:         {avg_conf:.4f}")
    print(f"  Confidence std dev:           {std_conf:.4f}")

    # ── 7-day rolling average ────────────────────────────────
    cur.execute("""
        SELECT AVG(confidence_score)
        FROM chat_logs
        WHERE timestamp >= NOW() - INTERVAL '7 days'
    """)
    rolling_avg = cur.fetchone()[0]
    if rolling_avg:
        trend = "↑" if rolling_avg >= avg_conf else "↓"
        print(f"  7-day rolling avg:            {rolling_avg:.4f}  {trend}")

        if rolling_avg < 0.70:
            print("  ⚠️  Confidence trending low — consider retraining")
    else:
        print("  7-day rolling avg:            N/A (no recent data)")

    # ── Top intents ──────────────────────────────────────────
    cur.execute("""
        SELECT predicted_intent, COUNT(*) as cnt
        FROM chat_logs
        GROUP BY predicted_intent
        ORDER BY cnt DESC
        LIMIT 5
    """)
    rows = cur.fetchall()
    print(f"\n  Top 5 intents:")
    for intent, count in rows:
        pct = (count / total) * 100
        print(f"    {intent:<30s} {count:>5}  ({pct:.1f}%)")

    # ── Low confidence intents ───────────────────────────────
    cur.execute("""
        SELECT predicted_intent, AVG(confidence_score) as avg_conf, COUNT(*) as cnt
        FROM chat_logs
        GROUP BY predicted_intent
        HAVING AVG(confidence_score) < 0.70
        ORDER BY avg_conf ASC
        LIMIT 5
    """)
    rows = cur.fetchall()
    if rows:
        print(f"\n  ⚠️  Low-confidence intents (retrain candidates):")
        for intent, avg_c, count in rows:
            print(f"    {intent:<30s} avg_conf={avg_c:.3f}  n={count}")

    # ── CSAT score ───────────────────────────────────────────
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE user_feedback IS NOT NULL) as total_feedback,
            COUNT(*) FILTER (WHERE user_feedback = 1) as positive,
            COUNT(*) FILTER (WHERE user_feedback = 0) as negative
        FROM chat_logs
    """)
    total_fb, positive, negative = cur.fetchone()

    print(f"\n  Feedback collected:           {total_fb}")
    if total_fb > 0:
        csat = (positive / total_fb) * 100
        print(f"  CSAT score:                   {csat:.1f}%  ({positive}👍 / {negative}👎)")
    else:
        print("  CSAT score:                   N/A (no feedback yet)")

    print("\n" + "=" * 55)

    cur.close()
    conn.close()


if __name__ == "__main__":
    run_report()
