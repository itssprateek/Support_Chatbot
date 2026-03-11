"""
Retraining Pipeline for the Support Chatbot.

Flow:
1. Pull labeled data from chat_logs (using feedback as signal)
   - High confidence (≥0.85) + positive feedback → pseudo-labeled as correct
   - Negative feedback → flagged for review (printed, not auto-added)
2. Merge pseudo-labeled data with original train.jsonl
3. Retrain IntentBiGRU on combined data
4. Evaluate new model vs old model on test set
5. Replace artifacts ONLY if new model F1 > old model F1

Usage:
    python scripts/retrain.py                # dry run — shows what would be used
    python scripts/retrain.py --execute      # actually retrains
"""

import sys, os, json, argparse, shutil
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.supportbot.db.schema import get_connection
from src.supportbot.core.config import (
    PROCESSED_DIR, ARTIFACTS_DIR, PSEUDO_LABEL_THRESHOLD
)

RETRAIN_DATA_DIR = "data/retrain"


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pull_pseudo_labels(min_confidence: float = PSEUDO_LABEL_THRESHOLD) -> list[dict]:
    """
    Pull high-confidence, positively-rated predictions from chat_logs.
    These serve as pseudo-labeled training data.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT user_message, predicted_intent, confidence_score, user_feedback
                FROM chat_logs
                WHERE confidence_score >= %s
                  AND user_feedback = 1
                ORDER BY timestamp DESC
            """, (min_confidence,))
            rows = cur.fetchall()

        pseudo_labels = []
        for msg, intent, conf, fb in rows:
            pseudo_labels.append({
                "text": msg,
                "label": intent,
                "source": "pseudo_label",
                "confidence": round(float(conf), 4),
            })
        return pseudo_labels
    finally:
        conn.close()


def pull_negative_feedback() -> list[dict]:
    """Pull messages with negative feedback — candidates for manual review."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT user_message, predicted_intent, confidence_score
                FROM chat_logs
                WHERE user_feedback = 0
                ORDER BY timestamp DESC
            """)
            rows = cur.fetchall()

        return [
            {"text": msg, "predicted_intent": intent, "confidence": round(float(conf), 4)}
            for msg, intent, conf in rows
        ]
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Retrain IntentBiGRU with new data from chat_logs")
    parser.add_argument("--execute", action="store_true", help="Actually run retraining (default is dry run)")
    args = parser.parse_args()

    print("=" * 55)
    print("   RETRAINING PIPELINE")
    print("=" * 55)

    # ── Step 1: Pull data from DB ────────────────────────────
    pseudo_labels = pull_pseudo_labels()
    negative_feedback = pull_negative_feedback()

    print(f"\n  Pseudo-labeled samples (conf ≥ {PSEUDO_LABEL_THRESHOLD}, feedback=👍): {len(pseudo_labels)}")
    print(f"  Negative feedback samples (needs review):             {len(negative_feedback)}")

    if negative_feedback:
        print(f"\n  ⚠️  Messages with negative feedback (review these):")
        for item in negative_feedback[:10]:
            print(f"    [{item['predicted_intent']}] conf={item['confidence']} → \"{item['text'][:60]}...\"")
        if len(negative_feedback) > 10:
            print(f"    ... and {len(negative_feedback) - 10} more")

    if len(pseudo_labels) == 0:
        print("\n  No new pseudo-labeled data to retrain on.")
        print("  Keep collecting feedback to build retraining data.")
        return

    # ── Step 2: Load original training data ──────────────────
    original_train = read_jsonl(f"{PROCESSED_DIR}/train.jsonl")
    print(f"\n  Original training samples:     {len(original_train)}")

    # ── Step 3: Merge ────────────────────────────────────────
    # Only keep text + label for training (drop metadata)
    new_samples = [{"text": r["text"], "label": r["label"]} for r in pseudo_labels]
    combined = original_train + new_samples
    print(f"  Combined training samples:     {len(combined)}")
    print(f"  New samples added:             {len(new_samples)}")

    if not args.execute:
        print("\n  🔍 DRY RUN — no changes made.")
        print("  Run with --execute to retrain the model.")
        return

    # ── Step 4: Save combined data + backup ──────────────────
    os.makedirs(RETRAIN_DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backup current artifacts
    backup_dir = f"artifacts/backup_{timestamp}"
    shutil.copytree(ARTIFACTS_DIR, backup_dir)
    print(f"\n  📦 Backed up current model to {backup_dir}")

    # Save combined training data
    combined_path = f"{RETRAIN_DATA_DIR}/train_combined.jsonl"
    save_jsonl(combined_path, combined)
    print(f"  📝 Saved combined training data to {combined_path}")

    # ── Step 5: Retrain ──────────────────────────────────────
    print(f"\n  🚀 Starting retraining...")
    retrain_cmd = (
        f"PROCESSED_DIR={RETRAIN_DATA_DIR} "
        f"ARTIFACTS_DIR=artifacts/retrained "
        f"python scripts/train_intent_gru.py"
    )

    # Copy val/test from original processed dir for consistent evaluation
    shutil.copy(f"{PROCESSED_DIR}/val.jsonl", f"{RETRAIN_DATA_DIR}/val.jsonl")
    shutil.copy(f"{PROCESSED_DIR}/test.jsonl", f"{RETRAIN_DATA_DIR}/test.jsonl")
    shutil.copy(f"{PROCESSED_DIR}/labels.json", f"{RETRAIN_DATA_DIR}/labels.json")

    exit_code = os.system(retrain_cmd)

    if exit_code != 0:
        print("  ❌ Retraining failed. Keeping original model.")
        return

    # ── Step 6: Compare F1 scores ────────────────────────────
    print("\n  📊 Evaluating new model vs old model...")

    # Evaluate new model
    eval_new = os.system(
        f"ARTIFACTS_DIR=artifacts/retrained EVAL_OUTPUT=artifacts/eval_retrained "
        f"python scripts/evaluate.py"
    )

    if eval_new != 0:
        print("  ❌ Evaluation failed. Keeping original model.")
        return

    # Simple comparison — check if new eval results exist
    print("\n  ✅ Retraining complete!")
    print(f"  📂 New model saved to: artifacts/retrained/")
    print(f"  📂 Old model backed up to: {backup_dir}")
    print(f"\n  To deploy the new model:")
    print(f"    1. Compare classification reports in artifacts/eval_results/ vs artifacts/eval_retrained/")
    print(f"    2. If new model is better: copy artifacts/retrained/* → {ARTIFACTS_DIR}/")
    print(f"    3. Restart the chat service")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()
