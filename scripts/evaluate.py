"""
Evaluate the trained IntentBiGRU model on the test set.

Produces:
- Per-intent precision, recall, F1 (classification report)
- Macro & weighted F1 scores
- Confusion matrix saved as PNG
- Top confused intent pairs
"""

import os, json, sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.supportbot.ml.model import IntentBiGRU
from src.supportbot.ml.tokenizer import WordTokenizer

# Optional — sklearn for metrics, matplotlib for confusion matrix
try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# ── config ───────────────────────────────────────────────────────
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts/intent_model")
EVAL_OUTPUT   = os.getenv("EVAL_OUTPUT", "artifacts/eval_results")


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_model(artifacts_dir: str, device: str):
    """Load trained model, tokenizer, labels from artifacts."""
    labels = json.load(open(f"{artifacts_dir}/labels.json", "r", encoding="utf-8"))
    config = json.load(open(f"{artifacts_dir}/config.json", "r", encoding="utf-8"))
    tok    = WordTokenizer.load(f"{artifacts_dir}/tokenizer.json")

    model = IntentBiGRU(
        vocab_size=len(tok.vocab),
        embed_dim=config["embed"],
        hidden_dim=config["hidden"],
        num_classes=len(labels),
    ).to(device)

    model.load_state_dict(torch.load(f"{artifacts_dir}/model.pt", map_location=device))
    model.eval()
    return model, tok, labels, config


def predict_batch(model, tok, texts: list[str], max_len: int, device: str):
    """Run inference on a list of texts, return predicted labels + confidences."""
    encoded = [tok.encode(t, max_len) for t in texts]
    x = torch.tensor(encoded, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=1).cpu().numpy()

    pred_ids     = probs.argmax(axis=1)
    confidences  = probs[np.arange(len(probs)), pred_ids]
    return pred_ids, confidences


def find_top_confusions(cm, labels, top_n=5):
    """Find the most confused intent pairs from the confusion matrix."""
    confusions = []
    n = len(labels)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                confusions.append((labels[i], labels[j], int(cm[i][j])))
    confusions.sort(key=lambda x: x[2], reverse=True)
    return confusions[:top_n]


def main():
    os.makedirs(EVAL_OUTPUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── load ─────────────────────────────────────────────────────
    model, tok, labels, config = load_model(ARTIFACTS_DIR, device)
    test_data = read_jsonl(f"{PROCESSED_DIR}/test.jsonl")
    max_len   = config["max_len"]

    label2id = {l: i for i, l in enumerate(labels)}

    texts      = [r["text"] for r in test_data]
    true_ids   = [label2id[r["label"]] for r in test_data]

    print(f"Evaluating on {len(texts)} test samples  |  {len(labels)} classes")
    print(f"Device: {device}\n")

    # ── predict ──────────────────────────────────────────────────
    CHUNK = 256
    all_preds = []
    all_confs = []

    for i in range(0, len(texts), CHUNK):
        chunk_texts = texts[i : i + CHUNK]
        pids, confs = predict_batch(model, tok, chunk_texts, max_len, device)
        all_preds.extend(pids.tolist())
        all_confs.extend(confs.tolist())

    true_arr = np.array(true_ids)
    pred_arr = np.array(all_preds)
    conf_arr = np.array(all_confs)

    # ── overall accuracy ─────────────────────────────────────────
    accuracy = (true_arr == pred_arr).sum() / len(true_arr)
    avg_conf = conf_arr.mean()
    print(f"Test Accuracy:        {accuracy:.4f}")
    print(f"Avg Confidence:       {avg_conf:.4f}")
    print(f"Confidence Std Dev:   {conf_arr.std():.4f}\n")

    # ── classification report ────────────────────────────────────
    if HAS_SKLEARN:
        true_labels = [labels[i] for i in true_arr]
        pred_labels = [labels[i] for i in pred_arr]

        report = classification_report(true_labels, pred_labels, digits=3, zero_division=0)
        print("Classification Report:")
        print("=" * 70)
        print(report)

        # save to file
        with open(f"{EVAL_OUTPUT}/classification_report.txt", "w") as f:
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Avg Confidence: {avg_conf:.4f}\n\n")
            f.write(report)
        print(f"Report saved to {EVAL_OUTPUT}/classification_report.txt")

        # ── confusion matrix ─────────────────────────────────────
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)

        # top confusions
        top_confused = find_top_confusions(cm, labels)
        print("\nTop Confused Intent Pairs:")
        print("-" * 50)
        for true_l, pred_l, count in top_confused:
            print(f"  {true_l:30s} → {pred_l:30s}  ({count})")

        # save confusion matrix as image
        if HAS_PLT:
            fig, ax = plt.subplots(figsize=(14, 12))
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title("Intent Classification — Confusion Matrix", fontsize=14)
            fig.colorbar(im, ax=ax)

            tick_marks = np.arange(len(labels))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(labels, fontsize=7)

            ax.set_ylabel("True Intent")
            ax.set_xlabel("Predicted Intent")
            plt.tight_layout()

            cm_path = f"{EVAL_OUTPUT}/confusion_matrix.png"
            plt.savefig(cm_path, dpi=150)
            plt.close()
            print(f"\nConfusion matrix saved to {cm_path}")

    else:
        print("⚠ Install scikit-learn for full classification report:")
        print("  pip install scikit-learn")

    # ── confidence distribution by correctness ───────────────────
    correct_mask = (true_arr == pred_arr)
    if correct_mask.sum() > 0:
        print(f"\nConfidence when CORRECT:   {conf_arr[correct_mask].mean():.4f}")
    if (~correct_mask).sum() > 0:
        print(f"Confidence when WRONG:     {conf_arr[~correct_mask].mean():.4f}")

    # ── save raw predictions for further analysis ────────────────
    predictions = []
    for i in range(len(texts)):
        predictions.append({
            "text": texts[i],
            "true_intent": labels[true_ids[i]],
            "pred_intent": labels[all_preds[i]],
            "confidence": round(all_confs[i], 4),
            "correct": bool(true_ids[i] == all_preds[i]),
        })

    with open(f"{EVAL_OUTPUT}/predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\nRaw predictions saved to {EVAL_OUTPUT}/predictions.json")
    print("Done.")


if __name__ == "__main__":
    main()
