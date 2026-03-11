"""
Train IntentBiGRU model on processed JSONL data.

Key features:
- Class weights computed from training set (handles intent imbalance)
- Early stopping on best validation accuracy
- Saves model + tokenizer + config to artifacts/
"""

import os, json, sys
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── project imports ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.supportbot.ml.model import IntentBiGRU
from src.supportbot.ml.tokenizer import WordTokenizer

# ── hyperparameters ──────────────────────────────────────────────
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts/intent_model")

MAX_LEN  = 40
BATCH    = 64
EPOCHS   = 10
EMBED    = 128
HIDDEN   = 128
LR       = 1e-3


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compute_class_weights(rows: list[dict], label2id: dict) -> torch.Tensor:
    """Inverse-frequency class weights so rare intents get higher loss."""
    counts = Counter(r["label"] for r in rows)
    total = sum(counts.values())
    num_classes = len(label2id)
    weights = torch.zeros(num_classes)
    for label, idx in label2id.items():
        freq = counts.get(label, 1)
        weights[idx] = total / (num_classes * freq)
    return weights


class IntentDataset(Dataset):
    def __init__(self, rows, tok, label2id):
        self.rows = rows
        self.tok = tok
        self.label2id = label2id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        x = torch.tensor(self.tok.encode(r["text"], MAX_LEN), dtype=torch.long)
        y = torch.tensor(self.label2id[r["label"]], dtype=torch.long)
        return x, y


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── load data ────────────────────────────────────────────────
    train  = read_jsonl(f"{PROCESSED_DIR}/train.jsonl")
    val    = read_jsonl(f"{PROCESSED_DIR}/val.jsonl")
    labels = json.load(open(f"{PROCESSED_DIR}/labels.json", "r", encoding="utf-8"))

    label2id = {l: i for i, l in enumerate(labels)}

    print(f"Train: {len(train)}  |  Val: {len(val)}  |  Classes: {len(labels)}")

    # ── tokenizer ────────────────────────────────────────────────
    tok = WordTokenizer()
    tok.build_vocab([r["text"] for r in train])
    print(f"Vocab size: {len(tok.vocab)}")

    # ── class weights ────────────────────────────────────────────
    class_weights = compute_class_weights(train, label2id)
    print(f"Class weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")

    # ── dataloaders ──────────────────────────────────────────────
    train_ds = IntentDataset(train, tok, label2id)
    val_ds   = IntentDataset(val, tok, label2id)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH)

    # ── model ────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IntentBiGRU(
        vocab_size=len(tok.vocab),
        embed_dim=EMBED,
        hidden_dim=HIDDEN,
        num_classes=len(labels),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}  |  Device: {device}")

    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── training loop ────────────────────────────────────────────
    best_val = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)

        # — validate —
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / max(total, 1)
        marker = ""

        if acc > best_val:
            best_val = acc
            marker = "  ← best"
            torch.save(model.state_dict(), f"{ARTIFACTS_DIR}/model.pt")
            tok.save(f"{ARTIFACTS_DIR}/tokenizer.json")
            json.dump(labels, open(f"{ARTIFACTS_DIR}/labels.json", "w", encoding="utf-8"), indent=2)
            json.dump(
                {"max_len": MAX_LEN, "embed": EMBED, "hidden": HIDDEN},
                open(f"{ARTIFACTS_DIR}/config.json", "w", encoding="utf-8"), indent=2,
            )

        print(f"Epoch {epoch:02d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={acc:.4f}{marker}")

    print(f"\nTraining complete. Best val accuracy: {best_val:.4f}")
    print(f"Artifacts saved to: {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
