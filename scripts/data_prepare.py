import os, json, re
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET_NAME = os.getenv("HF_DATASET", "bitext/Bitext-customer-support-llm-chatbot-training-dataset")
OUT_DIR = os.getenv("PROCESSED_DIR", "data/processed")

PLACEHOLDER_PATTERNS = [
    r"\{\{.*?\}\}",          # {{Order Number}}
    r"\{.*?\}",              # {order_id}
    r"<.*?>",                # <ORDER_ID>
]

def clean_text(t: str) -> str:
    t = t.strip()
    for p in PLACEHOLDER_PATTERNS:
        t = re.sub(p, "ORDER_ID", t)
    t = re.sub(r"\s+", " ", t)
    return t

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = load_dataset(DATASET_NAME)
    # choose a split that exists (some datasets use "train" only)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    data = ds[split_name]

    # Build arrays
    texts = []
    labels = []
    for row in data:
        instr = row.get("instruction") or ""
        intent = row.get("intent") or "OTHER"
        instr = clean_text(instr)
        if len(instr) < 3:
            continue
        texts.append(instr)
        labels.append(intent)

    # Stratified splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Save JSONL (small enough; if huge, keep uncommitted)
    def save_jsonl(path, X, y):
        with open(path, "w", encoding="utf-8") as f:
            for t, lab in zip(X, y):
                f.write(json.dumps({"text": t, "label": lab}, ensure_ascii=False) + "\n")

    save_jsonl(os.path.join(OUT_DIR, "train.jsonl"), X_train, y_train)
    save_jsonl(os.path.join(OUT_DIR, "val.jsonl"), X_val, y_val)
    save_jsonl(os.path.join(OUT_DIR, "test.jsonl"), X_test, y_test)

    labels_sorted = sorted(set(labels))
    with open(os.path.join(OUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels_sorted, f, ensure_ascii=False, indent=2)

    print("Saved:", OUT_DIR)
    print("Train/Val/Test:", len(X_train), len(X_val), len(X_test))
    print("Num labels:", len(labels_sorted))

if __name__ == "__main__":
    main()