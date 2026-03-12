import json
import torch
import torch.nn.functional as F

from src.supportbot.ml.model import IntentBiGRU
from src.supportbot.ml.tokenizer import WordTokenizer

class IntentPredictor:
    def __init__(self, artifacts_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.labels = json.load(open(f"{artifacts_dir}/labels.json", "r", encoding="utf-8"))
        self.id2label = {i:l for i,l in enumerate(self.labels)}
        self.config = json.load(open(f"{artifacts_dir}/config.json", "r", encoding="utf-8"))
        self.tokenizer = WordTokenizer.load(f"{artifacts_dir}/tokenizer.json")

        self.model = IntentBiGRU(
            vocab_size=len(self.tokenizer.vocab),
            embed_dim=self.config["embed"],
            hidden_dim=self.config["hidden"],
            num_classes=len(self.labels)
        ).to(self.device)

        self.model.load_state_dict(torch.load(f"{artifacts_dir}/model.pt", map_location=self.device))
        self.model.eval()

    def predict(self, text: str):
        x = torch.tensor([self.tokenizer.encode(text, self.config["max_len"])], dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        return self.id2label[idx], float(probs[idx])