import json
import re
from collections import Counter

class WordTokenizer:
    def __init__(self, vocab=None, unk_token="<UNK>", pad_token="<PAD>"):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.vocab = vocab or {pad_token: 0, unk_token: 1}

    def build_vocab(self, texts, max_vocab=20000, min_freq=2):
        words = []
        for t in texts:
            words.extend(self._tokenize(t))
        counts = Counter(words)
        for w, c in counts.most_common(max_vocab):
            if c < min_freq:
                continue
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)

    def encode(self, text, max_len=40):
        tokens = self._tokenize(text)[:max_len]
        ids = [self.vocab.get(tok, self.vocab[self.unk_token]) for tok in tokens]
        if len(ids) < max_len:
            ids += [self.vocab[self.pad_token]] * (max_len - len(ids))
        return ids

    def _tokenize(self, text):
        text = text.lower().strip()
        return re.findall(r"[a-z0-9]+", text)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)