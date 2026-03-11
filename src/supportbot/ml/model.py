import torch
import torch.nn as nn

class IntentBiGRU(nn.Module):
    """Bidirectional GRU for intent classification.
    
    Architecture: Embedding → BiGRU → Dropout → Linear
    Uses last hidden state from both directions (hidden_dim * 2).
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        last = out[:, -1, :]  # last timestep, both directions
        return self.fc(self.dropout(last))