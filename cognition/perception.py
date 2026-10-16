import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class SensoryInput:
    def __init__(self, vision="", sound="", touch="", context=""):
        self.vision = vision
        self.sound = sound
        self.touch = touch
        self.context = context

    def text(self):
        parts = [self.vision, self.sound, self.touch, self.context]
        return " | ".join(p for p in parts if p)


if TORCH_AVAILABLE:
    class TinyPerceptionNN(nn.Module):
        def __init__(self, vocab_size=100, emb_dim=32, hidden=64, out_dim=32):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb_dim)
            self.network = nn.Sequential(
                nn.Linear(emb_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x):
            x = self.embedding(x).mean(dim=1)
            return self.network(x)

    class PerceptionModel:
        def __init__(self, vocab=None):
            self.vocab = vocab or {}
            self.model = TinyPerceptionNN(vocab_size=max(2, len(self.vocab) + 1))

        def encode_text(self, text):
            tokens = []
            for token in text.lower().split():
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab) + 1
                tokens.append(self.vocab[token])
            if not tokens:
                return np.zeros(32, dtype=float)
            x = torch.tensor([tokens], dtype=torch.long)
            with torch.no_grad():
                vec = self.model(x).numpy()[0]
            return vec
else:
    class PerceptionModel:
        def __init__(self, vocab=None):
            self.vocab = vocab or {}

        def encode_text(self, text):
            tokens = text.lower().split()
            vec = np.zeros(32, dtype=float)
            for token in tokens:
                idx = abs(hash(token)) % 32
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
