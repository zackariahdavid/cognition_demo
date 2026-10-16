import json
import time
from collections import deque


class WorkingMemory:
    def __init__(self, capacity=7):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, item):
        self.buffer.append({"time": time.time(), "item": item})

    def get(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()


class LongTermMemory:
    def __init__(self):
        self.concepts = {}
        self.episodic = []

    def update_concept(self, key, delta=1):
        self.concepts[key] = self.concepts.get(key, 0) + delta

    def get_relevance(self, key):
        return self.concepts.get(key, 0)

    def store_episode(self, episode):
        self.episodic.append({"time": time.time(), "episode": episode})

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"concepts": self.concepts, "episodic": self.episodic}, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.concepts = data.get("concepts", {})
        self.episodic = data.get("episodic", [])


class SemanticMemoryVectorStore:
    def __init__(self, dim=32):
        self.dim = dim
        self.vectors = []

    def add(self, key, vec):
        arr = [float(v) for v in vec]
        if len(arr) < self.dim:
            arr += [0.0] * (self.dim - len(arr))
        elif len(arr) > self.dim:
            arr = arr[:self.dim]
        self.vectors.append((key, arr))

    def query(self, vec, topk=3):
        target = [float(v) for v in vec]
        if len(target) < self.dim:
            target += [0.0] * (self.dim - len(target))
        elif len(target) > self.dim:
            target = target[:self.dim]

        results = []
        for key, stored in self.vectors:
            dot = sum(a * b for a, b in zip(target, stored))
            norm = (sum(v * v for v in target) ** 0.5) * (sum(v * v for v in stored) ** 0.5 + 1e-9)
            sim = dot / (norm + 1e-9)
            results.append((sim, key))
        results.sort(reverse=True)
        return results[:topk]
