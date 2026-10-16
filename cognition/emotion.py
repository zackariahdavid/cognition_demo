class EmotionState:
    def __init__(self, valence=0.0, arousal=0.0):
        self.valence = float(valence)
        self.arousal = float(arousal)

    def update_from_outcome(self, outcome):
        outcome = str(outcome).lower()
        if "success" in outcome or "food" in outcome or "win" in outcome:
            self.valence = min(1.0, self.valence + 0.1)
            self.arousal = max(0.0, self.arousal - 0.05)
        if "danger" in outcome or "fail" in outcome or "pain" in outcome or "hit" in outcome:
            self.valence = max(-1.0, self.valence - 0.2)
            self.arousal = min(1.0, self.arousal + 0.2)

    def decay(self, amount=0.01):
        if self.valence > 0:
            self.valence = max(0.0, self.valence - amount)
        else:
            self.valence = min(0.0, self.valence + amount)
        self.arousal = max(0.0, self.arousal - amount)
