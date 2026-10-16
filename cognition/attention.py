import numpy as np


class AttentionSystem:
    def __init__(self):
        self.base_salience = {
            "food": 1.0,
            "danger": 1.0,
            "social": 0.8,
            "rest": 0.6,
            "curiosity": 0.5,
        }

    def compute_salience(self, percept_vector, long_term_memory, emotion_state):
        salience = dict(self.base_salience)

        for key, value in long_term_memory.concepts.items():
            salience[key] = salience.get(key, 0.1) + 0.1 * value

        for key in salience:
            salience[key] *= (1 + 0.2 * emotion_state.valence) * (1 + 0.1 * emotion_state.arousal)

        if isinstance(percept_vector, dict):
            for token in percept_vector:
                low = str(token).lower()
                if low in ["hungry", "eat", "food", "plate", "snack", "meal"]:
                    salience["food"] += 0.5
                if low in ["bang", "loud", "threat", "dark", "danger", "alarm"]:
                    salience["danger"] += 0.7
                if low in ["hello", "friend", "smile", "talk", "greet", "message"]:
                    salience["social"] += 0.4
                if low in ["tired", "sleepy", "bench", "quiet", "rest"]:
                    salience["rest"] += 0.4
                if low in ["curious", "novel", "interesting", "new", "discover"]:
                    salience["curiosity"] += 0.5
        else:
            try:
                magnitude = float(np.linalg.norm(percept_vector))
                salience["curiosity"] += 0.1 * magnitude
            except Exception:
                pass

        total = sum(salience.values()) + 1e-9
        return {k: v / total for k, v in salience.items()}
