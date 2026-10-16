import random


class QLearner:
    def __init__(self, alpha=0.3, gamma=0.9):
        self.Q = {}
        self.alpha = alpha
        self.gamma = gamma

    def _key(self, state, action):
        return f"{state}||{action}"

    def update(self, state, action, reward, next_state, next_actions):
        current = self.Q.get(self._key(state, action), 0.0)
        future = max(self.Q.get(self._key(next_state, a), 0.0) for a in next_actions)
        new_value = current + self.alpha * (reward + self.gamma * future - current)
        self.Q[self._key(state, action)] = new_value

    def best_action(self, state, actions):
        best_action = None
        best_value = float("-inf")
        for action in actions:
            value = self.Q.get(self._key(state, action), 0.0)
            if value > best_value:
                best_value = value
                best_action = action
        if best_action is None:
            return random.choice(actions)
        return best_action
