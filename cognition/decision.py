import random


class DecisionMaker:
    def __init__(self, rl_agent=None):
        self.rl_agent = rl_agent

    def select_action(self, state_repr, possible_actions, emotion_state, working_memory):
        if self.rl_agent:
            q_action = self.rl_agent.best_action(state_repr, possible_actions)
            explore_rate = 0.1 + 0.3 * emotion_state.arousal
            if random.random() < explore_rate:
                return random.choice(possible_actions)
            return q_action

        scores = []
        wm_size = len(working_memory.get())
        for action in possible_actions:
            score = 0.0
            if any(k in action for k in ["food", "eat"]):
                score += 1.0 * (1 + emotion_state.valence)
            if any(k in action for k in ["danger", "avoid", "move", "threat"]):
                score += 1.5 * (1 + emotion_state.arousal)
            if any(k in action for k in ["social", "greet"]):
                score += 0.8 * (1 + emotion_state.valence)
            if any(k in action for k in ["rest", "relax"]):
                score += 0.7 * (1 + max(0.0, emotion_state.valence))
            score += 0.05 * wm_size
            scores.append((score, action))

        scores.sort(reverse=True)
        return scores[0][1] if scores else random.choice(possible_actions)
