import random

from .attention import AttentionSystem
from .decision import DecisionMaker
from .emotion import EmotionState
from .learning import QLearner
from .memory import LongTermMemory, SemanticMemoryVectorStore, WorkingMemory
from .perception import PerceptionModel, SensoryInput
from .planner import Planner


class CognitiveAgent:
    def __init__(self, enable_rl=True):
        self.perceptor = PerceptionModel()
        self.working_memory = WorkingMemory()
        self.long_term_memory = LongTermMemory()
        self.semantic_store = SemanticMemoryVectorStore(dim=32)
        self.attention = AttentionSystem()
        self.emotion = EmotionState()
        self.planner = Planner()
        self.rl = QLearner() if enable_rl else None
        self.decider = DecisionMaker(rl_agent=self.rl)
        self.step_count = 0

    def perceive(self, sensory: SensoryInput):
        text = sensory.text()
        encoded = self.perceptor.encode_text(text)
        self.working_memory.add({"text": text, "vector": encoded})
        self.semantic_store.add(text[:50], encoded)
        return encoded

    def attend(self, encoded):
        return self.attention.compute_salience(encoded, self.long_term_memory, self.emotion)

    def plan(self, attention_probs):
        goals = self.planner.propose_goals(attention_probs, self.long_term_memory)
        if not goals:
            return []
        goal = goals[0]
        plan = self.planner.plan_for_goal(goal)
        self.working_memory.add({"goal": goal, "plan": plan})
        return plan

    def decide_and_act(self, plan):
        if not plan:
            return "observe", "no_effect"
        action = self.decider.select_action(
            self._state_repr(),
            plan,
            self.emotion,
            self.working_memory,
        )
        outcome = self._execute(action)
        self.learn(action, outcome)
        self.emotion.update_from_outcome(outcome)
        self.long_term_memory.store_episode({"action": action, "outcome": outcome})
        self.step_count += 1

        if self.step_count % 5 == 0:
            self.self_reflect()

        return action, outcome

    def _execute(self, action):
        if action in ["approach_food", "eat"]:
            return random.choice(["food_found_success", "food_not_found"])
        if action in ["locate_threat", "move_away"]:
            return random.choice(["danger_avoided", "danger_hit"])
        if action in ["approach_person", "greet"]:
            return random.choice(["social_success", "social_ignored"])
        if action in ["find_seat", "sit_relax"]:
            return "rest_success"
        if action == "scan":
            return random.choice(["observe_clear", "observe_novel"])
        return "no_effect"

    def learn(self, action, outcome):
        reward = 0.0
        if "success" in outcome or "found" in outcome or "avoided" in outcome:
            reward = 1.0
        elif "not_found" in outcome or "hit" in outcome or "ignored" in outcome:
            reward = -0.5

        if "food" in outcome:
            self.long_term_memory.update_concept("food", 1 if reward > 0 else 0)
        if "danger" in outcome:
            self.long_term_memory.update_concept("danger", 1 if reward < 0 else 0)
        if "social" in outcome:
            self.long_term_memory.update_concept("social", 1 if reward > 0 else 0)
        if "rest" in outcome:
            self.long_term_memory.update_concept("rest", 1 if reward > 0 else 0)

        if self.rl:
            state = self._state_repr()
            next_state = state
            next_actions = ["observe", "scan", "approach_food", "eat", "move_away", "greet", "sit_relax"]
            self.rl.update(state, action, reward, next_state, next_actions)

    def self_reflect(self):
        episodes = self.long_term_memory.episodic[-10:]
        successes = sum(1 for e in episodes if "success" in str(e["episode"]).lower() or "found" in str(e["episode"]).lower() or "avoided" in str(e["episode"]).lower())
        ratio = successes / max(1, len(episodes))

        if ratio < 0.3:
            self.emotion.arousal = min(1.0, self.emotion.arousal + 0.2)
            if self.rl:
                self.rl.alpha = min(0.9, self.rl.alpha + 0.05)
        else:
            if self.rl:
                self.rl.alpha = max(0.01, self.rl.alpha - 0.02)

    def _state_repr(self):
        wm = self.working_memory.get()[-2:]
        wm_summary = "|".join(
            str(entry.get("text", entry)) if isinstance(entry, dict) else str(entry)
            for entry in wm
        )
        top_concepts = sorted(self.long_term_memory.concepts.items(), key=lambda x: -x[1])[:3]
        top_summary = ",".join(f"{k}:{v}" for k, v in top_concepts)
        return f"WM[{wm_summary}]||LTM[{top_summary}]"
