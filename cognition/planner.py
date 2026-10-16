class Planner:
    def __init__(self):
        pass

    def propose_goals(self, attention_probs, long_term_memory):
        sorted_concepts = sorted(attention_probs.items(), key=lambda x: -x[1])
        tendencies = [concept for concept, _ in sorted_concepts[:3]]
        goals = []

        for concept in tendencies:
            if concept == "food":
                goals.append({"goal": "acquire_food", "priority": attention_probs.get("food", 0)})
            elif concept == "danger":
                goals.append({"goal": "avoid_danger", "priority": attention_probs.get("danger", 0)})
            elif concept == "social":
                goals.append({"goal": "socialize", "priority": attention_probs.get("social", 0)})
            elif concept == "rest":
                goals.append({"goal": "rest", "priority": attention_probs.get("rest", 0)})
            else:
                goals.append({"goal": "explore", "priority": attention_probs.get(concept, 0)})

        if not goals:
            goals.append({"goal": "explore", "priority": 1.0})

        return goals

    def plan_for_goal(self, goal):
        goal_name = goal.get("goal")

        if goal_name == "acquire_food":
            return ["scan", "approach_food", "eat"]
        if goal_name == "avoid_danger":
            return ["scan", "locate_threat", "move_away"]
        if goal_name == "socialize":
            return ["scan", "approach_person", "greet"]
        if goal_name == "rest":
            return ["scan", "find_seat", "sit_relax"]
        return ["observe"]
