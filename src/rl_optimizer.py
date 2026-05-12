from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict

ACTIONS = ["Balanced", "Latency", "Cost", "Carbon", "Energy"]

@dataclass
class EpsilonGreedyOptimizer:
    epsilon: float = 0.15
    counts: Dict[str, int] = field(default_factory=lambda: {a: 0 for a in ACTIONS})
    values: Dict[str, float] = field(default_factory=lambda: {a: 0.0 for a in ACTIONS})

    def select_action(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return max(self.values, key=self.values.get)

    def update(self, action: str, reward: float) -> None:
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n

def reward_from_score(score: float) -> float:
    return -float(score)

def simulate_objective_bandit(objective_scores: Dict[str, float], iterations: int = 100, epsilon: float = 0.15):
    opt = EpsilonGreedyOptimizer(epsilon=epsilon)
    trace = []
    for i in range(iterations):
        action = opt.select_action()
        score = objective_scores[action]
        reward = reward_from_score(score)
        opt.update(action, reward)
        trace.append({"iteration": i + 1, "action": action, "score": score, "reward": reward, "estimated_value": opt.values[action]})
    return trace, opt.values, opt.counts
