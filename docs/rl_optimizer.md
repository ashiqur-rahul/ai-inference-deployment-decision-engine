# RL-Based Optimizer

The project includes a lightweight epsilon-greedy contextual bandit in `src/rl_optimizer.py`.

It demonstrates how a deployment system could learn which optimization objective performs best under repeated workload scenarios.

## Example

```python
from rl_optimizer import simulate_objective_bandit

scores = {
    "Balanced": 1.2,
    "Latency": 1.5,
    "Cost": 0.9,
    "Carbon": 0.8,
    "Energy": 1.0,
}

trace, values, counts = simulate_objective_bandit(scores)
```
