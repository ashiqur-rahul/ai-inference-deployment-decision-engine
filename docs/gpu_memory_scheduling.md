# GPU Memory-Aware Scheduling

The project includes `src/gpu_scheduler.py`.

It estimates whether a model and batch size fit into a target hardware memory profile.

## Example

```python
from gpu_scheduler import evaluate_memory_fit

decision = evaluate_memory_fit(
    hardware="NVIDIA T4",
    model_size_mb=44,
    batch_size=32,
)

print(decision)
```

## Limitation

This is a planning heuristic, not a replacement for real GPU profiling.
