import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.gpu_scheduler import evaluate_memory_fit
from src.rl_optimizer import simulate_objective_bandit
from src.cloud_carbon import get_static_carbon_intensity


def test_gpu_memory_decision():
    decision = evaluate_memory_fit("NVIDIA T4", model_size_mb=44, batch_size=8)
    assert decision.estimated_memory_gb > 0


def test_rl_optimizer_runs():
    scores = {
        "Balanced": 1.0,
        "Latency": 1.3,
        "Cost": 0.8,
        "Carbon": 0.7,
        "Energy": 0.9,
    }
    trace, values, counts = simulate_objective_bandit(scores, iterations=10)
    assert len(trace) == 10


def test_static_carbon_signal():
    signal = get_static_carbon_intensity("Germany")
    assert signal.carbon_intensity_gco2_kwh > 0