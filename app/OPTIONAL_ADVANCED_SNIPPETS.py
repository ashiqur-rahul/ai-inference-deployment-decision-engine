# Optional Streamlit snippets. Add only if you want these UI cards.

from gpu_scheduler import evaluate_memory_fit
from cloud_carbon import get_best_available_carbon_signal
from rl_optimizer import simulate_objective_bandit

# GPU memory fit
memory_decision = evaluate_memory_fit(
    hardware=best.hardware,
    model_size_mb=best.model_size_mb,
    batch_size=batch_size,
)
st.metric("Estimated Memory", f"{memory_decision.estimated_memory_gb:.2f} GB")
st.metric("Available Memory", f"{memory_decision.available_memory_gb:.1f} GB")
st.write(memory_decision.recommendation)

# Live carbon signal
carbon_signal = get_best_available_carbon_signal(region)
st.metric("Grid Carbon Signal", f"{carbon_signal.carbon_intensity_gco2_kwh:.0f} gCO₂/kWh")
st.caption(f"Source: {carbon_signal.source}")

# RL objective optimizer
objective_scores = {
    "Balanced": 1.2,
    "Latency": 1.5,
    "Cost": 0.9,
    "Carbon": 0.8,
    "Energy": 1.0,
}
trace, values, counts = simulate_objective_bandit(objective_scores)
st.dataframe(trace)
