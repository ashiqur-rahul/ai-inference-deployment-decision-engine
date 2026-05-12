from src.presets import SCENARIO_PRESETS, get_preset
from src.explainability import recommendation_trace
import pandas as pd


def test_presets_exist():
    assert "Low-carbon mode" in SCENARIO_PRESETS
    assert get_preset("Low-carbon mode")["optimize_for"] == "Carbon"


def test_recommendation_trace_counts():
    df = pd.DataFrame({
        "meets_latency": [True, False, True],
        "meets_accuracy": [True, True, False],
        "meets_cost": [True, True, True],
        "violates_slo": [False, True, False],
    })
    trace = recommendation_trace(df)
    assert "All candidates" in trace["stage"].tolist()
    assert int(trace.loc[trace["stage"] == "All candidates", "remaining"].iloc[0]) == 3
