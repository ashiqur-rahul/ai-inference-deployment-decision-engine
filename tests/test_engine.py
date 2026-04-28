import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from data_loader import load_data
from engine import evaluate_all, recommend
def test_recommendation_exists():
    data = load_data()
    results = evaluate_all(data)
    rec = recommend(results)
    assert "hardware" in rec
    assert "strategy" in rec
    assert len(results) > 0
def test_results_have_expected_columns():
    data = load_data()
    results = evaluate_all(data)
    expected = {"latency_ms", "estimated_accuracy", "facility_energy_kwh", "monthly_cost_usd", "score"}
    assert expected.issubset(set(results.columns))
