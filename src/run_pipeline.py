from pathlib import Path
import sys, json
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from data_loader import load_data
from engine import evaluate_all, recommend
FIG = ROOT / "outputs" / "figures"
REP = ROOT / "outputs" / "reports"
FIG.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

def main():
    data = load_data()
    measured_latency = measured_accuracy = None
    summary_path = REP / "accuracy_benchmark_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("status") == "completed":
            measured_latency = summary.get("best_mean_latency_ms")
            measured_accuracy = summary.get("best_accuracy")
    results = evaluate_all(data, measured_latency_ms=measured_latency, measured_accuracy=measured_accuracy)
    results.to_csv(REP / "deployment_results.csv", index=False)
    rec = recommend(results)
    (REP / "recommendation.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    (REP / "decision_report.md").write_text(f"# Deployment Decision Report\n\nRecommended: **{rec['strategy']}** on **{rec['hardware']}**\n\n{rec['reason']}\n", encoding="utf-8")
    plt.figure(figsize=(9,6)); plt.scatter(results.latency_ms, results.energy_per_1000_inferences_wh); plt.xlabel("Latency (ms)"); plt.ylabel("Wh per 1,000 inferences"); plt.title("Latency vs Energy Trade-off"); plt.tight_layout(); plt.savefig(FIG/"latency_energy_tradeoff.png", dpi=220); plt.close()
    plt.figure(figsize=(9,6)); plt.scatter(results.estimated_accuracy, results.energy_per_1000_inferences_wh); plt.xlabel("Estimated accuracy"); plt.ylabel("Wh per 1,000 inferences"); plt.title("Accuracy vs Energy Trade-off"); plt.tight_layout(); plt.savefig(FIG/"accuracy_energy_tradeoff.png", dpi=220); plt.close()
    top = results.head(12)
    plt.figure(figsize=(11,6)); plt.bar(top.strategy, top.facility_energy_kwh); plt.xticks(rotation=35, ha="right"); plt.ylabel("kWh/month"); plt.title("Top Configurations by Facility Energy"); plt.tight_layout(); plt.savefig(FIG/"energy_comparison.png", dpi=220); plt.close()
    plt.figure(figsize=(11,6)); plt.bar(top.strategy, top.monthly_cost_usd); plt.xticks(rotation=35, ha="right"); plt.ylabel("USD/month"); plt.title("Top Configurations by Monthly Cost"); plt.tight_layout(); plt.savefig(FIG/"cost_comparison.png", dpi=220); plt.close()
    print("Pipeline completed successfully.")
if __name__ == "__main__":
    main()
