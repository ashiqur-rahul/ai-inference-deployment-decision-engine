from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def create_benchmark_comparison():
    import pandas as pd
    import matplotlib.pyplot as plt

    rows = []

    sklearn_path = REPORT_DIR / "accuracy_benchmark_summary.json"
    if sklearn_path.exists():
        data = json.loads(sklearn_path.read_text())
        if data.get("status") == "completed":
            rows.append({
                "benchmark": "sklearn digits",
                "runtime": data.get("best_model"),
                "mean_latency_ms": data.get("best_mean_latency_ms"),
                "accuracy": data.get("best_accuracy"),
                "notes": "Real accuracy + latency on sklearn digits"
            })

    torch_path = REPORT_DIR / "real_benchmark_report.json"
    if torch_path.exists():
        data = json.loads(torch_path.read_text())
        if data.get("status") == "completed":
            rows.append({
                "benchmark": "PyTorch MobileNetV2",
                "runtime": data.get("device", "local"),
                "mean_latency_ms": data.get("mean_latency_ms"),
                "accuracy": None,
                "notes": "Local timing only"
            })

    onnx_csv = REPORT_DIR / "onnx_benchmark_results.csv"
    if onnx_csv.exists():
        df_onnx = pd.read_csv(onnx_csv)
        best = df_onnx.sort_values("throughput_images_per_second", ascending=False).iloc[0]
        rows.append({
            "benchmark": "ONNX Runtime MobileNetV2",
            "runtime": "CPUExecutionProvider",
            "mean_latency_ms": best["mean_latency_ms"],
            "accuracy": None,
            "notes": f"Best throughput at batch size {int(best['batch_size'])}"
        })

    if not rows:
        result = {
            "status": "skipped",
            "reason": "No benchmark reports found.",
            "next_step": "Run accuracy_benchmark.py, real_benchmark.py, or onnx_benchmark.py first."
        }
        (REPORT_DIR / "benchmark_comparison_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    df = pd.DataFrame(rows)
    csv_path = REPORT_DIR / "benchmark_comparison.csv"
    df.to_csv(csv_path, index=False)

    plot_df = df.dropna(subset=["mean_latency_ms"])
    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["benchmark"], plot_df["mean_latency_ms"])
    plt.ylabel("Mean latency (ms)")
    plt.title("Benchmark Latency Comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig_path = FIG_DIR / "benchmark_latency_comparison.png"
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close()

    result = {
        "status": "completed",
        "csv_path": str(csv_path),
        "figure_path": str(fig_path),
        "benchmarks": rows
    }

    (REPORT_DIR / "benchmark_comparison_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    print(json.dumps(create_benchmark_comparison(), indent=2))
