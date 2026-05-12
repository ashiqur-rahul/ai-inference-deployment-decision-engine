from pathlib import Path
import json
import time
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mobilenet_v2.onnx"
REPORT_DIR = ROOT / "outputs" / "reports"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def benchmark_onnx_runtime(batch_sizes=(1, 4, 8), iterations=50, warmup=10):
    try:
        import onnxruntime as ort
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as exc:
        result = {
            "status": "skipped",
            "reason": f"Required package unavailable: {exc}",
            "next_step": "Install with: pip install onnxruntime pandas matplotlib"
        }
        (REPORT_DIR / "onnx_benchmark_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    if not MODEL_PATH.exists():
        result = {
            "status": "skipped",
            "reason": "ONNX model not found.",
            "next_step": "Run: python src/onnx_export.py"
        }
        (REPORT_DIR / "onnx_benchmark_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    rows = []

    for batch_size in batch_sizes:
        x = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

        for _ in range(warmup):
            session.run(None, {input_name: x})

        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            session.run(None, {input_name: x})
            end = time.perf_counter()
            timings.append((end - start) * 1000)

        mean_latency = statistics.mean(timings)
        median_latency = statistics.median(timings)
        p95_latency = sorted(timings)[int(0.95 * len(timings)) - 1]
        throughput = (batch_size / mean_latency) * 1000

        rows.append({
            "runtime": "ONNX Runtime CPU",
            "model": "MobileNetV2",
            "batch_size": batch_size,
            "iterations": iterations,
            "mean_latency_ms": mean_latency,
            "median_latency_ms": median_latency,
            "p95_latency_ms": p95_latency,
            "throughput_images_per_second": throughput,
        })

    df = pd.DataFrame(rows)
    csv_path = REPORT_DIR / "onnx_benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(df["batch_size"], df["mean_latency_ms"], marker="o")
    plt.xlabel("Batch size")
    plt.ylabel("Mean latency (ms)")
    plt.title("ONNX Runtime Latency by Batch Size")
    plt.tight_layout()
    fig_path = FIG_DIR / "onnx_latency_by_batch.png"
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close()

    best = df.sort_values("throughput_images_per_second", ascending=False).iloc[0].to_dict()

    result = {
        "status": "completed",
        "model": "MobileNetV2",
        "runtime": "ONNX Runtime CPUExecutionProvider",
        "benchmark_rows": rows,
        "best_throughput_configuration": best,
        "csv_path": str(csv_path),
        "figure_path": str(fig_path),
        "note": "This benchmark measures local ONNX Runtime latency. Accuracy is not evaluated here because the exported model uses random weights."
    }

    (REPORT_DIR / "onnx_benchmark_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    print(json.dumps(benchmark_onnx_runtime(), indent=2))
