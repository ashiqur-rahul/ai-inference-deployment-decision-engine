from pathlib import Path
import json, time, statistics
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"
MODEL_PATH = ROOT / "models" / "mobilenet_v2.onnx"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def run_onnx_benchmark(iterations=30, warmup=5):
    try:
        import onnxruntime as ort
    except Exception as exc:
        result = {"status": "skipped", "reason": f"onnxruntime unavailable: {exc}"}
        (REPORT_DIR / "onnx_benchmark_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    if not MODEL_PATH.exists():
        result = {"status": "skipped", "reason": "ONNX model not found. Run python src/onnx_export.py first."}
        (REPORT_DIR / "onnx_benchmark_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    x = np.random.randn(1,3,224,224).astype(np.float32)
    for _ in range(warmup): session.run(None, {input_name: x})
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, {input_name: x})
        times.append((time.perf_counter()-start)*1000)
    result = {"status": "completed", "model": "MobileNetV2 ONNX", "mean_latency_ms": statistics.mean(times), "median_latency_ms": statistics.median(times), "min_latency_ms": min(times), "max_latency_ms": max(times)}
    (REPORT_DIR / "onnx_benchmark_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
if __name__ == "__main__":
    print(json.dumps(run_onnx_benchmark(), indent=2))
