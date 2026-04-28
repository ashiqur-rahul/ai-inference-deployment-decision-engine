from pathlib import Path
import json, time, statistics
ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def run_torch_benchmark(iterations=30, warmup=5, batch_size=1):
    try:
        import torch
        import torchvision.models as models
    except Exception as exc:
        result = {"status": "skipped", "reason": f"torch/torchvision unavailable: {exc}", "measured_latency_ms": None}
        (REPORT_DIR / "real_benchmark_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None).to(device).eval()
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device.type == "cuda": torch.cuda.synchronize()
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(x)
            if device.type == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
    result = {"status": "completed", "device": str(device), "model": "MobileNetV2", "mean_latency_ms": statistics.mean(times), "median_latency_ms": statistics.median(times), "min_latency_ms": min(times), "max_latency_ms": max(times)}
    (REPORT_DIR / "real_benchmark_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
if __name__ == "__main__":
    print(json.dumps(run_torch_benchmark(), indent=2))
