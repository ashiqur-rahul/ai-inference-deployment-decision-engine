from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def export_mobilenet_onnx():
    try:
        import torch
        import torchvision.models as models
    except Exception as exc:
        result = {"status": "skipped", "reason": f"torch/torchvision unavailable: {exc}"}
        (REPORT_DIR / "onnx_export_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    model = models.mobilenet_v2(weights=None).eval()
    dummy = torch.randn(1, 3, 224, 224)
    path = MODEL_DIR / "mobilenet_v2.onnx"
    try:
        torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["output"], opset_version=12, dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
        result = {"status": "completed", "onnx_path": str(path)}
    except Exception as exc:
        result = {"status": "failed", "reason": str(exc)}
    (REPORT_DIR / "onnx_export_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
if __name__ == "__main__":
    print(json.dumps(export_mobilenet_onnx(), indent=2))
