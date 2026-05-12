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
        result = {
            "status": "skipped",
            "reason": f"torch/torchvision unavailable: {exc}",
            "next_step": "Install with: pip install torch torchvision onnx"
        }
        (REPORT_DIR / "onnx_export_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    model = models.mobilenet_v2(weights=None)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = MODEL_DIR / "mobilenet_v2.onnx"

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=12,
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
        )

        result = {
            "status": "completed",
            "model": "MobileNetV2",
            "onnx_path": str(onnx_path),
            "input_shape": [1, 3, 224, 224],
            "opset_version": 12,
            "dynamic_batching": True,
            "note": "ONNX export completed. The model uses random weights and is intended for deployment benchmarking."
        }

    except Exception as exc:
        result = {
            "status": "failed",
            "reason": str(exc)
        }

    (REPORT_DIR / "onnx_export_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    print(json.dumps(export_mobilenet_onnx(), indent=2))
