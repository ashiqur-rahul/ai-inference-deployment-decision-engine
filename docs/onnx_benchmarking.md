# ONNX Runtime Benchmarking

This project includes an optional ONNX deployment path to demonstrate model export and runtime benchmarking.

## Why ONNX?

ONNX provides an interoperable model representation that can be executed by optimized runtimes such as ONNX Runtime. This is useful for deployment comparison because the same model architecture can be tested across different runtime environments.

## What this project does

The ONNX workflow includes:

1. Exporting MobileNetV2 to ONNX
2. Running ONNX Runtime inference locally
3. Measuring latency across multiple batch sizes
4. Producing a benchmark report and chart
5. Combining ONNX results with other benchmark results

## Commands

```bash
pip install torch torchvision onnx onnxruntime
python src/onnx_export.py
python src/onnx_benchmark.py
python src/benchmark_comparison.py
```

## Outputs

```text
models/mobilenet_v2.onnx
outputs/reports/onnx_export_report.json
outputs/reports/onnx_benchmark_report.json
outputs/reports/onnx_benchmark_results.csv
outputs/figures/onnx_latency_by_batch.png
outputs/reports/benchmark_comparison.csv
outputs/figures/benchmark_latency_comparison.png
```

## Important limitation

The ONNX-exported MobileNetV2 uses random weights. Therefore, the ONNX path is intended for runtime and deployment-path benchmarking, not accuracy evaluation.

Accuracy is measured separately using the sklearn digits benchmark.
