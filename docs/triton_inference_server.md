# Triton Inference Server Scaffold

This project includes a minimal Triton model repository scaffold.

Path:

```text
deployment/triton/model_repository/mobilenet_onnx/
```

## Steps

1. Export ONNX model:

```bash
python src/onnx_export.py
```

2. Copy model:

```powershell
copy models\mobilenet_v2.onnx deployment\triton\model_repository\mobilenet_onnx\1\model.onnx
```

3. Run Triton container:

```powershell
docker run --rm -p8001:8001 -p8002:8002 -p8000:8000 ^
  -v %cd%\deployment\triton\model_repository:/models ^
  nvcr.io/nvidia/tritonserver:24.02-py3 ^
  tritonserver --model-repository=/models
```

Adjust the Triton image version according to your CUDA/GPU environment.
