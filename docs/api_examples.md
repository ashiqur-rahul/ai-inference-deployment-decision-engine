# Production API Examples

## Run API

```bash
uvicorn api.main:app --reload
```

## Python Client Example

```python
import requests

payload = {
    "model_name": "MobileNetV2",
    "region_name": "Germany",
    "monthly_requests": 5000000,
    "max_latency_ms": 75,
    "min_accuracy": 0.70,
    "max_cost": 200,
    "traffic_pattern": "Burst",
    "serving_type": "Real-time",
    "deployment_type": "Hybrid",
    "latency_slo_ms": 100,
    "batch_size": 1,
    "autoscaling_enabled": True,
    "max_instance_utilization": 0.70,
    "optimize_for": "Carbon"
}

response = requests.post("http://127.0.0.1:8000/recommend", json=payload)
print(response.json())
```

## PowerShell Example

```powershell
$body = @{
  model_name = "MobileNetV2"
  region_name = "Germany"
  monthly_requests = 5000000
  max_latency_ms = 75
  min_accuracy = 0.70
  max_cost = 200
  traffic_pattern = "Burst"
  serving_type = "Real-time"
  deployment_type = "Hybrid"
  latency_slo_ms = 100
  batch_size = 1
  autoscaling_enabled = $true
  max_instance_utilization = 0.70
  optimize_for = "Carbon"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/recommend" -Method Post -Body $body -ContentType "application/json"
```
