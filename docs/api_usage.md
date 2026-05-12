# API Usage Examples

## Run API Locally

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Health Check

```bash
curl http://127.0.0.1:8000/health
```

---

## Recommendation Request

```bash
curl -X POST "http://127.0.0.1:8000/recommend" ^
  -H "Content-Type: application/json" ^
  -d "{\"model_name\":\"MobileNetV2\",\"region_name\":\"Germany\",\"monthly_requests\":5000000,\"max_latency_ms\":75,\"min_accuracy\":0.70,\"max_cost\":200,\"traffic_pattern\":\"Burst\",\"serving_type\":\"Real-time\",\"deployment_type\":\"Hybrid\",\"latency_slo_ms\":100,\"batch_size\":1,\"autoscaling_enabled\":true,\"max_instance_utilization\":0.70,\"optimize_for\":\"Carbon\"}"
```

For PowerShell, this is often easier:

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

---

## Compare Objectives

```powershell
$body = @{
  model_name = "MobileNetV2"
  region_name = "Germany"
  optimize_for = "Balanced"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/compare-objectives" -Method Post -Body $body -ContentType "application/json"
```
