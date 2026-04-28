import sys
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from data_loader import load_data
from engine import evaluate_all, recommend
app = FastAPI(title="AI Deployment Decision Engine API")
class RecommendationRequest(BaseModel):
    model_name: str = "MobileNetV2"
    region_name: str = "Germany"
    monthly_requests: int = 5_000_000
    max_latency_ms: float = 75
    min_accuracy: float = 0.70
    max_cost: float = 200
    pue: float = 1.56
    utilization: float = 0.55
@app.get("/")
def root():
    return {"message": "AI Deployment Decision Engine API"}
@app.post("/recommend")
def get_recommendation(req: RecommendationRequest):
    data = load_data()
    results = evaluate_all(data, req.model_name, req.region_name, req.monthly_requests, req.max_latency_ms, req.min_accuracy, req.max_cost, req.pue, req.utilization)
    return recommend(results)
