from pathlib import Path
import sys
from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from data_loader import load_data
from engine import evaluate_all, recommend


app = FastAPI(
    title="AI Inference Deployment Decision Engine API",
    description=(
        "API for recommending AI deployment configurations under latency, accuracy, "
        "cost, energy, carbon, SLO, workload, and infrastructure constraints."
    ),
    version="2.0.0",
)

DATA = load_data()


class RecommendationRequest(BaseModel):
    model_name: str = Field("MobileNetV2", description="Model profile name")
    region_name: str = Field("Germany", description="Region profile name")

    monthly_requests: int = Field(5_000_000, ge=1)
    max_latency_ms: float = Field(75, gt=0)
    min_accuracy: float = Field(0.70, ge=0, le=1)
    max_cost: float = Field(200, gt=0)

    pue: float = Field(1.56, ge=1.0)
    utilization: float = Field(0.55, ge=0.01, le=1.0)

    traffic_pattern: Literal["Steady", "Burst", "Spiky"] = "Burst"
    serving_type: Literal["Real-time", "Batch"] = "Real-time"
    deployment_type: Literal["Hybrid", "Cloud", "Edge"] = "Hybrid"

    latency_slo_ms: float = Field(100, gt=0)
    batch_size: Literal[1, 4, 8, 16, 32] = 1
    autoscaling_enabled: bool = True
    max_instance_utilization: float = Field(0.70, ge=0.10, le=0.99)

    optimize_for: Literal["Balanced", "Latency", "Cost", "Carbon", "Energy"] = "Balanced"

    measured_latency_ms: Optional[float] = Field(None, gt=0)
    measured_accuracy: Optional[float] = Field(None, ge=0, le=1)


@app.get("/")
def root():
    return {
        "message": "AI Inference Deployment Decision Engine API",
        "docs": "/docs",
        "health": "/health",
        "recommendation_endpoint": "/recommend",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": DATA["models"].model_name.tolist(),
        "regions": DATA["regions"].region.tolist(),
        "hardware_options": DATA["hardware"].hardware.tolist(),
    }


@app.get("/metadata")
def metadata():
    return {
        "objectives": ["Balanced", "Latency", "Cost", "Carbon", "Energy"],
        "traffic_patterns": ["Steady", "Burst", "Spiky"],
        "serving_types": ["Real-time", "Batch"],
        "deployment_types": ["Hybrid", "Cloud", "Edge"],
        "batch_sizes": [1, 4, 8, 16, 32],
    }


@app.post("/recommend")
def get_recommendation(payload: RecommendationRequest):
    results = evaluate_all(
        DATA,
        model_name=payload.model_name,
        region_name=payload.region_name,
        monthly_requests=payload.monthly_requests,
        max_latency_ms=payload.max_latency_ms,
        min_accuracy=payload.min_accuracy,
        max_cost=payload.max_cost,
        pue=payload.pue,
        utilization=payload.utilization,
        measured_latency_ms=payload.measured_latency_ms,
        measured_accuracy=payload.measured_accuracy,
        traffic_pattern=payload.traffic_pattern,
        serving_type=payload.serving_type,
        deployment_type=payload.deployment_type,
        latency_slo_ms=payload.latency_slo_ms,
        batch_size=payload.batch_size,
        autoscaling_enabled=payload.autoscaling_enabled,
        max_instance_utilization=payload.max_instance_utilization,
        optimize_for=payload.optimize_for,
    )

    recommendation = recommend(results)

    return {
        "recommendation": recommendation,
        "top_configurations": results.head(10).to_dict(orient="records"),
    }


@app.post("/compare-objectives")
def compare_objectives(payload: RecommendationRequest):
    comparisons = {}

    for objective in ["Balanced", "Latency", "Cost", "Carbon", "Energy"]:
        results = evaluate_all(
            DATA,
            model_name=payload.model_name,
            region_name=payload.region_name,
            monthly_requests=payload.monthly_requests,
            max_latency_ms=payload.max_latency_ms,
            min_accuracy=payload.min_accuracy,
            max_cost=payload.max_cost,
            pue=payload.pue,
            utilization=payload.utilization,
            measured_latency_ms=payload.measured_latency_ms,
            measured_accuracy=payload.measured_accuracy,
            traffic_pattern=payload.traffic_pattern,
            serving_type=payload.serving_type,
            deployment_type=payload.deployment_type,
            latency_slo_ms=payload.latency_slo_ms,
            batch_size=payload.batch_size,
            autoscaling_enabled=payload.autoscaling_enabled,
            max_instance_utilization=payload.max_instance_utilization,
            optimize_for=objective,
        )

        rec = recommend(results)
        comparisons[objective] = rec["best_result"]

    return comparisons
