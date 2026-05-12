try:
    from prometheus_client import Counter, Gauge
except Exception:
    Counter = None
    Gauge = None

def metrics_available() -> bool:
    return Counter is not None and Gauge is not None

if metrics_available():
    REQUEST_COUNTER = Counter("deployment_recommendation_requests_total", "Number of recommendation requests")
    LAST_LATENCY_MS = Gauge("deployment_selected_latency_ms", "Latency of selected deployment configuration")
    LAST_CARBON_KG = Gauge("deployment_selected_carbon_kg", "Carbon estimate of selected deployment configuration")
    LAST_COST_USD = Gauge("deployment_selected_cost_usd", "Monthly cost estimate of selected deployment configuration")
else:
    REQUEST_COUNTER = None
    LAST_LATENCY_MS = None
    LAST_CARBON_KG = None
    LAST_COST_USD = None

def record_recommendation(best_result: dict) -> None:
    if not metrics_available():
        return
    REQUEST_COUNTER.inc()
    LAST_LATENCY_MS.set(float(best_result.get("latency_ms", 0)))
    LAST_CARBON_KG.set(float(best_result.get("carbon_kgco2", 0)))
    LAST_COST_USD.set(float(best_result.get("monthly_cost_usd", 0)))
