import pandas as pd

HOURS_PER_MONTH = 730

def estimate(model, hardware, strategy, region, monthly_requests, max_latency_ms, min_accuracy, max_cost, pue=1.56, utilization=0.55, measured_latency_ms=None, measured_accuracy=None):
    if measured_latency_ms is None:
        latency_ms = float(model.baseline_latency_ms_cpu) / float(hardware.speed_factor) * float(strategy.latency_multiplier)
        latency_source = "scenario"
    else:
        latency_ms = float(measured_latency_ms) * float(strategy.latency_multiplier)
        latency_source = "measured_adjusted"

    throughput_rps = max(1.0, 1000.0 / max(latency_ms, 1e-6)) / float(strategy.compute_multiplier)
    base_accuracy = float(model.baseline_accuracy) if measured_accuracy is None else float(measured_accuracy)
    accuracy_source = "scenario" if measured_accuracy is None else "measured_dataset"
    accuracy = max(0.0, base_accuracy - float(strategy.accuracy_drop))
    model_size_mb = float(model.model_size_mb) * float(strategy.model_size_multiplier)

    active_hours = min(HOURS_PER_MONTH, (monthly_requests / throughput_rps) / 3600.0)
    idle_hours = max(0, HOURS_PER_MONTH - active_hours)
    active_power = float(hardware.tdp_watts) * utilization * float(strategy.energy_multiplier)
    idle_power = float(hardware.idle_watts)

    it_kwh = ((active_power * active_hours) + (idle_power * idle_hours)) / 1000.0
    facility_kwh = it_kwh * pue
    carbon_kg = facility_kwh * float(region.carbon_intensity_gco2_kwh) / 1000.0
    infra_cost = active_hours * float(hardware.hourly_infra_cost_usd)
    platform_fee = monthly_requests * float(hardware.request_platform_fee_usd)
    energy_cost = facility_kwh * float(region.electricity_price_usd_kwh)
    cost = infra_cost + platform_fee + energy_cost
    wh_per_1000 = (facility_kwh * 1000 / max(monthly_requests, 1)) * 1000

    meets_latency = latency_ms <= max_latency_ms
    meets_accuracy = accuracy >= min_accuracy
    meets_cost = cost <= max_cost
    score = 0.35*(latency_ms/max(max_latency_ms,1)) + 0.30*(wh_per_1000/10) + 0.20*(cost/max(max_cost,1)) + 0.15*max(0,min_accuracy-accuracy)*10
    if not meets_latency: score += 5
    if not meets_accuracy: score += 5
    if not meets_cost: score += 3

    return {
        "model": model.model_name,
        "hardware": hardware.hardware,
        "strategy": strategy.strategy,
        "latency_ms": latency_ms,
        "latency_source": latency_source,
        "throughput_rps": throughput_rps,
        "estimated_accuracy": accuracy,
        "accuracy_source": accuracy_source,
        "model_size_mb": model_size_mb,
        "facility_energy_kwh": facility_kwh,
        "it_energy_kwh": it_kwh,
        "energy_per_1000_inferences_wh": wh_per_1000,
        "carbon_kgco2": carbon_kg,
        "monthly_cost_usd": cost,
        "active_hours_month": active_hours,
        "meets_latency": meets_latency,
        "meets_accuracy": meets_accuracy,
        "meets_cost": meets_cost,
        "score": score,
    }

def evaluate_all(data, model_name="MobileNetV2", region_name="Germany", monthly_requests=5_000_000, max_latency_ms=75, min_accuracy=0.70, max_cost=200, pue=1.56, utilization=0.55, measured_latency_ms=None, measured_accuracy=None):
    model = data["models"][data["models"].model_name == model_name].iloc[0]
    region = data["regions"][data["regions"].region == region_name].iloc[0]
    rows = []
    for _, hardware in data["hardware"].iterrows():
        for _, strategy in data["strategies"].iterrows():
            rows.append(estimate(model, hardware, strategy, region, monthly_requests, max_latency_ms, min_accuracy, max_cost, pue, utilization, measured_latency_ms, measured_accuracy))
    return pd.DataFrame(rows).sort_values("score")

def recommend(results):
    feasible = results[results.meets_latency & results.meets_accuracy & results.meets_cost]
    if len(feasible):
        best = feasible.iloc[0]
        status = "Feasible recommendation found"
    else:
        best = results.iloc[0]
        status = "No configuration meets all constraints. Returning closest option."
    failed = results[~(results.meets_latency & results.meets_accuracy & results.meets_cost)].head(5)
    rejected = []
    for _, row in failed.iterrows():
        reasons = []
        if not row.meets_latency: reasons.append("latency")
        if not row.meets_accuracy: reasons.append("accuracy")
        if not row.meets_cost: reasons.append("cost")
        rejected.append({"hardware": row.hardware, "strategy": row.strategy, "failed_constraints": reasons, "score": float(row.score)})
    reason = f"{best.strategy} on {best.hardware} selected: {best.latency_ms:.1f} ms latency, {best.estimated_accuracy:.3f} estimated accuracy, {best.facility_energy_kwh:.2f} kWh/month, {best.carbon_kgco2:.2f} kgCO2/month, ${best.monthly_cost_usd:.2f}/month."
    return {"status": status, "hardware": best.hardware, "strategy": best.strategy, "reason": reason, "best_result": best.to_dict(), "rejected_examples": rejected}
