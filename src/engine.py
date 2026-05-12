import pandas as pd

HOURS_PER_MONTH = 730


def traffic_multiplier(pattern: str) -> float:
    return {"Steady": 1.0, "Burst": 1.35, "Spiky": 1.75}.get(pattern, 1.0)


def queue_delay_ms(arrival_rps: float, service_rps: float) -> float:
    if service_rps <= 0:
        return 9999.0

    rho = min(arrival_rps / service_rps, 0.98)

    if rho <= 0:
        return 0.0

    return (rho / max(service_rps - arrival_rps, 1e-6)) * 1000


def autoscaling_plan(
    arrival_rps: float,
    service_rps_per_instance: float,
    max_instance_utilization: float = 0.70,
) -> int:
    if service_rps_per_instance <= 0:
        return 999

    required = arrival_rps / (service_rps_per_instance * max_instance_utilization)
    return max(1, int(required) + (0 if required.is_integer() else 1))


def scheduling_class(hardware_name: str) -> str:
    name = str(hardware_name).lower()

    if "jetson" in name:
        return "Edge"
    if any(x in name for x in ["nvidia", "gpu", "a100", "h100", "t4"]):
        return "GPU"
    return "CPU"


def apply_serving_strategy(latency_ms: float, energy_wh: float, serving_type: str):
    if serving_type == "Batch":
        return latency_ms * 1.45, energy_wh * 0.72

    if serving_type == "Real-time":
        return latency_ms * 0.90, energy_wh * 1.15

    return latency_ms, energy_wh


def apply_deployment_type(
    latency_ms: float,
    cost: float,
    energy_wh: float,
    hardware_name: str,
    deployment_type: str,
):
    hw_class = scheduling_class(hardware_name)

    if deployment_type == "Edge":
        if hw_class == "Edge":
            latency_ms *= 0.70
            cost *= 0.65
            energy_wh *= 0.85
        else:
            latency_ms *= 1.20
            cost *= 1.05

    elif deployment_type == "Cloud":
        if hw_class == "GPU":
            latency_ms *= 0.82
            cost *= 1.25
        elif hw_class == "CPU":
            latency_ms *= 1.05
            cost *= 1.10
        elif hw_class == "Edge":
            latency_ms *= 1.35
            cost *= 1.20

    elif deployment_type == "Hybrid":
        latency_ms *= 0.90
        cost *= 1.10
        energy_wh *= 0.95

    return latency_ms, cost, energy_wh


def compute_slo_penalty(latency_ms: float, latency_slo_ms: float) -> float:
    if latency_ms <= latency_slo_ms:
        return 0.0

    return (latency_ms - latency_slo_ms) / max(latency_slo_ms, 1e-6)


def objective_weights(optimize_for: str):
    """
    Multi-objective weighting profile.

    Balanced:
        General deployment suitability.
    Latency:
        Prioritizes low latency and SLO performance.
    Cost:
        Prioritizes monthly operating cost.
    Carbon:
        Prioritizes carbon emissions and energy efficiency.
    Energy:
        Prioritizes Wh per 1,000 inferences and monthly energy.
    """
    profiles = {
        "Balanced": {
            "latency": 0.25,
            "energy": 0.20,
            "cost": 0.20,
            "accuracy_gap": 0.15,
            "carbon": 0.10,
            "slo": 0.10,
        },
        "Latency": {
            "latency": 0.45,
            "energy": 0.10,
            "cost": 0.10,
            "accuracy_gap": 0.15,
            "carbon": 0.05,
            "slo": 0.15,
        },
        "Cost": {
            "latency": 0.15,
            "energy": 0.15,
            "cost": 0.45,
            "accuracy_gap": 0.10,
            "carbon": 0.05,
            "slo": 0.10,
        },
        "Carbon": {
            "latency": 0.15,
            "energy": 0.20,
            "cost": 0.10,
            "accuracy_gap": 0.10,
            "carbon": 0.35,
            "slo": 0.10,
        },
        "Energy": {
            "latency": 0.15,
            "energy": 0.40,
            "cost": 0.10,
            "accuracy_gap": 0.10,
            "carbon": 0.15,
            "slo": 0.10,
        },
    }

    return profiles.get(optimize_for, profiles["Balanced"])


def estimate(
    model,
    hardware,
    strategy,
    region,
    monthly_requests,
    max_latency_ms,
    min_accuracy,
    max_cost,
    pue=1.56,
    utilization=0.55,
    measured_latency_ms=None,
    measured_accuracy=None,
    traffic_pattern="Steady",
    serving_type="Real-time",
    deployment_type="Hybrid",
    latency_slo_ms=100,
    batch_size=1,
    autoscaling_enabled=True,
    max_instance_utilization=0.70,
    optimize_for="Balanced",
):
    if measured_latency_ms is None:
        latency_ms = (
            float(model.baseline_latency_ms_cpu)
            / float(hardware.speed_factor)
            * float(strategy.latency_multiplier)
        )
        latency_source = "scenario"
    else:
        latency_ms = float(measured_latency_ms) * float(strategy.latency_multiplier)
        latency_source = "measured_adjusted"

    batch_size = max(1, int(batch_size))

    if batch_size > 1:
        latency_ms *= 1 + 0.08 * (batch_size - 1)

    throughput_rps = max(1.0, 1000.0 / max(latency_ms, 1e-6)) / float(strategy.compute_multiplier)
    throughput_rps *= batch_size * 0.85

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

    avg_arrival_rps = monthly_requests / (HOURS_PER_MONTH * 3600)
    peak_arrival_rps = avg_arrival_rps * traffic_multiplier(traffic_pattern)

    required_instances = (
        autoscaling_plan(peak_arrival_rps, throughput_rps, max_instance_utilization)
        if autoscaling_enabled
        else 1
    )

    effective_service_rps = throughput_rps * required_instances
    q_delay = queue_delay_ms(peak_arrival_rps, effective_service_rps)

    latency_ms += q_delay

    latency_ms, wh_per_1000 = apply_serving_strategy(latency_ms, wh_per_1000, serving_type)
    latency_ms, cost, wh_per_1000 = apply_deployment_type(
        latency_ms,
        cost,
        wh_per_1000,
        hardware.hardware,
        deployment_type,
    )

    if autoscaling_enabled and required_instances > 1:
        cost *= 1 + 0.18 * (required_instances - 1)
        facility_kwh *= 1 + 0.12 * (required_instances - 1)
        carbon_kg = facility_kwh * float(region.carbon_intensity_gco2_kwh) / 1000.0

    meets_latency = latency_ms <= max_latency_ms
    meets_accuracy = accuracy >= min_accuracy
    meets_cost = cost <= max_cost
    violates_slo = latency_ms > latency_slo_ms

    slo_penalty = compute_slo_penalty(latency_ms, latency_slo_ms)
    accuracy_gap = max(0, min_accuracy - accuracy)

    weights = objective_weights(optimize_for)

    score = (
        weights["latency"] * (latency_ms / max(max_latency_ms, 1))
        + weights["energy"] * (wh_per_1000 / 10)
        + weights["cost"] * (cost / max(max_cost, 1))
        + weights["accuracy_gap"] * accuracy_gap * 10
        + weights["carbon"] * (carbon_kg / 10)
        + weights["slo"] * slo_penalty
    )

    if not meets_latency:
        score += 5
    if not meets_accuracy:
        score += 5
    if not meets_cost:
        score += 3
    if violates_slo:
        score += 2 * slo_penalty

    return {
        "model": model.model_name,
        "hardware": hardware.hardware,
        "hardware_class": scheduling_class(hardware.hardware),
        "strategy": strategy.strategy,
        "traffic_pattern": traffic_pattern,
        "serving_type": serving_type,
        "deployment_type": deployment_type,
        "batch_size": batch_size,
        "optimize_for": optimize_for,
        "latency_ms": latency_ms,
        "latency_source": latency_source,
        "queue_delay_ms": q_delay,
        "latency_slo_ms": latency_slo_ms,
        "violates_slo": violates_slo,
        "avg_arrival_rps": avg_arrival_rps,
        "peak_arrival_rps": peak_arrival_rps,
        "throughput_rps": throughput_rps,
        "autoscaling_enabled": autoscaling_enabled,
        "required_instances": required_instances,
        "target_instance_utilization": max_instance_utilization,
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


def evaluate_all(
    data,
    model_name="MobileNetV2",
    region_name="Germany",
    monthly_requests=5_000_000,
    max_latency_ms=75,
    min_accuracy=0.70,
    max_cost=200,
    pue=1.56,
    utilization=0.55,
    measured_latency_ms=None,
    measured_accuracy=None,
    traffic_pattern="Steady",
    serving_type="Real-time",
    deployment_type="Hybrid",
    latency_slo_ms=100,
    batch_size=1,
    autoscaling_enabled=True,
    max_instance_utilization=0.70,
    optimize_for="Balanced",
):
    model = data["models"][data["models"].model_name == model_name].iloc[0]
    region = data["regions"][data["regions"].region == region_name].iloc[0]

    rows = []

    for _, hardware in data["hardware"].iterrows():
        for _, strategy in data["strategies"].iterrows():
            rows.append(
                estimate(
                    model,
                    hardware,
                    strategy,
                    region,
                    monthly_requests,
                    max_latency_ms,
                    min_accuracy,
                    max_cost,
                    pue,
                    utilization,
                    measured_latency_ms,
                    measured_accuracy,
                    traffic_pattern,
                    serving_type,
                    deployment_type,
                    latency_slo_ms,
                    batch_size,
                    autoscaling_enabled,
                    max_instance_utilization,
                    optimize_for,
                )
            )

    return pd.DataFrame(rows).sort_values("score")


def recommend(results):
    feasible = results[
        results.meets_latency
        & results.meets_accuracy
        & results.meets_cost
        & (~results.violates_slo)
    ]

    if len(feasible):
        best = feasible.iloc[0]
        status = "Feasible recommendation found"
    else:
        best = results.iloc[0]
        status = "No configuration meets all constraints. Returning closest option."

    failed = results[
        ~(
            results.meets_latency
            & results.meets_accuracy
            & results.meets_cost
            & (~results.violates_slo)
        )
    ].head(5)

    rejected = []

    for _, row in failed.iterrows():
        reasons = []

        if not row.meets_latency:
            reasons.append("latency")
        if not row.meets_accuracy:
            reasons.append("accuracy")
        if not row.meets_cost:
            reasons.append("cost")
        if row.violates_slo:
            reasons.append("SLO violation")

        rejected.append(
            {
                "hardware": row.hardware,
                "strategy": row.strategy,
                "failed_constraints": reasons,
                "score": float(row.score),
            }
        )

    reason = (
        f"{best.strategy} on {best.hardware} selected under {best.optimize_for} objective: "
        f"{best.latency_ms:.1f} ms latency, "
        f"{best.estimated_accuracy:.3f} estimated accuracy, "
        f"{best.facility_energy_kwh:.2f} kWh/month, "
        f"{best.carbon_kgco2:.2f} kgCO2/month, "
        f"${best.monthly_cost_usd:.2f}/month, "
        f"{int(best.required_instances)} serving instance(s)."
    )

    return {
        "status": status,
        "hardware": best.hardware,
        "strategy": best.strategy,
        "objective": best.optimize_for,
        "reason": reason,
        "best_result": best.to_dict(),
        "rejected_examples": rejected,
    }
