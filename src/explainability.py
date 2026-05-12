import pandas as pd


def score_breakdown(row, max_latency_ms, max_cost, min_accuracy, optimize_for):
    """
    Produces a transparent approximate score decomposition for dashboard explanation.

    This mirrors the high-level logic used by the decision engine. Exact penalties may
    differ if the engine adds feasibility penalties, but this gives the user a clear
    directional explanation.
    """
    weights = {
        "Balanced": {"latency": 0.25, "energy": 0.20, "cost": 0.20, "accuracy_gap": 0.15, "carbon": 0.10, "slo": 0.10},
        "Latency": {"latency": 0.45, "energy": 0.10, "cost": 0.10, "accuracy_gap": 0.15, "carbon": 0.05, "slo": 0.15},
        "Cost": {"latency": 0.15, "energy": 0.15, "cost": 0.45, "accuracy_gap": 0.10, "carbon": 0.05, "slo": 0.10},
        "Carbon": {"latency": 0.15, "energy": 0.20, "cost": 0.10, "accuracy_gap": 0.10, "carbon": 0.35, "slo": 0.10},
        "Energy": {"latency": 0.15, "energy": 0.40, "cost": 0.10, "accuracy_gap": 0.10, "carbon": 0.15, "slo": 0.10},
    }.get(optimize_for, {
        "latency": 0.25, "energy": 0.20, "cost": 0.20, "accuracy_gap": 0.15, "carbon": 0.10, "slo": 0.10
    })

    latency_component = weights["latency"] * (float(row["latency_ms"]) / max(max_latency_ms, 1))
    energy_component = weights["energy"] * (float(row["energy_per_1000_inferences_wh"]) / 10)
    cost_component = weights["cost"] * (float(row["monthly_cost_usd"]) / max(max_cost, 1))
    accuracy_gap = max(0, min_accuracy - float(row["estimated_accuracy"]))
    accuracy_component = weights["accuracy_gap"] * accuracy_gap * 10
    carbon_component = weights["carbon"] * (float(row["carbon_kgco2"]) / 10)

    if "latency_slo_ms" in row and row["latency_ms"] > row["latency_slo_ms"]:
        slo_penalty = (float(row["latency_ms"]) - float(row["latency_slo_ms"])) / max(float(row["latency_slo_ms"]), 1)
    else:
        slo_penalty = 0
    slo_component = weights["slo"] * slo_penalty

    data = [
        {"component": "Latency", "value": latency_component, "weight": weights["latency"]},
        {"component": "Energy", "value": energy_component, "weight": weights["energy"]},
        {"component": "Cost", "value": cost_component, "weight": weights["cost"]},
        {"component": "Accuracy gap", "value": accuracy_component, "weight": weights["accuracy_gap"]},
        {"component": "Carbon", "value": carbon_component, "weight": weights["carbon"]},
        {"component": "SLO penalty", "value": slo_component, "weight": weights["slo"]},
    ]
    return pd.DataFrame(data)


def recommendation_trace(results: pd.DataFrame):
    total = len(results)
    latency_pass = int(results["meets_latency"].sum())
    accuracy_pass = int(results["meets_accuracy"].sum())
    cost_pass = int(results["meets_cost"].sum())
    slo_pass = int((~results["violates_slo"]).sum())
    all_pass = int((results["meets_latency"] & results["meets_accuracy"] & results["meets_cost"] & (~results["violates_slo"])).sum())

    return pd.DataFrame([
        {"stage": "All candidates", "remaining": total, "explanation": "All hardware and optimization combinations evaluated"},
        {"stage": "Latency filter", "remaining": latency_pass, "explanation": "Configurations satisfying max latency"},
        {"stage": "Accuracy filter", "remaining": accuracy_pass, "explanation": "Configurations satisfying minimum accuracy"},
        {"stage": "Cost filter", "remaining": cost_pass, "explanation": "Configurations satisfying monthly cost ceiling"},
        {"stage": "SLO filter", "remaining": slo_pass, "explanation": "Configurations satisfying latency SLO"},
        {"stage": "All constraints", "remaining": all_pass, "explanation": "Configurations satisfying all production constraints"},
    ])


def executive_insights(best, results):
    lowest_carbon = results.sort_values("carbon_kgco2").iloc[0]
    cheapest = results.sort_values("monthly_cost_usd").iloc[0]
    fastest = results.sort_values("latency_ms").iloc[0]

    insights = [
        f"Selected configuration: {best.strategy} on {best.hardware}.",
        f"Fastest option: {fastest.strategy} on {fastest.hardware} at {fastest.latency_ms:.2f} ms.",
        f"Lowest-carbon option: {lowest_carbon.strategy} on {lowest_carbon.hardware} at {lowest_carbon.carbon_kgco2:.3f} kgCO₂/month.",
        f"Lowest-cost option: {cheapest.strategy} on {cheapest.hardware} at ${cheapest.monthly_cost_usd:.2f}/month.",
    ]

    if bool(best.get("is_pareto_efficient", False)):
        insights.append("The selected configuration is Pareto-efficient across latency, cost, and carbon.")
    else:
        insights.append("The selected configuration is not Pareto-efficient, but may still win under the selected scoring objective.")

    return insights
