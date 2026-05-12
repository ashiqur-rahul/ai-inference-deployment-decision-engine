import pandas as pd


def mark_pareto_frontier(df: pd.DataFrame, objective_columns=None) -> pd.DataFrame:
    if objective_columns is None:
        objective_columns = ["latency_ms", "monthly_cost_usd", "carbon_kgco2"]

    out = df.copy()
    values = out[objective_columns].astype(float).values
    is_pareto = []

    for candidate in values:
        dominated = (((values <= candidate).all(axis=1)) & ((values < candidate).any(axis=1))).any()
        is_pareto.append(not dominated)

    out["is_pareto_efficient"] = is_pareto
    return out


def objective_summary_table(results_by_objective: dict) -> pd.DataFrame:
    rows = []
    for objective, result in results_by_objective.items():
        rows.append({
            "objective": objective,
            "hardware": result.get("hardware"),
            "strategy": result.get("strategy"),
            "latency_ms": result.get("latency_ms"),
            "accuracy": result.get("estimated_accuracy"),
            "monthly_cost_usd": result.get("monthly_cost_usd"),
            "carbon_kgco2": result.get("carbon_kgco2"),
            "energy_per_1000_inferences_wh": result.get("energy_per_1000_inferences_wh"),
            "required_instances": result.get("required_instances"),
            "score": result.get("score"),
        })
    return pd.DataFrame(rows)


def normalize_for_comparison(df: pd.DataFrame, columns):
    out = df.copy()
    for col in columns:
        min_v = out[col].min()
        max_v = out[col].max()
        if max_v == min_v:
            out[f"{col}_score"] = 1.0
        else:
            out[f"{col}_score"] = 1 - ((out[col] - min_v) / (max_v - min_v))
    return out
