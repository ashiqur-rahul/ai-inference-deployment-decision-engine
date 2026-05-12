# Carbon-Aware Optimization

## Purpose

The decision engine supports different optimization objectives:

- Balanced
- Latency
- Cost
- Carbon
- Energy

This allows the system to answer different deployment questions.

For example:

- If the objective is **Latency**, the system prioritizes fast inference and SLO performance.
- If the objective is **Cost**, it prioritizes lower monthly operating cost.
- If the objective is **Carbon**, it prioritizes lower carbon emissions and energy efficiency.
- If the objective is **Energy**, it prioritizes lower energy per 1,000 inferences and total monthly energy.

---

## Why This Matters

Real AI deployment decisions are not single-objective.

A model that is fastest may not be cheapest.  
A model that is cheapest may not be lowest carbon.  
A model that is lowest carbon may not satisfy latency SLOs.

The carbon-aware objective makes the system more relevant for:

- sustainable AI
- energy-aware AI infrastructure
- green AI benchmarking
- policy and ESG-oriented AI analysis

---

## Score Design

The scoring function uses objective-specific weights across:

- latency
- energy per 1,000 inferences
- monthly cost
- accuracy gap
- carbon emissions
- SLO violation penalty

The objective changes the weight distribution rather than changing the raw metrics.

This keeps the system transparent and comparable across scenarios.
