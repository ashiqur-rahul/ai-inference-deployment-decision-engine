# Pareto Frontier and Objective Comparison

This project includes a Pareto frontier layer for comparing AI deployment configurations across competing objectives.

A configuration is Pareto-efficient when no other configuration is better across all selected objectives.

## Compared Dimensions

- Latency
- Monthly cost
- Carbon emissions
- Energy per 1,000 inferences
- Accuracy
- Required serving instances

## Why It Matters

Real deployment decisions are multi-objective. The fastest option may be expensive. The cheapest option may have higher carbon impact. The lowest-carbon option may not always satisfy a strict latency SLO.

The Pareto layer helps identify deployment configurations that are worth serious consideration rather than relying only on a single ranking score.

## Dashboard Views

- Pareto frontier: latency vs cost
- Pareto frontier: latency vs carbon
- Objective comparison table
- Objective-switching impact chart
- Best configuration under Balanced, Latency, Cost, Carbon, and Energy objectives
