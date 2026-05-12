# System Design Explanation

## One-Minute Explanation

I built an AI inference deployment decision engine that helps select the best deployment configuration under real-world constraints. Instead of only reporting model accuracy, the system evaluates latency, energy, carbon impact, cost, and accuracy together.

It includes a real benchmark layer using the sklearn digits dataset, an ONNX Runtime benchmarking path, a multi-objective decision engine, a Streamlit dashboard, and a FastAPI recommendation endpoint.

The main purpose is to demonstrate how AI systems can be deployed more intelligently by considering infrastructure and optimization trade-offs.

---

## Problem

Most AI projects answer:

> How accurate is the model?

But real deployment also asks:

> How fast is it?
> How expensive is it?
> How much energy does it consume?
> Which hardware should run it?
> Which optimization strategy should be selected?

This project addresses that deployment decision problem.

---

## Inputs

The system uses:

- model profiles
- hardware profiles
- optimization profiles
- regional carbon intensity
- electricity cost
- real benchmark results
- user-defined constraints

---

## Decision Logic

The decision engine evaluates each possible configuration and checks whether it satisfies:

- maximum latency
- minimum accuracy
- maximum monthly cost

Then it ranks configurations using a multi-objective score that considers:

- latency
- energy per 1,000 inferences
- monthly cost
- accuracy penalty
- constraint violations

---

## Why ONNX Matters

ONNX adds a realistic deployment optimization layer.

The pipeline:

```text
PyTorch model → ONNX export → ONNX Runtime benchmark
```

This demonstrates that the system is not only a modelling dashboard, but also includes an actual deployment-oriented inference path.

---

## Why This Project Is Strong

This project combines three important areas:

1. Machine learning benchmarking
2. AI deployment engineering
3. Energy-aware infrastructure modelling

That combination is rare in portfolio projects and directly connects to practical AI engineering work.

---

## Limitations

The project is transparent about what is real and what is modeled.

Real:

- sklearn benchmark accuracy
- sklearn latency
- ONNX Runtime latency
- generated plots and reports

Modeled:

- hardware energy estimates
- carbon estimates
- cost estimates
- optimization effects

This makes the project credible and honest.
