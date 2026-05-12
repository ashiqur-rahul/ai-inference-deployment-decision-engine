# System Architecture

## Overview

The AI Inference Optimization & Deployment Decision Engine is structured as a modular decision-support system for evaluating AI inference deployment choices under practical constraints.

The system combines:

- real ML benchmarking
- ONNX Runtime benchmarking
- scenario-based infrastructure modelling
- constraint-based recommendation logic
- dashboard and API interfaces

The goal is not only to train or benchmark a model, but to answer a deployment question:

> Which inference configuration should be selected when latency, accuracy, cost, energy, and carbon constraints matter?

---

## High-Level Architecture

```text
Data & Benchmark Inputs
        ↓
Decision Engine
        ↓
Recommendation Layer
        ↓
Dashboard / API / Reports
```

---

## Main Components

### 1. Input Layer

The system uses multiple input sources:

- `data/model_profiles.csv`
- `data/hardware_profiles.csv`
- `data/optimization_strategies.csv`
- `data/region_profiles.csv`
- real benchmark reports in `outputs/reports/`

These inputs describe model behavior, hardware assumptions, optimization effects, regional energy context, and measured benchmark values.

---

### 2. Benchmark Layer

The benchmark layer provides measured performance signals.

#### Real ML Benchmark

`src/accuracy_benchmark.py`

Uses the `sklearn` digits dataset to measure:

- model accuracy
- local inference latency
- confusion matrix
- benchmark result tables

#### ONNX Runtime Benchmark

`src/onnx_export.py` and `src/onnx_benchmark.py`

Implements:

```text
PyTorch → ONNX → ONNX Runtime → Latency Benchmark
```

This demonstrates a production-oriented inference optimization path.

---

### 3. Decision Engine

`src/engine.py`

The decision engine evaluates each deployment configuration across:

- latency
- accuracy
- energy consumption
- carbon emissions
- monthly cost
- feasibility against user constraints

It ranks configurations using a multi-objective score and returns the best feasible option.

---

### 4. Application Layer

#### Streamlit Dashboard

`app/app.py`

Provides:

- interactive scenario controls
- deployment recommendation
- trade-off visualizations
- benchmark inspection
- ONNX results
- methodology and references

#### FastAPI Backend

`api/main.py`

Provides a deployable API endpoint:

```text
POST /recommend
```

The endpoint returns the recommended hardware, strategy, reasoning, and key metrics.

---

### 5. Output Layer

The system generates reproducible outputs:

- `outputs/figures/`
- `outputs/reports/`
- `models/`

Examples:

- latency vs energy plot
- accuracy vs energy plot
- confusion matrix
- ONNX latency chart
- benchmark comparison report
- recommendation JSON

---

## Real vs Scenario-Based Components

| Layer | Type | Purpose |
|---|---|---|
| sklearn benchmark | Real measured | Accuracy and local latency |
| ONNX Runtime benchmark | Real measured | Runtime latency and throughput |
| hardware profiles | Scenario-based | Deployment modelling |
| energy and carbon estimates | Scenario-based | Infrastructure-level estimation |
| decision score | Analytical | Ranking and recommendation |

This separation keeps the system transparent and honest.

---

## Design Strength

The project is intentionally designed as a hybrid ML + systems project.

It shows:

- model benchmarking
- inference optimization
- decision intelligence
- explainable recommendations
- API readiness
- energy-aware AI deployment thinking

This makes it stronger than a standard model-training portfolio project.
