# Benchmarking Methodology

## Objective

This project evaluates AI inference deployment configurations under practical constraints such as latency, accuracy, energy, carbon, cost, queueing delay, and autoscaling pressure.

## Benchmark Layers

### 1. Real ML Benchmark

The local benchmark layer uses the sklearn digits dataset to estimate:

- classification accuracy
- local inference latency
- confusion matrix
- benchmark comparison tables

### 2. ONNX Runtime Benchmark

The ONNX layer demonstrates a deployment-oriented path:

```text
PyTorch model → ONNX export → ONNX Runtime benchmark
```

This is useful because many production inference systems use optimized runtime formats rather than raw training-framework execution.

### 3. Scenario-Based Infrastructure Model

The infrastructure model estimates:

- active serving hours
- idle power contribution
- IT energy
- facility energy through PUE
- regional carbon emissions
- monthly cost
- energy per 1,000 inferences

## Limitations

This project is transparent about measured vs scenario-based elements.

Measured:

- sklearn benchmark accuracy
- sklearn latency
- ONNX Runtime latency

Scenario-based:

- hardware power assumptions
- regional electricity prices
- carbon intensity values
- optimization multipliers
- autoscaling approximations

The purpose is not to claim exact production telemetry, but to demonstrate a reproducible decision framework.
