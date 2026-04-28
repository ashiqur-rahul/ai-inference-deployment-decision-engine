# 🧠 AI Inference Optimization & Deployment Decision Engine

An end-to-end **AI deployment decision system** that recommends optimal inference configurations by balancing:

- ⚡ Latency
- 🎯 Accuracy
- 💰 Cost
- 🔋 Energy consumption
- 🌍 Carbon impact

This project goes beyond model accuracy and demonstrates **real-world AI system design, deployment decision-making, and infrastructure-aware optimization**.

---

## 🚀 Why This Project Stands Out

Most AI projects stop at training a model.

This project answers a harder and more practical question:

> **How should we deploy AI systems efficiently in the real world?**

It combines:

- ✅ Real ML benchmarking
- ✅ System-level optimization
- ✅ Infrastructure-aware trade-off analysis
- ✅ Deployment-ready architecture
- ✅ Interactive dashboard
- ✅ API-ready backend

---

## 🧩 Key Capabilities

### 🔬 Real Benchmarking Layer

- Uses the real `sklearn` digits dataset
- Measures model accuracy
- Measures local inference latency
- Generates a confusion matrix

### ⚙️ Deployment Decision Engine

- Multi-objective scoring system
- Constraint-based filtering:
  - latency limits
  - accuracy thresholds
  - cost ceilings
- Recommends the best deployment configuration

### 📊 Interactive Dashboard

- Built with Streamlit
- Shows latency, energy, cost, carbon, and accuracy trade-offs
- Allows interactive scenario exploration

### 🔌 API-Ready System

- FastAPI backend
- Recommendation endpoint: `/recommend`
- Local Swagger UI available at `/docs`

### ⚡ Optimization Strategies

- Quantization
- Pruning
- Batching
- Caching
- ONNX Runtime scenario
- Combined optimization strategies

---

## ⚙️ How It Works

```text
Real Benchmark
      ↓
Scenario Modeling
      ↓
Optimization Strategies
      ↓
Constraint Filtering
      ↓
Multi-objective Ranking
      ↓
Deployment Recommendation
```

---

## 📊 Results Preview

### Latency vs Energy

![Latency vs Energy](outputs/figures/latency_energy_tradeoff.png)

### Accuracy vs Energy

![Accuracy vs Energy](outputs/figures/accuracy_energy_tradeoff.png)

### Energy Comparison

![Energy Comparison](outputs/figures/energy_comparison.png)

### Cost Comparison

![Cost Comparison](outputs/figures/cost_comparison.png)

### Confusion Matrix

![Confusion Matrix](outputs/figures/digits_confusion_matrix.png)

---

## 🚀 Quick Start

### 1. Create and activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the real benchmark

```bash
python src/accuracy_benchmark.py
```

### 4. Run the pipeline

```bash
python src/run_pipeline.py
```

### 5. Launch the Streamlit dashboard

```bash
streamlit run app/app.py
```

---

## 🔌 API Usage

Run the FastAPI backend locally:

```bash
uvicorn api.main:app --reload
```

Then open Swagger UI in your browser:

```text
http://127.0.0.1:8000/docs
```

> Note: `127.0.0.1` is a local address. It only works while the API is running on your own computer. For a public link, the API must be deployed, for example on Render.

### Endpoint

```text
POST /recommend
```

Returns:

- recommended hardware
- optimization strategy
- reasoning
- latency estimate
- accuracy estimate
- cost estimate
- energy and carbon estimates

---

## ⚡ Optional Advanced Benchmarks

### PyTorch benchmark

```bash
pip install torch torchvision
python src/real_benchmark.py
```

### ONNX export and benchmark

```bash
pip install torch torchvision onnx onnxruntime
python src/onnx_export.py
python src/onnx_benchmark.py
```

---

## 📁 Project Structure

```text
.
├── app/                 # Streamlit dashboard
├── api/                 # FastAPI service
├── src/                 # Core decision engine and benchmark scripts
├── data/                # Model, hardware, optimization, and region profiles
├── outputs/             # Generated reports and figures
├── docs/                # Methodology, assumptions, references
├── models/              # Saved models
├── tests/               # Unit tests
├── requirements.txt
└── README.md
```

---

## 📊 Key Outputs

- Deployment recommendation JSON
- Latency vs energy trade-off plot
- Accuracy vs energy analysis
- Cost comparison plot
- Real dataset confusion matrix
- Energy and carbon estimates
- API-based recommendation response

---

## 🌍 Methodology Note

This project uses a **hybrid methodology**:

1. **Real ML benchmarking**
   - real dataset
   - measured accuracy
   - measured local latency

2. **Scenario-based infrastructure modeling**
   - energy consumption
   - carbon impact
   - cost estimates
   - hardware and optimization profiles

It is designed to demonstrate an **AI deployment decision framework**, not exact production telemetry.

---

## 🔬 References

- NVIDIA GPU specifications
- Uptime Institute data center PUE references
- Ember electricity carbon intensity references
- US EPA eGRID
- Scikit-learn digits dataset

See: [`docs/references.md`](docs/references.md)

---

## 🎥 Demo Video

Coming soon.

---

## 👨‍💻 Author

**Ashiqur Rahman Rahul**  
AI Research Analyst  
AI Systems | Energy Efficiency | Infrastructure Optimization  
Berlin, Germany

---

## 📜 License

MIT License
