# 🧠 AI Inference Optimization & Deployment Decision Engine

An end-to-end **AI deployment decision system** that recommends optimal inference configurations by balancing:

- ⚡ Latency  
- 🎯 Accuracy  
- 💰 Cost  
- 🔋 Energy consumption  
- 🌍 Carbon impact  

This project goes beyond model performance and demonstrates **real-world AI system design and deployment thinking**.

---

## 🚀 Why This Project Stands Out

Most AI projects stop at training models.

👉 This project answers a much harder question:

> “How should we deploy AI systems efficiently in the real world?”

It combines:
- ✅ Real ML benchmarking  
- ✅ System-level optimization  
- ✅ Infrastructure-aware trade-offs  
- ✅ Deployment-ready architecture  

---

## 🧩 Key Capabilities

### 🔬 Real Benchmarking Layer
- sklearn digits dataset (real data)
- Measured accuracy (~97%+)
- Measured inference latency
- Confusion matrix visualization

### ⚙️ Decision Engine
- Multi-objective optimization
- Constraint-based filtering:
  - latency limits
  - accuracy thresholds
  - cost ceilings
- Ranking system for deployment configurations

### 📊 Interactive Dashboard
- Built with Streamlit
- Trade-off exploration

### 🔌 API-Ready System
- FastAPI backend
- Endpoint: /recommend

### ⚡ Optimization Strategies
- Quantization
- Pruning
- Batching
- Caching

---

## ⚙️ How It Works

Real Benchmark → Scenario Modeling → Optimization → Constraint Filtering → Ranking → Recommendation

---

## 📊 Dashboard Preview

(Add screenshots inside /assets folder)

![Latency vs Energy](assets/latency_energy_tradeoff.png)
![Accuracy vs Energy](assets/accuracy_energy_tradeoff.png)
![Energy Comparison](assets/energy_comparison.png)
![Confusion Matrix](assets/digits_confusion_matrix.png)

---

## 🚀 Quick Start

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/accuracy_benchmark.py
python src/run_pipeline.py

streamlit run app/app.py

---

## 🔌 API Usage

uvicorn api.main:app --reload

Open:
http://127.0.0.1:8000/docs

---

## 📁 Project Structure

app/ - dashboard  
api/ - API  
src/ - core logic  
data/ - datasets  
outputs/ - results  
docs/ - documentation  

---

## 🌍 Note

Hybrid system:
- Real ML benchmarking
- Scenario-based modeling (energy, cost, carbon)

---

## 👨‍💻 Author

Ashiqur Rahman Rahul  
Berlin, Germany
