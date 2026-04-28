import sys, json
from pathlib import Path
import streamlit as st
import plotly.express as px
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from data_loader import load_data
from engine import evaluate_all, recommend
st.set_page_config(page_title="AI Deployment Decision Engine", page_icon="🧠", layout="wide")
data = load_data()
st.title("🧠 AI Inference Optimization & Deployment Decision Engine")
st.caption("Constraint-based AI deployment selection across latency, accuracy, energy, carbon, and cost.")
with st.expander("What makes this more than a demo?", expanded=False):
    st.markdown("This app combines model profiles, deployment strategies, a real dataset benchmark, optional PyTorch timing, optional ONNX Runtime timing, and API deployment readiness.")
st.sidebar.header("Deployment Constraints")
model = st.sidebar.selectbox("Model", data["models"].model_name)
region = st.sidebar.selectbox("Region", data["regions"].region)
requests = st.sidebar.slider("Monthly requests", 100_000, 50_000_000, 5_000_000, 100_000)
latency = st.sidebar.slider("Max latency (ms)", 10, 500, 75, 5)
accuracy = st.sidebar.slider("Minimum accuracy", 0.50, 0.95, 0.70, 0.01)
cost = st.sidebar.slider("Max monthly cost (USD)", 10, 2000, 200, 10)
pue = st.sidebar.slider("PUE", 1.10, 2.00, 1.56, 0.01)
util = st.sidebar.slider("Utilization", 0.10, 0.95, 0.55, 0.05)
reports = ROOT / "outputs" / "reports"
measured_latency = measured_accuracy = None
source = None
acc_path = reports / "accuracy_benchmark_summary.json"
if acc_path.exists():
    s = json.loads(acc_path.read_text())
    if s.get("status") == "completed":
        measured_latency = s.get("best_mean_latency_ms"); measured_accuracy = s.get("best_accuracy"); source = "sklearn digits benchmark"
torch_path = reports / "real_benchmark_report.json"
if torch_path.exists():
    s = json.loads(torch_path.read_text())
    if s.get("status") == "completed":
        measured_latency = s.get("mean_latency_ms"); source = "PyTorch local benchmark"
onnx_path = reports / "onnx_benchmark_report.json"
if onnx_path.exists():
    s = json.loads(onnx_path.read_text())
    if s.get("status") == "completed":
        measured_latency = s.get("mean_latency_ms"); source = "ONNX Runtime local benchmark"
if measured_latency: st.sidebar.success(f"Using {source}: {measured_latency:.2f} ms")
if measured_accuracy: st.sidebar.success(f"Measured accuracy: {measured_accuracy:.3f}")
results = evaluate_all(data, model, region, requests, latency, accuracy, cost, pue, util, measured_latency_ms=measured_latency, measured_accuracy=measured_accuracy)
rec = recommend(results); best = results.iloc[0]
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Hardware", rec["hardware"]); c2.metric("Strategy", rec["strategy"]); c3.metric("Latency", f"{best.latency_ms:.1f} ms"); c4.metric("Accuracy", f"{best.estimated_accuracy:.3f}"); c5.metric("Cost", f"${best.monthly_cost_usd:.2f}")
st.info(rec["reason"])
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["Decision Overview","Trade-offs","All Configurations","Real Benchmark","Data & Methodology","References"])
with tab1:
    view = results.copy()
    view["Constraint Status"] = view.apply(lambda r: "Pass" if r.meets_latency and r.meets_accuracy and r.meets_cost else "Fail", axis=1)
    fig = px.scatter(view, x="latency_ms", y="monthly_cost_usd", color="Constraint Status", size="facility_energy_kwh", hover_data=["hardware","strategy","estimated_accuracy","energy_per_1000_inferences_wh"], title="Latency vs Cost with Energy as Bubble Size")
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    st.plotly_chart(px.scatter(results, x="latency_ms", y="energy_per_1000_inferences_wh", color="hardware", symbol="strategy", title="Latency vs Energy"), use_container_width=True)
    st.plotly_chart(px.scatter(results, x="estimated_accuracy", y="energy_per_1000_inferences_wh", color="strategy", hover_data=["hardware","monthly_cost_usd","latency_ms"], title="Accuracy vs Energy"), use_container_width=True)
with tab3:
    st.dataframe(results.sort_values("score"), use_container_width=True)
with tab4:
    st.code("python src/accuracy_benchmark.py")
    st.code("pip install torch torchvision\npython src/real_benchmark.py")
    st.code("pip install torch torchvision onnx onnxruntime\npython src/onnx_export.py\npython src/onnx_benchmark.py")
    if acc_path.exists(): st.json(json.loads(acc_path.read_text()))
    result_csv = reports / "accuracy_benchmark_results.csv"
    if result_csv.exists(): st.dataframe(pd.read_csv(result_csv), use_container_width=True)
    cm_path = ROOT / "outputs" / "figures" / "digits_confusion_matrix.png"
    if cm_path.exists(): st.image(str(cm_path), caption="Digits classification confusion matrix")
with tab5:
    st.dataframe(data["models"], use_container_width=True); st.dataframe(data["hardware"], use_container_width=True); st.dataframe(data["strategies"], use_container_width=True); st.dataframe(data["regions"], use_container_width=True)
with tab6:
    refs = ROOT / "docs" / "references.md"
    st.markdown(refs.read_text() if refs.exists() else "No references found.")
