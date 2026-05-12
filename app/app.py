import sys
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from data_loader import load_data
from engine import evaluate_all, recommend
from pareto import mark_pareto_frontier, objective_summary_table, normalize_for_comparison
from presets import SCENARIO_PRESETS, get_preset
from explainability import score_breakdown, recommendation_trace, executive_insights

st.set_page_config(page_title="AI Deployment Decision Engine", page_icon="⚙️", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --app-radius-lg: 1.15rem;
            --app-radius-md: 0.85rem;
            --app-border: rgba(128, 128, 128, 0.22);
            --app-blue: #2563eb;
            --app-green: #16a34a;
        }

        .main .block-container {
            padding-top: 1.75rem;
            padding-bottom: 2rem;
            max-width: 1520px;
        }

        .hero-card {
            padding: 1.35rem 1.5rem;
            border-radius: var(--app-radius-lg);
            border: 1px solid var(--app-border);
            background: linear-gradient(135deg, rgba(37,99,235,0.10), rgba(22,163,74,0.06));
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: clamp(1.75rem, 2.2vw, 2.35rem);
            line-height: 1.16;
            font-weight: 850;
            color: var(--text-color);
            margin-bottom: 0.35rem;
            letter-spacing: -0.035em;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            color: var(--text-color);
            opacity: 0.74;
            max-width: 980px;
        }

        .system-banner {
            padding: 0.78rem 1rem;
            border-radius: 0.85rem;
            border: 1px solid rgba(37,99,235,0.20);
            background: rgba(37,99,235,0.08);
            color: var(--text-color);
            margin: 0.8rem 0 0.9rem 0;
        }

        .summary-panel {
            padding: 1rem;
            border-radius: var(--app-radius-lg);
            border: 1px solid var(--app-border);
            background: rgba(128,128,128,0.045);
            margin-bottom: 1rem;
        }

        .insight-card {
            padding: 0.88rem 1rem;
            border-radius: var(--app-radius-md);
            background: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid var(--app-border);
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.045);
            margin-bottom: 0.6rem;
        }

        div[data-testid="stMetric"] {
            background: var(--secondary-background-color);
            border: 1px solid var(--app-border);
            padding: 1rem 1rem;
            border-radius: 1rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.14);
            border-color: rgba(37,99,235,0.35);
        }

        div[data-testid="stMetric"] * { color: var(--text-color) !important; }
        div[data-testid="stMetricLabel"] { opacity: 0.68; font-weight: 700; }
        div[data-testid="stMetricValue"] { font-weight: 750; letter-spacing: -0.03em; }

        button[data-baseweb="tab"] {
            font-size: 0.92rem;
            font-weight: 650;
            padding: 0.65rem 0.85rem;
            border-radius: 0.7rem 0.7rem 0 0;
        }
        button[data-baseweb="tab"]:hover { background-color: rgba(37,99,235,0.08); }
        button[aria-selected="true"] { border-bottom: 3px solid var(--app-blue) !important; }

        div[data-testid="stDataFrame"] { border-radius: 0.8rem; overflow: hidden; }
        code, pre { color: var(--text-color) !important; }

        section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { letter-spacing: -0.02em; }

        .app-footer {
            margin-top: 3rem;
            padding: 1.25rem 1rem;
            border-top: 1px solid var(--app-border);
            text-align: center;
            opacity: 0.78;
            font-size: 0.92rem;
            line-height: 1.55;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

px.defaults.template = "plotly_white"

data = load_data()
REPORTS_DIR = ROOT / "outputs" / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"


def read_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def show_file_status(label: str, path: Path):
    safe_path = path.as_posix()
    status_class = "background-color:#e8f7ee;border:1px solid #b7ebc6;color:#166534;" if path.exists() else "background-color:#fff7ed;border:1px solid #fed7aa;color:#9a3412;"
    status = "Found" if path.exists() else "Missing"
    st.markdown(
        f"""
        <div style="padding:0.65rem 0.85rem;border-radius:0.5rem;{status_class}margin-bottom:0.45rem;">
            <strong>{status}:</strong> {label}<br><code>{safe_path}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_metric(value, suffix="", digits=2):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return str(value)


def get_measured_metrics():
    measured_latency = None
    measured_accuracy = None
    benchmark_source = None

    accuracy_summary = read_json(REPORTS_DIR / "accuracy_benchmark_summary.json")
    if accuracy_summary and accuracy_summary.get("status") == "completed":
        measured_accuracy = accuracy_summary.get("best_accuracy")
        measured_latency = accuracy_summary.get("best_mean_latency_ms")
        benchmark_source = "sklearn digits benchmark"

    torch_report = read_json(REPORTS_DIR / "real_benchmark_report.json")
    if torch_report and torch_report.get("status") == "completed":
        measured_latency = torch_report.get("mean_latency_ms")
        benchmark_source = "PyTorch local benchmark"

    onnx_report = read_json(REPORTS_DIR / "onnx_benchmark_report.json")
    if onnx_report and onnx_report.get("status") == "completed":
        best_config = onnx_report.get("best_throughput_configuration")
        if best_config and best_config.get("mean_latency_ms") is not None:
            measured_latency = best_config.get("mean_latency_ms")
        elif onnx_report.get("mean_latency_ms") is not None:
            measured_latency = onnx_report.get("mean_latency_ms")
        benchmark_source = "ONNX Runtime benchmark"

    return measured_latency, measured_accuracy, benchmark_source


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">⚙️ AI Inference Optimization & Deployment Decision Engine</div>
        <div class="hero-subtitle">
            Constraint-based AI deployment selection across latency, accuracy, energy, carbon, queueing, autoscaling, and cost.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="system-banner">
        <strong>Production-style systems layer:</strong> workload-aware serving, queue delay, autoscaling,
        carbon-aware objectives, Pareto frontier analysis, recommendation trace, and API-ready deployment.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("What makes this more than a demo?", expanded=False):
    st.markdown(
        """
        This project evaluates AI deployment decisions under realistic systems constraints.

        It includes real benchmark results, ONNX Runtime benchmarking, workload-aware decision logic,
        queue delay modelling, autoscaling simulation, carbon-aware optimization, Pareto frontier analysis,
        recommendation traceability, and score decomposition.
        """
    )

st.sidebar.markdown("## 🚀 Presets")
preset_name = st.sidebar.selectbox("Preset", list(SCENARIO_PRESETS.keys()))
preset = get_preset(preset_name)
st.sidebar.caption(preset["description"])

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Configuration")
mode = st.sidebar.radio("Execution Mode", ["Measured (Benchmark-driven)", "Simulated"], index=0)
model = st.sidebar.selectbox("Model", data["models"].model_name)
region = st.sidebar.selectbox("Region", data["regions"].region)

requests = st.sidebar.slider("Monthly requests", 100_000, 50_000_000, int(preset["monthly_requests"]), 100_000)
latency = st.sidebar.slider("Max latency (ms)", 10, 500, int(preset["max_latency_ms"]), 5)
accuracy = st.sidebar.slider("Minimum accuracy", 0.50, 0.95, float(preset["min_accuracy"]), 0.01)
cost = st.sidebar.slider("Max monthly cost (USD)", 10, 2_000, int(preset["max_cost"]), 10)
pue = st.sidebar.slider("PUE", 1.10, 2.00, 1.56, 0.01)
util = st.sidebar.slider("Utilization", 0.10, 0.95, 0.55, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🧩 Systems Layer")
traffic_pattern = st.sidebar.selectbox("Traffic pattern", ["Steady", "Burst", "Spiky"], index=["Steady", "Burst", "Spiky"].index(preset["traffic_pattern"]))
serving_type = st.sidebar.selectbox("Serving type", ["Real-time", "Batch"], index=["Real-time", "Batch"].index(preset["serving_type"]))
deployment_type = st.sidebar.selectbox("Deployment type", ["Hybrid", "Cloud", "Edge"], index=["Hybrid", "Cloud", "Edge"].index(preset["deployment_type"]))
latency_slo = st.sidebar.slider("Latency SLO (ms)", 10, 500, int(preset["latency_slo_ms"]), 5)
batch_size = st.sidebar.selectbox("Inference batch size", [1, 4, 8, 16, 32], index=[1, 4, 8, 16, 32].index(preset["batch_size"]))
autoscaling_enabled = st.sidebar.checkbox("Enable autoscaling simulation", value=bool(preset["autoscaling_enabled"]))
max_instance_utilization = st.sidebar.slider("Target max instance utilization", 0.40, 0.95, float(preset["max_instance_utilization"]), 0.05)
optimize_for = st.sidebar.selectbox(
    "Optimization objective",
    ["Balanced", "Latency", "Cost", "Carbon", "Energy"],
    index=["Balanced", "Latency", "Cost", "Carbon", "Energy"].index(preset["optimize_for"]),
    help="Controls how the system ranks deployment options.",
)

measured_latency, measured_accuracy, benchmark_source = get_measured_metrics()
if mode == "Simulated":
    measured_latency = None
    measured_accuracy = None
    benchmark_source = None

st.sidebar.markdown("---")
st.sidebar.markdown("## 📡 Benchmark Layer")
if mode == "Simulated":
    st.sidebar.info("Simulation mode active. Benchmark values are ignored.")
else:
    if measured_latency:
        st.sidebar.success(f"Latency source: {benchmark_source}")
        st.sidebar.metric("Measured latency", f"{measured_latency:.2f} ms")
    else:
        st.sidebar.warning("No measured latency found. Using scenario latency.")
    if measured_accuracy:
        st.sidebar.metric("Measured accuracy", f"{measured_accuracy:.3f}")
    else:
        st.sidebar.warning("No measured accuracy found. Using scenario accuracy.")


def evaluate_for_objective(current_objective):
    return evaluate_all(
        data,
        model_name=model,
        region_name=region,
        monthly_requests=requests,
        max_latency_ms=latency,
        min_accuracy=accuracy,
        max_cost=cost,
        pue=pue,
        utilization=util,
        measured_latency_ms=measured_latency,
        measured_accuracy=measured_accuracy,
        traffic_pattern=traffic_pattern,
        serving_type=serving_type,
        deployment_type=deployment_type,
        latency_slo_ms=latency_slo,
        batch_size=batch_size,
        autoscaling_enabled=autoscaling_enabled,
        max_instance_utilization=max_instance_utilization,
        optimize_for=current_objective,
    )


with st.spinner("Evaluating deployment candidates..."):
    results = mark_pareto_frontier(evaluate_for_objective(optimize_for))
    recommendation = recommend(results)
    best = results.iloc[0]

st.markdown("## Executive Decision Summary")
st.markdown('<div class="summary-panel">', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Recommended Hardware", recommendation["hardware"])
k2.metric("Strategy", recommendation["strategy"])
k3.metric("Latency", f"{best.latency_ms:.1f} ms")
k4.metric("Accuracy", f"{best.estimated_accuracy:.3f}")
k5.metric("Monthly Cost", f"${best.monthly_cost_usd:.2f}")

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Objective", optimize_for)
s2.metric("Required Instances", int(best.required_instances))
s3.metric("Queue Delay", f"{best.queue_delay_ms:.2f} ms")
s4.metric("Carbon", f"{best.carbon_kgco2:.3f} kgCO₂")
s5.metric("Pareto Efficient", "Yes" if bool(best.is_pareto_efficient) else "No")

st.success(recommendation["reason"])
insight_cols = st.columns(2)
for i, insight in enumerate(executive_insights(best, results)):
    with insight_cols[i % 2]:
        st.markdown(f"<div class='insight-card'>{insight}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("⚙️ Why this configuration was selected", expanded=True):
    st.markdown(f"""
    **Selected Hardware:** {recommendation["hardware"]}  
    **Strategy:** {recommendation["strategy"]}  
    **Objective:** `{optimize_for}`

    **Systems context**
    - Traffic pattern: `{best.traffic_pattern}`
    - Serving type: `{best.serving_type}`
    - Deployment type: `{best.deployment_type}`
    - Hardware class: `{best.hardware_class}`
    - Batch size: `{best.batch_size}`
    - Required serving instances: `{int(best.required_instances)}`
    - Queue delay: **{best.queue_delay_ms:.2f} ms**
    - SLO violation: `{bool(best.violates_slo)}`
    - Pareto efficient: `{bool(best.is_pareto_efficient)}`

    **Key trade-offs**
    - Latency: **{best.latency_ms:.2f} ms**
    - Accuracy: **{best.estimated_accuracy:.3f}**
    - Energy per 1,000 inferences: **{best.energy_per_1000_inferences_wh:.3f} Wh**
    - Monthly energy: **{best.facility_energy_kwh:.2f} kWh**
    - Monthly carbon: **{best.carbon_kgco2:.3f} kgCO₂**
    - Monthly cost: **${best.monthly_cost_usd:.2f}**
    - Ranking score: **{best.score:.4f}**
    """)
    st.download_button("Download recommendation JSON", data=json.dumps(recommendation, indent=2), file_name="recommendation.json", mime="application/json")


def themed_chart(fig):
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=25), legend_title_text="", title_font_size=18)
    return fig


tabs = st.tabs([
    "Decision Overview", "Recommendation Trace", "Score Transparency", "Trade-offs",
    "Pareto Frontier", "Objective Comparison", "Systems Analysis", "SLO & Queueing",
    "All Configurations", "Real Benchmark", "ONNX Benchmark", "API & Deployment",
    "Data & Methodology", "References"
])

with tabs[0]:
    st.subheader("Deployment Decision Overview")
    view = results.copy()
    view["Constraint Status"] = view.apply(lambda row: "Pass" if row.meets_latency and row.meets_accuracy and row.meets_cost and not row.violates_slo else "Fail", axis=1)
    fig = px.scatter(view, x="latency_ms", y="monthly_cost_usd", color="Constraint Status", size="facility_energy_kwh", hover_data=["hardware", "strategy", "hardware_class", "carbon_kgco2", "is_pareto_efficient"], title="Latency vs Cost with Energy as Bubble Size")
    st.plotly_chart(themed_chart(fig), width="stretch")
    st.json(recommendation)

with tabs[1]:
    st.subheader("Live Recommendation Trace")
    trace = recommendation_trace(results)
    st.dataframe(trace, width="stretch")
    fig_trace = px.bar(trace, x="stage", y="remaining", text="remaining", title="Candidate Filtering Path")
    st.plotly_chart(themed_chart(fig_trace), width="stretch")
    rejected = recommendation.get("rejected_examples", [])
    st.markdown("### Rejected examples")
    if rejected:
        st.dataframe(pd.DataFrame(rejected), width="stretch")
    else:
        st.info("No rejected examples found.")

with tabs[2]:
    st.subheader("Weighted Score Transparency")
    breakdown = score_breakdown(best, latency, cost, accuracy, optimize_for)
    st.dataframe(breakdown, width="stretch")
    fig_breakdown = px.bar(breakdown, x="component", y="value", color="component", title="Approximate Score Contribution Breakdown")
    st.plotly_chart(themed_chart(fig_breakdown), width="stretch")
    st.caption("Lower contribution is better. This view explains the selected objective weighting.")

with tabs[3]:
    st.subheader("Trade-off Analysis")
    fig1 = px.scatter(results, x="latency_ms", y="energy_per_1000_inferences_wh", color="hardware_class", symbol="strategy", hover_data=["hardware", "estimated_accuracy", "monthly_cost_usd", "carbon_kgco2", "is_pareto_efficient"], title="Latency vs Energy per 1,000 Inferences")
    st.plotly_chart(themed_chart(fig1), width="stretch")
    fig2 = px.scatter(results, x="estimated_accuracy", y="energy_per_1000_inferences_wh", color="strategy", hover_data=["hardware", "monthly_cost_usd", "latency_ms", "carbon_kgco2"], title="Accuracy vs Energy")
    st.plotly_chart(themed_chart(fig2), width="stretch")

with tabs[4]:
    st.subheader("Pareto Frontier Analysis")
    st.markdown("Pareto-efficient configurations are not dominated across latency, monthly cost, and carbon emissions.")
    p1 = px.scatter(results, x="latency_ms", y="monthly_cost_usd", color="is_pareto_efficient", symbol="hardware_class", size="carbon_kgco2", hover_data=["hardware", "strategy", "carbon_kgco2", "energy_per_1000_inferences_wh", "score"], title="Pareto View: Latency vs Cost, Carbon as Bubble Size")
    st.plotly_chart(themed_chart(p1), width="stretch")
    p2 = px.scatter(results, x="latency_ms", y="carbon_kgco2", color="is_pareto_efficient", symbol="hardware_class", size="monthly_cost_usd", hover_data=["hardware", "strategy", "monthly_cost_usd", "energy_per_1000_inferences_wh", "score"], title="Pareto View: Latency vs Carbon, Cost as Bubble Size")
    st.plotly_chart(themed_chart(p2), width="stretch")
    pareto_cols = ["hardware", "hardware_class", "strategy", "latency_ms", "monthly_cost_usd", "carbon_kgco2", "energy_per_1000_inferences_wh", "estimated_accuracy", "required_instances", "score"]
    st.dataframe(results[results["is_pareto_efficient"]][pareto_cols].sort_values("score"), width="stretch")

with tabs[5]:
    st.subheader("Objective Comparison")
    objective_results = {}
    for obj in ["Balanced", "Latency", "Cost", "Carbon", "Energy"]:
        obj_df = mark_pareto_frontier(evaluate_for_objective(obj))
        objective_results[obj] = recommend(obj_df)["best_result"]
    objective_table = objective_summary_table(objective_results)
    st.dataframe(objective_table, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(themed_chart(px.bar(objective_table, x="objective", y="carbon_kgco2", color="hardware", title="Best Configuration Carbon by Objective")), width="stretch")
    with c2:
        st.plotly_chart(themed_chart(px.bar(objective_table, x="objective", y="monthly_cost_usd", color="hardware", title="Best Configuration Cost by Objective")), width="stretch")
    fig_obj = px.scatter(objective_table, x="latency_ms", y="carbon_kgco2", size="monthly_cost_usd", color="objective", hover_data=["hardware", "strategy", "accuracy", "energy_per_1000_inferences_wh"], title="Objective Switching Impact: Latency vs Carbon, Cost as Bubble Size")
    st.plotly_chart(themed_chart(fig_obj), width="stretch")
    normalized = normalize_for_comparison(objective_table, ["latency_ms", "monthly_cost_usd", "carbon_kgco2", "energy_per_1000_inferences_wh"])
    melted = normalized.melt(id_vars=["objective"], value_vars=["latency_ms_score", "monthly_cost_usd_score", "carbon_kgco2_score", "energy_per_1000_inferences_wh_score"], var_name="metric", value_name="normalized_score")
    st.plotly_chart(themed_chart(px.bar(melted, x="metric", y="normalized_score", color="objective", barmode="group", title="Normalized Objective Comparison: Higher Is Better")), width="stretch")

with tabs[6]:
    st.subheader("Systems Analysis: Scheduling, Serving, and Autoscaling")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(themed_chart(px.bar(results.groupby("hardware_class", as_index=False)["score"].mean(), x="hardware_class", y="score", title="Average Score by Hardware Class")), width="stretch")
    with c2:
        st.plotly_chart(themed_chart(px.bar(results.head(12), x="hardware", y="required_instances", color="hardware_class", title="Required Instances for Top Configurations")), width="stretch")
    cols = ["hardware", "hardware_class", "strategy", "serving_type", "deployment_type", "batch_size", "required_instances", "queue_delay_ms", "latency_ms", "energy_per_1000_inferences_wh", "monthly_cost_usd", "carbon_kgco2", "score"]
    st.dataframe(results[cols].head(20), width="stretch")

with tabs[7]:
    st.subheader("SLO Violation and Queueing Analysis")
    slo_df = results.copy()
    slo_df["SLO Status"] = slo_df["violates_slo"].map({True: "Violates SLO", False: "Meets SLO"})
    fig_slo = px.scatter(slo_df, x="queue_delay_ms", y="latency_ms", color="SLO Status", symbol="hardware_class", hover_data=["hardware", "strategy", "required_instances", "peak_arrival_rps", "throughput_rps"], title="Queue Delay vs End-to-End Latency")
    st.plotly_chart(themed_chart(fig_slo), width="stretch")
    fig_rps = px.scatter(slo_df, x="peak_arrival_rps", y="throughput_rps", color="hardware_class", symbol="SLO Status", hover_data=["hardware", "strategy", "required_instances"], title="Peak Arrival Rate vs Service Throughput")
    st.plotly_chart(themed_chart(fig_rps), width="stretch")
    st.dataframe(slo_df[["hardware", "hardware_class", "strategy", "latency_ms", "latency_slo_ms", "violates_slo", "queue_delay_ms", "required_instances", "peak_arrival_rps", "throughput_rps"]].sort_values("latency_ms"), width="stretch")

with tabs[8]:
    st.subheader("All Evaluated Configurations")
    st.dataframe(results.sort_values("score"), width="stretch")
    st.download_button("Download configuration results as CSV", data=results.to_csv(index=False).encode("utf-8"), file_name="deployment_configuration_results.csv", mime="text/csv")

with tabs[9]:
    st.subheader("Real Dataset Benchmark")
    st.code("python src/accuracy_benchmark.py", language="bash")
    benchmark_files = {"Accuracy benchmark summary": REPORTS_DIR / "accuracy_benchmark_summary.json", "Accuracy benchmark CSV": REPORTS_DIR / "accuracy_benchmark_results.csv", "Confusion matrix": FIGURES_DIR / "digits_confusion_matrix.png"}
    for label, path in benchmark_files.items():
        show_file_status(label, path)
    summary = read_json(benchmark_files["Accuracy benchmark summary"])
    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Best model", summary.get("best_model", "N/A"))
        c2.metric("Accuracy", safe_metric(summary.get("best_accuracy"), digits=3))
        c3.metric("Latency", safe_metric(summary.get("best_mean_latency_ms"), " ms", digits=3))
        st.json(summary)
    if benchmark_files["Accuracy benchmark CSV"].exists():
        st.dataframe(pd.read_csv(benchmark_files["Accuracy benchmark CSV"]), width="stretch")
    if benchmark_files["Confusion matrix"].exists():
        st.image(str(benchmark_files["Confusion matrix"]), caption="Digits classification confusion matrix")

with tabs[10]:
    st.subheader("ONNX Deployment Benchmark")
    st.info("Real deployment optimization workflow: PyTorch → ONNX → ONNX Runtime → latency benchmark.")
    st.code("pip install torch torchvision onnx onnxruntime onnxscript\npython src/onnx_export.py\npython src/onnx_benchmark.py\npython src/benchmark_comparison.py", language="bash")
    onnx_files = {"ONNX export report": REPORTS_DIR / "onnx_export_report.json", "ONNX benchmark report": REPORTS_DIR / "onnx_benchmark_report.json", "ONNX benchmark CSV": REPORTS_DIR / "onnx_benchmark_results.csv", "ONNX latency plot": FIGURES_DIR / "onnx_latency_by_batch.png", "Benchmark comparison report": REPORTS_DIR / "benchmark_comparison_report.json", "Benchmark comparison CSV": REPORTS_DIR / "benchmark_comparison.csv", "Benchmark comparison plot": FIGURES_DIR / "benchmark_latency_comparison.png"}
    for label, path in onnx_files.items():
        show_file_status(label, path)
    export_data = read_json(onnx_files["ONNX export report"])
    if export_data:
        st.markdown("### ONNX export report")
        st.json(export_data)
    benchmark_data = read_json(onnx_files["ONNX benchmark report"])
    if benchmark_data:
        best_cfg = benchmark_data.get("best_throughput_configuration", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Best batch size", best_cfg.get("batch_size", "N/A"))
        c2.metric("Latency", safe_metric(best_cfg.get("mean_latency_ms"), " ms", 2))
        c3.metric("Throughput", safe_metric(best_cfg.get("throughput_images_per_second"), " img/s", 1))
        st.json(benchmark_data)
    if onnx_files["ONNX benchmark CSV"].exists():
        st.dataframe(pd.read_csv(onnx_files["ONNX benchmark CSV"]), width="stretch")
    if onnx_files["ONNX latency plot"].exists():
        st.image(str(onnx_files["ONNX latency plot"]), caption="ONNX Runtime latency by batch size")
    if onnx_files["Benchmark comparison CSV"].exists():
        st.dataframe(pd.read_csv(onnx_files["Benchmark comparison CSV"]), width="stretch")
    if onnx_files["Benchmark comparison plot"].exists():
        st.image(str(onnx_files["Benchmark comparison plot"]), caption="Latency comparison across available benchmarks")

with tabs[11]:
    st.subheader("API & Deployment")
    st.code("uvicorn api.main:app --reload", language="bash")
    st.markdown("Local API documentation: `http://127.0.0.1:8000/docs`")
    st.markdown("Endpoint: `POST /recommend`")
    for doc in ["api_usage.md", "api_examples.md", "deployment_guide.md"]:
        api_doc = ROOT / "docs" / doc
        if api_doc.exists():
            st.divider()
            st.markdown(api_doc.read_text(encoding="utf-8"))

with tabs[12]:
    st.subheader("Data & Methodology")
    st.markdown("""
    Hybrid methodology:
    - real ML benchmarking
    - optional PyTorch and ONNX timing
    - queue delay modelling and autoscaling simulation
    - edge/cloud/hybrid scheduling logic
    - carbon-aware multi-objective scoring
    - Pareto frontier analysis
    - scenario-based energy, cost, and carbon modelling
    """)
    for doc in ["benchmarking_methodology.md", "executive_summary.md"]:
        path = ROOT / "docs" / doc
        if path.exists():
            st.markdown(path.read_text(encoding="utf-8"))
            st.divider()
    st.markdown("### Model profiles")
    st.dataframe(data["models"], width="stretch")
    st.markdown("### Hardware profiles")
    st.dataframe(data["hardware"], width="stretch")
    st.markdown("### Optimization strategies")
    st.dataframe(data["strategies"], width="stretch")
    st.markdown("### Region profiles")
    st.dataframe(data["regions"], width="stretch")

with tabs[13]:
    st.subheader("References")
    for doc in ["references.md", "onnx_benchmarking.md", "faang_systems_extensions.md", "carbon_aware_optimization.md", "pareto_frontier.md"]:
        path = ROOT / "docs" / doc
        if path.exists():
            st.markdown(path.read_text(encoding="utf-8"))
            st.divider()

st.markdown(
    """
    <div class="app-footer">
        <strong>AI Deployment Decision Engine</strong><br>
        Built with Streamlit · ONNX Runtime · Pareto Optimization · Energy-Aware Scheduling · Carbon-Aware Infrastructure Logic
    </div>
    """,
    unsafe_allow_html=True,
)
