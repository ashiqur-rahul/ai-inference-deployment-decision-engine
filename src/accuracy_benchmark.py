from pathlib import Path
import json
import time
import pickle
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "models"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def run_digits_accuracy_benchmark():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=42, stratify=digits.target)

    candidates = {
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000))]),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42)
    }

    rows = []
    best_model_name = None
    best_acc = -1
    best_model = None
    best_y_pred = None

    for name, model in candidates.items():
        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - train_start
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        timings = []
        for sample in X_test[:200]:
            start = time.perf_counter()
            _ = model.predict([sample])
            timings.append((time.perf_counter() - start) * 1000)

        mean_latency = sum(timings) / len(timings)
        rows.append({"model": name, "dataset": "sklearn_digits", "accuracy": acc, "mean_latency_ms": mean_latency, "train_time_seconds": train_time, "test_samples": len(X_test)})
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = model
            best_y_pred = y_pred

    results = pd.DataFrame(rows)
    results.to_csv(REPORT_DIR / "accuracy_benchmark_results.csv", index=False)
    (REPORT_DIR / "digits_classification_report.txt").write_text(classification_report(y_test, best_y_pred), encoding="utf-8")

    with open(MODEL_DIR / "best_digits_classifier.pkl", "wb") as f:
        pickle.dump(best_model, f)

    cm = confusion_matrix(y_test, best_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Digits Classification Confusion Matrix ({best_model_name})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "digits_confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close()

    result = {"status": "completed", "dataset": "sklearn_digits", "best_model": best_model_name, "best_accuracy": float(best_acc), "best_mean_latency_ms": float(results.loc[results.model == best_model_name, "mean_latency_ms"].iloc[0]), "note": "Real local dataset benchmark using sklearn digits."}
    (REPORT_DIR / "accuracy_benchmark_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    print(json.dumps(run_digits_accuracy_benchmark(), indent=2))
