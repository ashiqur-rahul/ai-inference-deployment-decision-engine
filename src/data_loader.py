from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def load_data():
    return {
        "models": pd.read_csv(DATA_DIR / "model_profiles.csv"),
        "hardware": pd.read_csv(DATA_DIR / "hardware_profiles.csv"),
        "strategies": pd.read_csv(DATA_DIR / "optimization_strategies.csv"),
        "regions": pd.read_csv(DATA_DIR / "region_profiles.csv"),
    }
