"""
utils.py — Shared utility functions for COVID-19 forecasting pipeline.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUTS_FIG = os.path.join(PROJECT_ROOT, "outputs", "figures")
OUTPUTS_MODELS = os.path.join(PROJECT_ROOT, "outputs", "models")

for _d in [DATA_RAW, DATA_PROCESSED, OUTPUTS_FIG, OUTPUTS_MODELS]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dict of standard regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0  # avoid division by zero for MAPE
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE (%)": round(mape, 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }

def save_metrics(metrics: dict, model_name: str):
    """Persist metrics to a JSON file for later comparison."""
    path = os.path.join(OUTPUTS_MODELS, f"metrics_{model_name}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def load_all_metrics() -> pd.DataFrame:
    """Load all saved metric JSON files and return a comparison DataFrame."""
    rows = []
    for fname in sorted(os.listdir(OUTPUTS_MODELS)):
        if fname.startswith("metrics_") and fname.endswith(".json"):
            with open(os.path.join(OUTPUTS_MODELS, fname)) as f:
                d = json.load(f)
            d["model"] = fname.replace("metrics_", "").replace(".json", "")
            rows.append(d)
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()

# ---------------------------------------------------------------------------
# Indian states list (major states + UTs used in analysis)
# ---------------------------------------------------------------------------
INDIAN_STATES = [
    "Maharashtra", "Kerala", "Karnataka", "Tamil Nadu", "Andhra Pradesh",
    "Uttar Pradesh", "West Bengal", "Delhi", "Rajasthan", "Gujarat",
    "Madhya Pradesh", "Bihar", "Odisha", "Telangana", "Haryana",
    "Punjab", "Assam", "Jharkhand", "Chhattisgarh", "Uttarakhand",
    "Goa", "Jammu and Kashmir", "Himachal Pradesh", "Puducherry",
    "Chandigarh", "Manipur", "Meghalaya", "Tripura", "Mizoram",
    "Nagaland", "Arunachal Pradesh", "Sikkim", "Ladakh",
]

# ---------------------------------------------------------------------------
# State-level demographic data (Census 2011 + estimates)
# ---------------------------------------------------------------------------
STATE_DEMOGRAPHICS = {
    "Maharashtra":       {"population": 112374333, "density": 365, "urban_pct": 45.2, "literacy": 82.3},
    "Kerala":            {"population": 33406061,  "density": 860, "urban_pct": 47.7, "literacy": 94.0},
    "Karnataka":         {"population": 61095297,  "density": 319, "urban_pct": 38.6, "literacy": 75.4},
    "Tamil Nadu":        {"population": 72147030,  "density": 555, "urban_pct": 48.4, "literacy": 80.1},
    "Andhra Pradesh":    {"population": 49577103,  "density": 308, "urban_pct": 33.5, "literacy": 67.0},
    "Uttar Pradesh":     {"population": 199812341, "density": 829, "urban_pct": 22.3, "literacy": 67.7},
    "West Bengal":       {"population": 91276115,  "density": 1028,"urban_pct": 31.9, "literacy": 76.3},
    "Delhi":             {"population": 16787941,  "density": 11320,"urban_pct": 97.5,"literacy": 86.2},
    "Rajasthan":         {"population": 68548437,  "density": 200, "urban_pct": 24.9, "literacy": 66.1},
    "Gujarat":           {"population": 60439692,  "density": 308, "urban_pct": 42.6, "literacy": 78.0},
    "Madhya Pradesh":    {"population": 72626809,  "density": 236, "urban_pct": 27.6, "literacy": 69.3},
    "Bihar":             {"population": 104099452, "density": 1106,"urban_pct": 11.3, "literacy": 61.8},
    "Odisha":            {"population": 41974218,  "density": 270, "urban_pct": 16.7, "literacy": 72.9},
    "Telangana":         {"population": 35003674,  "density": 312, "urban_pct": 38.9, "literacy": 66.5},
    "Haryana":           {"population": 25351462,  "density": 573, "urban_pct": 34.8, "literacy": 75.6},
    "Punjab":            {"population": 27743338,  "density": 551, "urban_pct": 37.5, "literacy": 75.8},
    "Assam":             {"population": 31205576,  "density": 398, "urban_pct": 14.1, "literacy": 72.2},
    "Jharkhand":         {"population": 32988134,  "density": 414, "urban_pct": 24.1, "literacy": 66.4},
    "Chhattisgarh":      {"population": 25545198,  "density": 189, "urban_pct": 23.2, "literacy": 70.3},
    "Uttarakhand":       {"population": 10086292,  "density": 189, "urban_pct": 30.2, "literacy": 78.8},
    "Goa":               {"population": 1458545,   "density": 394, "urban_pct": 62.2, "literacy": 88.7},
    "Jammu and Kashmir": {"population": 12541302,  "density": 124, "urban_pct": 27.4, "literacy": 67.2},
    "Himachal Pradesh":  {"population": 6864602,   "density": 123, "urban_pct": 10.0, "literacy": 82.8},
    "Puducherry":        {"population": 1247953,   "density": 2598,"urban_pct": 68.3, "literacy": 85.8},
    "Chandigarh":        {"population": 1055450,   "density": 9252,"urban_pct": 97.3, "literacy": 86.0},
    "Manipur":           {"population": 2570390,   "density": 115, "urban_pct": 32.5, "literacy": 76.9},
    "Meghalaya":         {"population": 2966889,   "density": 132, "urban_pct": 20.1, "literacy": 74.4},
    "Tripura":           {"population": 3673917,   "density": 350, "urban_pct": 26.2, "literacy": 87.2},
    "Mizoram":           {"population": 1097206,   "density": 52,  "urban_pct": 52.1, "literacy": 91.3},
    "Nagaland":          {"population": 1978502,   "density": 119, "urban_pct": 28.9, "literacy": 79.6},
    "Arunachal Pradesh": {"population": 1383727,   "density": 17,  "urban_pct": 22.7, "literacy": 65.4},
    "Sikkim":            {"population": 610577,    "density": 86,  "urban_pct": 25.0, "literacy": 81.4},
    "Ladakh":            {"population": 274000,    "density": 3,   "urban_pct": 26.0, "literacy": 77.0},
}
