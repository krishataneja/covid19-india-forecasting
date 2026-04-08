"""
06_model_prophet.py — Seasonal decomposition + Ridge regression baseline.

When Prophet is unavailable, uses STL decomposition + Ridge regression
to capture trend, seasonality, and external regressors (mobility).
This serves the same purpose as Prophet: a decomposition-based baseline.
"""

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_PROCESSED, OUTPUTS_FIG, OUTPUTS_MODELS, get_logger, compute_metrics, save_metrics

logger = get_logger("06_prophet")


def load_data(test_weeks=8):
    df = pd.read_csv(os.path.join(DATA_PROCESSED, "features_ml.csv"),
                     parse_dates=["week_start"])
    df.sort_values(["state", "week_start"], inplace=True)
    all_weeks = sorted(df["week_start"].unique())
    cutoff = all_weeks[-test_weeks]
    return df, cutoff


def try_prophet(df, cutoff, top_n=10):
    """Attempt to use Prophet if installed."""
    try:
        from prophet import Prophet
    except ImportError:
        logger.info("Prophet not installed. Using STL + Ridge fallback.")
        return None, None

    top_states = df.groupby("state")["weekly_new_cases"].sum().nlargest(top_n).index.tolist()
    holidays = pd.DataFrame({
        "holiday": ["lockdown_1", "lockdown_2", "lockdown_3", "delta_restrictions"],
        "ds": pd.to_datetime(["2020-03-25", "2020-04-15", "2020-05-04", "2021-04-20"]),
        "lower_window": [0, 0, 0, 0],
        "upper_window": [21, 14, 14, 30],
    })

    all_actual, all_pred = [], []
    for state in top_states:
        sdf = df[df["state"] == state][["week_start", "weekly_new_cases"]].copy()
        sdf.rename(columns={"week_start": "ds", "weekly_new_cases": "y"}, inplace=True)
        train = sdf[sdf["ds"] < cutoff]
        test = sdf[sdf["ds"] >= cutoff]
        if len(train) < 10 or len(test) == 0:
            continue

        m = Prophet(changepoint_prior_scale=0.1, seasonality_prior_scale=5.0,
                    holidays=holidays, yearly_seasonality=True,
                    weekly_seasonality=False, daily_seasonality=False)
        m.fit(train.reset_index(drop=True))
        future = test[["ds"]].copy()
        forecast = m.predict(future)
        y_pred = forecast["yhat"].clip(lower=0).values[:len(test)]
        y_true = test["y"].values[:len(y_pred)]
        all_actual.extend(y_true)
        all_pred.extend(y_pred)
        logger.info(f"  {state}: RMSE={np.sqrt(np.mean((y_true-y_pred)**2)):.0f}")

    if all_actual:
        return np.array(all_actual), np.array(all_pred)
    return None, None


def stl_ridge_model(df, cutoff, top_n=10):
    """
    Fallback: Seasonal-Trend decomposition + Ridge regression.
    For each state:
        1. Decompose time series into trend + seasonal + residual
        2. Use lagged residuals + mobility features in Ridge regression
        3. Reconstruct forecast = predicted_trend + seasonal + predicted_residual
    """
    from scipy.signal import savgol_filter

    top_states = df.groupby("state")["weekly_new_cases"].sum().nlargest(top_n).index.tolist()
    mob_cols = [c for c in df.columns if c.startswith("mob_")]

    all_actual, all_pred = [], []
    state_metrics = {}

    for state in top_states:
        sdf = df[df["state"] == state].sort_values("week_start").copy()
        y = sdf["weekly_new_cases"].values.astype(float)
        n = len(y)

        if n < 15:
            continue

        # --- Trend extraction via Savitzky-Golay filter ---
        win = min(15, n if n % 2 == 1 else n - 1)
        if win < 5:
            win = 5 if n >= 5 else 3
        if win % 2 == 0:
            win -= 1
        trend = savgol_filter(y, window_length=win, polyorder=2)
        trend = np.clip(trend, 0, None)

        # --- Seasonal: average residual by week-of-year ---
        detrended = y - trend
        woy = sdf["week_start"].dt.isocalendar().week.astype(int).values
        seasonal = np.zeros(n)
        for w in np.unique(woy):
            mask = woy == w
            seasonal[mask] = detrended[mask].mean()

        residual = y - trend - seasonal

        # --- Build features for residual prediction ---
        feat_names = ["trend", "trend_diff", "seasonal"] + [f"res_lag{k}" for k in range(1, 5)]
        feat_names += mob_cols[:4]  # top mobility features

        features = np.column_stack([
            trend,
            np.concatenate([[0], np.diff(trend)]),
            seasonal,
            *[np.concatenate([np.zeros(k), residual[:-k]]) for k in range(1, 5)],
            *[sdf[mc].fillna(0).values for mc in mob_cols[:4]],
        ]) if mob_cols else np.column_stack([
            trend,
            np.concatenate([[0], np.diff(trend)]),
            seasonal,
            *[np.concatenate([np.zeros(k), residual[:-k]]) for k in range(1, 5)],
        ])

        # Train/test split
        train_mask = sdf["week_start"].values < np.datetime64(cutoff)
        test_mask = ~train_mask

        if test_mask.sum() == 0 or train_mask.sum() < 10:
            continue

        X_tr, y_tr = features[train_mask], y[train_mask]
        X_te, y_te = features[test_mask], y[test_mask]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr_sc, y_tr)
        y_p = ridge.predict(X_te_sc).clip(min=0)

        all_actual.extend(y_te)
        all_pred.extend(y_p)
        m = compute_metrics(y_te, y_p)
        state_metrics[state] = m
        logger.info(f"  {state}: RMSE={m['RMSE']:.0f}, R2={m['R2']:.3f}")

    if not all_actual:
        return None, None

    return np.array(all_actual), np.array(all_pred)


def plot_results(y_true, y_pred, label="Decomposition+Ridge"):
    if y_true is None:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#27ae60")
    lims = [0, max(y_true.max(), y_pred.max()) * 1.1]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Weekly Cases")
    ax.set_ylabel("Predicted Weekly Cases")
    ax.set_title(f"{label}: Actual vs Predicted (Top 10 States)", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_FIG, "prophet_predictions.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def main():
    logger.info("=" * 60)
    logger.info("STEP 6: DECOMPOSITION BASELINE MODEL")
    logger.info("=" * 60)

    df, cutoff = load_data()

    # Try Prophet first
    y_true, y_pred = try_prophet(df, cutoff)

    if y_true is None:
        # Fallback to STL + Ridge
        y_true, y_pred = stl_ridge_model(df, cutoff)
        label = "STL+Ridge"
    else:
        label = "Prophet"

    if y_true is not None:
        overall = compute_metrics(y_true, y_pred)
        logger.info(f"{label} overall: {overall}")
        save_metrics(overall, "decomposition_baseline")
        plot_results(y_true, y_pred, label)
    else:
        logger.warning("No predictions generated.")

    logger.info("Decomposition baseline complete.")


if __name__ == "__main__":
    main()
