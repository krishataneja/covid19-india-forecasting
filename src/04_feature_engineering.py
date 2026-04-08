"""
04_feature_engineering.py — Create ML-ready features for forecasting.

Feature categories:
    1. Lag features: cases at t-1, t-2, t-3, t-4
    2. Rolling statistics: 4-week rolling mean, std, min, max
    3. Growth rate & reproduction number proxy
    4. Temporal encodings: week-of-year (sin/cos), month
    5. Mobility features (already merged)
    6. Demographic features (already merged)

Output: data/processed/features_ml.csv
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_PROCESSED, get_logger

logger = get_logger("04_features")


def add_lag_features(df: pd.DataFrame, target: str = "weekly_new_cases",
                     lags: list = [1, 2, 3, 4]) -> pd.DataFrame:
    """Add lagged values of the target variable per state."""
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("state")[target].shift(lag)
    logger.info(f"Added {len(lags)} lag features")
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "weekly_new_cases",
                         windows: list = [4, 8]) -> pd.DataFrame:
    """Rolling mean, std, min, max over specified windows."""
    for w in windows:
        grp = df.groupby("state")[target]
        df[f"roll_mean_{w}w"] = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}w"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0))
        df[f"roll_min_{w}w"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).min())
        df[f"roll_max_{w}w"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
    logger.info(f"Added rolling features for windows {windows}")
    return df


def add_growth_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Week-over-week growth rate and approximate Rt."""
    df["growth_rate"] = df.groupby("state")["weekly_new_cases"].pct_change().clip(-1, 10).fillna(0)

    # Simple Rt proxy: ratio of this week to previous week's cases
    df["rt_proxy"] = (df["weekly_new_cases"] /
                      df.groupby("state")["weekly_new_cases"].shift(1).clip(lower=1))
    df["rt_proxy"] = df["rt_proxy"].clip(0, 5).fillna(1)
    logger.info("Added growth rate and Rt proxy")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic temporal features to capture seasonality."""
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month

    # Sine/cosine encoding for week of year (captures annual seasonality)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # Time index (weeks since start) — captures overall trend
    min_date = df["week_start"].min()
    df["weeks_since_start"] = ((df["week_start"] - min_date).dt.days / 7).astype(int)

    logger.info("Added temporal features")
    return df


def add_death_case_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Case fatality rate proxy."""
    df["cfr"] = (df["total_deaths"] / df["total_confirmed"].clip(lower=1)).clip(0, 0.1)
    df["recovery_rate"] = (df["total_recovered"] / df["total_confirmed"].clip(lower=1)).clip(0, 1)
    logger.info("Added CFR and recovery rate")
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select final feature set and target, drop rows with NaNs from lagging."""
    target = "weekly_new_cases"

    # Feature columns
    feature_patterns = [
        "lag_", "roll_", "growth_rate", "rt_proxy",
        "week_sin", "week_cos", "month", "weeks_since_start",
        "mob_", "log_population", "log_density", "urban_pct", "literacy",
        "cfr", "recovery_rate", "active_cases",
    ]
    feature_cols = [c for c in df.columns
                    if any(c.startswith(p) or c == p for p in feature_patterns)]

    keep = ["state", "year_week", "week_start", target] + feature_cols
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Drop rows where lag features are NaN (first few weeks per state)
    before = len(df)
    df.dropna(subset=[c for c in df.columns if c.startswith("lag_")], inplace=True)
    logger.info(f"Dropped {before - len(df)} rows with NaN lags. Remaining: {len(df)}")

    # Fill any remaining NaN with 0
    df.fillna(0, inplace=True)
    return df


def main():
    logger.info("=" * 60)
    logger.info("STEP 4: FEATURE ENGINEERING")
    logger.info("=" * 60)

    path = os.path.join(DATA_PROCESSED, "weekly_state_panel.csv")
    df = pd.read_csv(path, parse_dates=["week_start", "week_end"])

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_growth_rate(df)
    df = add_temporal_features(df)
    df = add_death_case_ratio(df)
    df = select_features(df)

    out_path = os.path.join(DATA_PROCESSED, "features_ml.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")

    # Feature summary
    feature_cols = [c for c in df.columns if c not in ["state", "year_week", "week_start", "weekly_new_cases"]]
    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"Features: {feature_cols}")


if __name__ == "__main__":
    main()
