"""
02_preprocess.py — Clean, validate, and merge raw datasets into a weekly
state-level panel dataset ready for feature engineering.

Steps:
    1. Load & clean COVID-19 case data (daily → weekly aggregation)
    2. Load & clean Google Mobility data (daily → weekly mean)
    3. Merge with demographic features
    4. Handle missing values via forward-fill + interpolation
    5. Output: data/processed/weekly_state_panel.csv
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_RAW, DATA_PROCESSED, INDIAN_STATES, STATE_DEMOGRAPHICS, get_logger

logger = get_logger("02_preprocess")


def load_covid_data() -> pd.DataFrame:
    """Load and standardise COVID-19 case data."""
    path = os.path.join(DATA_RAW, "covid_19_india.csv")
    df = pd.read_csv(path)
    logger.info(f"Loaded COVID data: {df.shape}")

    # Standardise column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if "date" in cl:
            col_map[c] = "date"
        elif "state" in cl or "region" in cl:
            col_map[c] = "state"
        elif "confirm" in cl:
            col_map[c] = "confirmed"
        elif "cure" in cl or "recover" in cl:
            col_map[c] = "recovered"
        elif "death" in cl:
            col_map[c] = "deaths"
    df.rename(columns=col_map, inplace=True)

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    for col in ["confirmed", "recovered", "deaths"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Filter to known states
    df = df[df["state"].isin(INDIAN_STATES)].copy()
    df.sort_values(["state", "date"], inplace=True)
    logger.info(f"After cleaning: {df.shape}, states: {df['state'].nunique()}")
    return df


def compute_daily_new_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Derive daily new cases from cumulative confirmed."""
    df = df.sort_values(["state", "date"])
    df["new_cases"] = df.groupby("state")["confirmed"].diff().clip(lower=0).fillna(0).astype(int)
    df["new_deaths"] = df.groupby("state")["deaths"].diff().clip(lower=0).fillna(0).astype(int)
    df["new_recovered"] = df.groupby("state")["recovered"].diff().clip(lower=0).fillna(0).astype(int)
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily data to ISO week level."""
    df["year"] = df["date"].dt.isocalendar().year.astype(int)
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["year_week"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)

    weekly = df.groupby(["state", "year_week", "year", "week"]).agg(
        weekly_new_cases=("new_cases", "sum"),
        weekly_new_deaths=("new_deaths", "sum"),
        weekly_new_recovered=("new_recovered", "sum"),
        total_confirmed=("confirmed", "last"),
        total_deaths=("deaths", "last"),
        total_recovered=("recovered", "last"),
        week_start=("date", "min"),
        week_end=("date", "max"),
    ).reset_index()

    weekly["active_cases"] = weekly["total_confirmed"] - weekly["total_recovered"] - weekly["total_deaths"]
    weekly["active_cases"] = weekly["active_cases"].clip(lower=0)
    logger.info(f"Weekly aggregation: {weekly.shape}")
    return weekly


def load_mobility_data() -> pd.DataFrame:
    """Load Google Mobility data, filter for India, aggregate weekly."""
    path = os.path.join(DATA_RAW, "mobility_india.csv")
    if not os.path.exists(path):
        # Try the global file and filter
        global_path = os.path.join(DATA_RAW, "Global_Mobility_Report.csv")
        if os.path.exists(global_path):
            logger.info("Filtering India from Global Mobility Report ...")
            df = pd.read_csv(global_path, low_memory=False)
            df = df[df["country_region"] == "India"].copy()
        else:
            logger.warning("No mobility data found. Skipping.")
            return pd.DataFrame()
    else:
        df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    # Use sub_region_1 as state
    state_col = "sub_region_1"
    if state_col not in df.columns:
        logger.warning("No sub_region_1 column in mobility data.")
        return pd.DataFrame()

    df = df[df[state_col].isin(INDIAN_STATES)].copy()
    df.rename(columns={state_col: "state"}, inplace=True)

    mobility_cols = [c for c in df.columns if "percent_change" in c]

    # Weekly mean
    df["year"] = df["date"].dt.isocalendar().year.astype(int)
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["year_week"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)

    weekly_mob = df.groupby(["state", "year_week"])[mobility_cols].mean().round(2).reset_index()

    # Shorten column names for convenience
    rename = {}
    for c in mobility_cols:
        short = c.replace("_percent_change_from_baseline", "")
        rename[c] = f"mob_{short}"
    weekly_mob.rename(columns=rename, inplace=True)

    logger.info(f"Weekly mobility: {weekly_mob.shape}")
    return weekly_mob


def merge_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Census 2011 demographic features."""
    demo_rows = [{"state": s, **v} for s, v in STATE_DEMOGRAPHICS.items()]
    demo_df = pd.DataFrame(demo_rows)

    # Normalise population to log scale for modelling
    demo_df["log_population"] = np.log10(demo_df["population"])
    demo_df["log_density"] = np.log10(demo_df["density"].clip(lower=1))

    df = df.merge(demo_df, on="state", how="left")
    logger.info(f"After demographic merge: {df.shape}")
    return df


def main():
    logger.info("=" * 60)
    logger.info("STEP 2: PREPROCESSING")
    logger.info("=" * 60)

    # 1. COVID data
    covid = load_covid_data()
    covid = compute_daily_new_cases(covid)
    weekly = aggregate_weekly(covid)

    # 2. Mobility
    mob = load_mobility_data()
    if not mob.empty:
        weekly = weekly.merge(mob, on=["state", "year_week"], how="left")
        # Forward-fill mobility gaps within each state
        mob_cols = [c for c in weekly.columns if c.startswith("mob_")]
        weekly[mob_cols] = weekly.groupby("state")[mob_cols].transform(
            lambda x: x.ffill().bfill().fillna(0)
        )

    # 3. Demographics
    weekly = merge_demographics(weekly)

    # 4. Sort and save
    weekly.sort_values(["state", "year_week"], inplace=True)
    weekly.reset_index(drop=True, inplace=True)

    out_path = os.path.join(DATA_PROCESSED, "weekly_state_panel.csv")
    weekly.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}  ({weekly.shape[0]:,} rows × {weekly.shape[1]} cols)")

    # Summary stats
    logger.info(f"Date range: {weekly['week_start'].min()} → {weekly['week_end'].max()}")
    logger.info(f"States: {weekly['state'].nunique()}")
    logger.info(f"Total weekly new cases range: {weekly['weekly_new_cases'].min()} – {weekly['weekly_new_cases'].max()}")


if __name__ == "__main__":
    main()
