"""
01_download_data.py — Download real datasets OR generate realistic simulated data.

Usage:
    python src/01_download_data.py --simulate    # Generate synthetic data
    python src/01_download_data.py               # Verify real data in data/raw/

Real data sources:
    1. COVID-19 India:  https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-india
       → Place 'covid_19_india.csv' in data/raw/
    2. Google Mobility: https://www.google.com/covid19/mobility/
       → Place 'Global_Mobility_Report.csv' in data/raw/
    3. Demographics:    Census of India 2011 (embedded in utils.py)
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (DATA_RAW, INDIAN_STATES, STATE_DEMOGRAPHICS, get_logger)

logger = get_logger("01_download")

# ============================================================================
# Realistic COVID-19 data simulator
# ============================================================================

def _wave_curve(dates, peak_date, peak_val, width_days=45):
    """Generate a Gaussian-like wave of daily cases."""
    peak_idx = np.searchsorted(dates, np.datetime64(peak_date))
    x = np.arange(len(dates)) - peak_idx
    return peak_val * np.exp(-0.5 * (x / width_days) ** 2)


def simulate_covid_data() -> pd.DataFrame:
    """
    Generate realistic state-wise daily COVID-19 data for India
    mimicking two waves (Sep 2020 peak, May 2021 peak).

    Columns match the Kaggle covid_19_india.csv schema:
        Date, State, Confirmed, Cured, Deaths
    """
    logger.info("Generating simulated COVID-19 case data ...")
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-03-01", "2021-10-31", freq="D")
    records = []

    for state in INDIAN_STATES:
        pop = STATE_DEMOGRAPHICS.get(state, {}).get("population", 5_000_000)
        density = STATE_DEMOGRAPHICS.get(state, {}).get("density", 300)
        urban = STATE_DEMOGRAPHICS.get(state, {}).get("urban_pct", 30)

        # Scale factor: higher pop & urbanization → more cases
        scale = (pop / 1e8) * (1 + urban / 100) * (1 + density / 5000)

        # Wave 1: peaks around Sep 2020
        w1 = _wave_curve(dates.values, "2020-09-15",
                         peak_val=max(500, int(8000 * scale)), width_days=50)
        # Wave 2: peaks around May 2021 (Delta wave, ~3x bigger)
        w2 = _wave_curve(dates.values, "2021-05-10",
                         peak_val=max(1500, int(25000 * scale)), width_days=35)

        daily_cases = w1 + w2
        # Add noise
        daily_cases = np.maximum(0, daily_cases + rng.normal(0, daily_cases * 0.15))
        daily_cases = np.round(daily_cases).astype(int)

        # Cumulative confirmed
        confirmed = np.cumsum(daily_cases)

        # Recovery rate ~95-97%, deaths ~1.3-2%
        recovery_rate = rng.uniform(0.95, 0.97)
        death_rate = rng.uniform(0.013, 0.02)
        cured = np.round(confirmed * recovery_rate * rng.uniform(0.85, 1.0, size=len(dates))).astype(int)
        cured = np.minimum(cured, confirmed)
        deaths = np.round(confirmed * death_rate * rng.uniform(0.8, 1.1, size=len(dates))).astype(int)
        deaths = np.minimum(deaths, confirmed - cured)

        for i, d in enumerate(dates):
            records.append({
                "Date": d.strftime("%Y-%m-%d"),
                "State": state,
                "Confirmed": int(confirmed[i]),
                "Cured": int(cured[i]),
                "Deaths": int(deaths[i]),
            })

    df = pd.DataFrame(records)
    logger.info(f"  → {len(df):,} rows, {df['State'].nunique()} states, "
                f"{df['Date'].nunique()} days")
    return df


def simulate_mobility_data() -> pd.DataFrame:
    """
    Generate simulated Google Mobility data for Indian states.
    Columns mirror the official schema:
        date, sub_region_1, retail_and_recreation_percent_change_from_baseline, ...
    """
    logger.info("Generating simulated Google Mobility data ...")
    rng = np.random.default_rng(123)
    dates = pd.date_range("2020-02-15", "2021-10-31", freq="D")
    records = []

    categories = [
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]

    for state in INDIAN_STATES:
        for i, d in enumerate(dates):
            row = {"date": d.strftime("%Y-%m-%d"), "country_region": "India",
                   "sub_region_1": state}

            # Lockdown effect: sharp drop Mar-May 2020
            lockdown_factor = 1.0
            if pd.Timestamp("2020-03-25") <= d <= pd.Timestamp("2020-05-31"):
                lockdown_factor = 0.3 + 0.7 * ((d - pd.Timestamp("2020-03-25")).days / 67)
            elif pd.Timestamp("2021-04-15") <= d <= pd.Timestamp("2021-06-15"):
                lockdown_factor = 0.5 + 0.5 * ((d - pd.Timestamp("2021-04-15")).days / 61)

            for cat in categories:
                if "residential" in cat:
                    # Residential goes UP during lockdowns
                    base = rng.normal(5, 3) + (1 - lockdown_factor) * 25
                else:
                    # Other categories drop during lockdowns
                    base = rng.normal(-5, 8) - (1 - lockdown_factor) * 40
                row[cat] = round(base, 1)

            records.append(row)

    df = pd.DataFrame(records)
    logger.info(f"  → {len(df):,} rows")
    return df


def create_demographics_csv() -> pd.DataFrame:
    """Export the embedded Census 2011 demographics to CSV."""
    logger.info("Creating demographics CSV from Census 2011 data ...")
    rows = []
    for state, info in STATE_DEMOGRAPHICS.items():
        rows.append({"State": state, **info})
    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Data preparation")
    parser.add_argument("--simulate", action="store_true",
                        help="Generate synthetic data instead of using real CSVs")
    args = parser.parse_args()

    if args.simulate:
        logger.info("=== SIMULATION MODE ===")

        df_covid = simulate_covid_data()
        df_covid.to_csv(os.path.join(DATA_RAW, "covid_19_india.csv"), index=False)

        df_mob = simulate_mobility_data()
        df_mob.to_csv(os.path.join(DATA_RAW, "mobility_india.csv"), index=False)

        df_demo = create_demographics_csv()
        df_demo.to_csv(os.path.join(DATA_RAW, "india_demographics.csv"), index=False)

        logger.info("All simulated data saved to data/raw/")
    else:
        # Verify real data exists
        required = ["covid_19_india.csv"]
        for f in required:
            path = os.path.join(DATA_RAW, f)
            if os.path.exists(path):
                logger.info(f"✓ Found {f}")
            else:
                logger.warning(f"✗ Missing {f} — download from Kaggle or run with --simulate")

        # Always create demographics (embedded data)
        df_demo = create_demographics_csv()
        df_demo.to_csv(os.path.join(DATA_RAW, "india_demographics.csv"), index=False)
        logger.info("Demographics CSV created.")


if __name__ == "__main__":
    main()
