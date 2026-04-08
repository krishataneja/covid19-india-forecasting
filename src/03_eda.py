"""
03_eda.py — Exploratory data analysis with publication-quality visualisations.

Outputs saved to outputs/figures/:
    - national_weekly_cases.png
    - top10_state_heatmap.png
    - mobility_vs_cases.png
    - correlation_matrix.png
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_PROCESSED, OUTPUTS_FIG, get_logger

logger = get_logger("03_eda")
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150


def load_data():
    path = os.path.join(DATA_PROCESSED, "weekly_state_panel.csv")
    df = pd.read_csv(path, parse_dates=["week_start", "week_end"])
    logger.info(f"Loaded {df.shape}")
    return df


def plot_national_curve(df):
    """National weekly new cases with wave annotations."""
    national = df.groupby("week_start")["weekly_new_cases"].sum().reset_index()
    national.sort_values("week_start", inplace=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(national["week_start"], national["weekly_new_cases"],
                    alpha=0.35, color="#e74c3c")
    ax.plot(national["week_start"], national["weekly_new_cases"],
            color="#c0392b", linewidth=1.5)

    ax.set_title("India — Weekly New COVID-19 Cases", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Weekly New Cases")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    # Annotate waves
    peak1 = national.loc[
        (national["week_start"] >= "2020-07-01") & (national["week_start"] <= "2020-12-01")
    ]["weekly_new_cases"].idxmax()
    peak2 = national.loc[
        (national["week_start"] >= "2021-03-01") & (national["week_start"] <= "2021-07-01")
    ]["weekly_new_cases"].idxmax()

    for peak, label in [(peak1, "Wave 1"), (peak2, "Wave 2 (Delta)")]:
        ax.annotate(label,
                    xy=(national.loc[peak, "week_start"], national.loc[peak, "weekly_new_cases"]),
                    xytext=(0, 20), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black"))

    plt.tight_layout()
    path = os.path.join(OUTPUTS_FIG, "national_weekly_cases.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_state_heatmap(df):
    """Heatmap of top 10 states' weekly cases over time."""
    top10 = df.groupby("state")["weekly_new_cases"].sum().nlargest(10).index.tolist()
    subset = df[df["state"].isin(top10)].copy()

    pivot = subset.pivot_table(index="state", columns="week_start",
                               values="weekly_new_cases", aggfunc="sum")
    # Log transform for better visibility
    pivot_log = np.log10(pivot.clip(lower=1))

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot_log, cmap="YlOrRd", ax=ax, cbar_kws={"label": "log₁₀(weekly cases)"})
    ax.set_title("Top 10 States — Weekly COVID-19 Cases (log scale)", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Reduce x-tick density
    n_ticks = len(pivot.columns)
    step = max(1, n_ticks // 12)
    ax.set_xticks(range(0, n_ticks, step))
    ax.set_xticklabels([pivot.columns[i].strftime("%b %Y") for i in range(0, n_ticks, step)],
                       rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_FIG, "top10_state_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_mobility_vs_cases(df):
    """Scatter of mobility change vs case growth for key categories."""
    mob_cols = [c for c in df.columns if c.startswith("mob_")]
    if not mob_cols:
        logger.warning("No mobility columns found, skipping mobility plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for i, col in enumerate(mob_cols[:6]):
        ax = axes[i]
        # Sample for readability
        sample = df.dropna(subset=[col, "weekly_new_cases"])
        if len(sample) > 2000:
            sample = sample.sample(2000, random_state=42)
        ax.scatter(sample[col], np.log10(sample["weekly_new_cases"].clip(lower=1)),
                   alpha=0.25, s=8, color="#2c3e50")
        ax.set_xlabel(col.replace("mob_", "").replace("_", " ").title())
        ax.set_ylabel("log₁₀(cases)")
        ax.set_title(col.replace("mob_", "").replace("_", " ").title(), fontsize=10)

    fig.suptitle("Google Mobility vs Weekly Cases", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_FIG, "mobility_vs_cases.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_correlation_matrix(df):
    """Correlation heatmap of all numeric features."""
    numeric = df.select_dtypes(include=[np.number])
    # Drop columns with near-zero variance
    numeric = numeric.loc[:, numeric.std() > 0.01]
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, annot=False,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.7, "label": "Pearson r"})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_FIG, "correlation_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def main():
    logger.info("=" * 60)
    logger.info("STEP 3: EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    df = load_data()

    # Print summary statistics
    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Date range: {df['week_start'].min().date()} → {df['week_end'].max().date()}")
    logger.info(f"States: {df['state'].nunique()}")
    logger.info(f"\nWeekly cases stats:\n{df['weekly_new_cases'].describe()}")

    # Generate all plots
    plot_national_curve(df)
    plot_state_heatmap(df)
    plot_mobility_vs_cases(df)
    plot_correlation_matrix(df)

    logger.info("EDA complete. All figures saved to outputs/figures/")


if __name__ == "__main__":
    main()
