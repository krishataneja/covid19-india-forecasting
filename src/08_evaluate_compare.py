"""
08_evaluate_compare.py — Compare all trained models and produce final summary.

Outputs:
    - Model comparison table (printed + saved as CSV)
    - Comparison bar chart
    - Final summary statistics
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUTS_FIG, OUTPUTS_MODELS, get_logger, load_all_metrics

logger = get_logger("08_compare")


def plot_comparison(metrics_df):
    """Bar chart comparing models across metrics."""
    if metrics_df.empty:
        logger.warning("No metrics to compare.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors = {"xgboost": "#3498db", "prophet": "#27ae60", "lstm": "#8e44ad"}

    for i, metric in enumerate(["RMSE", "MAE", "MAPE (%)", "R2"]):
        ax = axes[i]
        if metric not in metrics_df.columns:
            continue
        vals = metrics_df[metric]
        bars = ax.bar(vals.index, vals.values,
                      color=[colors.get(m, "#95a5a6") for m in vals.index])
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric)

        # Add value labels
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}" if metric != "R2" else f"{val:.3f}",
                    ha="center", va="bottom", fontsize=10)

    fig.suptitle("Model Comparison — COVID-19 Weekly Case Forecasting",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_FIG, "model_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def main():
    logger.info("=" * 60)
    logger.info("STEP 8: MODEL COMPARISON")
    logger.info("=" * 60)

    metrics_df = load_all_metrics()

    if metrics_df.empty:
        logger.warning("No model metrics found. Run model training scripts first.")
        return

    # Display comparison table
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 60)
    print(metrics_df.to_string())

    # Save comparison
    metrics_df.to_csv(os.path.join(OUTPUTS_MODELS, "model_comparison.csv"))

    # Best model
    if "RMSE" in metrics_df.columns:
        best = metrics_df["RMSE"].idxmin()
        logger.info(f"\nBest model by RMSE: {best} ({metrics_df.loc[best, 'RMSE']:.1f})")

    if "R2" in metrics_df.columns:
        best_r2 = metrics_df["R2"].idxmax()
        logger.info(f"Best model by R²: {best_r2} ({metrics_df.loc[best_r2, 'R2']:.4f})")

    plot_comparison(metrics_df)

    logger.info("\n✓ All results saved to outputs/")
    logger.info("✓ Figures saved to outputs/figures/")
    logger.info("✓ Model artifacts saved to outputs/models/")


if __name__ == "__main__":
    main()
