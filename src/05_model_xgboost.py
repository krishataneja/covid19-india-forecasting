"""
05_model_xgboost.py — Gradient Boosting model for COVID-19 forecasting.

Uses sklearn GradientBoostingRegressor. If xgboost is installed, uses XGBRegressor.

Strategy:
    - Temporal train/test split (last 8 weeks as test)
    - TimeSeriesSplit cross-validation (5 folds)
    - Hyperparameter grid search
    - Feature importance analysis
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

try:
    import xgboost as xgb
    USE_XGB = True
except ImportError:
    USE_XGB = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_PROCESSED, OUTPUTS_FIG, OUTPUTS_MODELS, get_logger, compute_metrics, save_metrics

logger = get_logger("05_xgboost")


def load_and_split(test_weeks: int = 8):
    df = pd.read_csv(os.path.join(DATA_PROCESSED, "features_ml.csv"),
                     parse_dates=["week_start"])
    df.sort_values(["state", "week_start"], inplace=True)
    all_weeks = sorted(df["week_start"].unique())
    cutoff = all_weeks[-test_weeks]
    logger.info(f"Train/test cutoff: {pd.Timestamp(cutoff).date()}")
    train = df[df["week_start"] < cutoff].copy()
    test = df[df["week_start"] >= cutoff].copy()
    logger.info(f"Train: {train.shape}, Test: {test.shape}")
    return train, test, df


def get_feature_cols(df):
    exclude = {"state", "year_week", "week_start", "weekly_new_cases"}
    return [c for c in df.columns if c not in exclude]


def train_model(train, test):
    feature_cols = get_feature_cols(train)
    target = "weekly_new_cases"
    X_train = train[feature_cols].values
    y_train = train[target].values
    X_test = test[feature_cols].values
    y_test = test[target].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    param_grid = [
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.8, "min_samples_leaf": 5},
        {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.03,
         "subsample": 0.7, "min_samples_leaf": 10},
        {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
         "subsample": 0.9, "min_samples_leaf": 8},
    ]

    tscv = TimeSeriesSplit(n_splits=5)
    best_rmse = float("inf")
    best_params = {}

    backend = "XGBoost" if USE_XGB else "sklearn GradientBoosting"
    logger.info(f"Using {backend}. Grid search: {len(param_grid)} configs x 5 folds ...")

    for params in param_grid:
        fold_rmses = []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_sc)):
            if USE_XGB:
                model = xgb.XGBRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    subsample=params["subsample"],
                    reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, verbosity=0, tree_method="hist",
                )
                model.fit(X_train_sc[tr_idx], y_train[tr_idx],
                          eval_set=[(X_train_sc[va_idx], y_train[va_idx])],
                          verbose=False)
            else:
                model = GradientBoostingRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    subsample=params["subsample"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=42,
                )
                model.fit(X_train_sc[tr_idx], y_train[tr_idx])

            preds = model.predict(X_train_sc[va_idx])
            rmse = np.sqrt(np.mean((y_train[va_idx] - preds) ** 2))
            fold_rmses.append(rmse)

        mean_rmse = np.mean(fold_rmses)
        logger.info(f"  depth={params['max_depth']}, lr={params['learning_rate']}, "
                     f"n={params['n_estimators']} -> CV RMSE: {mean_rmse:.1f}")
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params

    logger.info(f"Best params: {best_params}, CV RMSE: {best_rmse:.1f}")

    if USE_XGB:
        final_model = xgb.XGBRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=best_params["subsample"],
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0, tree_method="hist",
        )
    else:
        final_model = GradientBoostingRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=best_params["subsample"],
            min_samples_leaf=best_params.get("min_samples_leaf", 5),
            random_state=42,
        )
    final_model.fit(X_train_sc, y_train)
    y_pred = final_model.predict(X_test_sc).clip(min=0)

    metrics = compute_metrics(y_test, y_pred)
    logger.info(f"Test metrics: {metrics}")
    save_metrics(metrics, "gradient_boosting")

    return final_model, y_test, y_pred, feature_cols, scaler


def plot_feature_importance(model, feature_cols):
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({"feature": feature_cols, "importance": importance})
    feat_imp = feat_imp.nlargest(20, "importance")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=feat_imp, y="feature", x="importance", ax=ax, color="#3498db")
    ax.set_title("Gradient Boosting — Top 20 Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance (Impurity Reduction)")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_FIG, "xgboost_feature_importance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_predictions(test, y_test, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="#2c3e50")
    lims = [0, max(y_test.max(), y_pred.max()) * 1.1]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Weekly Cases")
    ax.set_ylabel("Predicted Weekly Cases")
    ax.set_title("Gradient Boosting: Actual vs Predicted")
    ax.legend()

    test_df = test.copy()
    test_df["predicted"] = y_pred
    top3 = test_df.groupby("state")["weekly_new_cases"].sum().nlargest(3).index
    ax = axes[1]
    for state in top3:
        s = test_df[test_df["state"] == state].sort_values("week_start")
        ax.plot(s["week_start"], s["weekly_new_cases"], label=f"{state} (actual)", linewidth=1.5)
        ax.plot(s["week_start"], s["predicted"], "--", label=f"{state} (pred)", linewidth=1.5)
    ax.set_title("Top 3 States — Forecast vs Actual")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlabel("")
    plt.xticks(rotation=45)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_FIG, "xgboost_predictions.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def main():
    logger.info("=" * 60)
    logger.info("STEP 5: GRADIENT BOOSTING MODEL")
    logger.info("=" * 60)

    train, test, df = load_and_split()
    model, y_test, y_pred, feature_cols, scaler = train_model(train, test)
    plot_feature_importance(model, feature_cols)
    plot_predictions(test, y_test, y_pred)
    joblib.dump(model, os.path.join(OUTPUTS_MODELS, "gradient_boosting_model.pkl"))
    logger.info("Gradient Boosting training complete.")


if __name__ == "__main__":
    main()
