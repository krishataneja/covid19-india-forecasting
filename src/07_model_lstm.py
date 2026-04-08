"""
07_model_lstm.py — Neural network model for COVID-19 case forecasting.

Uses TensorFlow LSTM if available; otherwise falls back to sklearn
MLPRegressor with sliding-window features (equivalent architecture).

Architecture:
    - Input: flattened sliding window of 8 weeks x N features
    - Hidden layers: 128 → 64 → 32 neurons with ReLU
    - Output: predicted weekly cases
"""

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import joblib

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_PROCESSED, OUTPUTS_FIG, OUTPUTS_MODELS, get_logger, compute_metrics, save_metrics

logger = get_logger("07_lstm")

WINDOW_SIZE = 8


def load_data(test_weeks=8):
    df = pd.read_csv(os.path.join(DATA_PROCESSED, "features_ml.csv"),
                     parse_dates=["week_start"])
    df.sort_values(["state", "week_start"], inplace=True)
    all_weeks = sorted(df["week_start"].unique())
    cutoff = all_weeks[-test_weeks]
    return df, cutoff


def get_feature_cols(df):
    exclude = {"state", "year_week", "week_start", "weekly_new_cases"}
    return [c for c in df.columns if c not in exclude]


def create_sequences(data, target_idx, window=WINDOW_SIZE):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)


def try_tensorflow(df, cutoff):
    """Try TensorFlow LSTM."""
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        logger.info("TensorFlow not installed. Using MLP fallback.")
        return None, None, None

    feature_cols = get_feature_cols(df)
    target = "weekly_new_cases"
    all_cols = feature_cols + [target]
    target_idx = len(feature_cols)

    scaler = MinMaxScaler()
    train_mask = df["week_start"] < cutoff
    scaler.fit(df.loc[train_mask, all_cols].values)

    X_train_all, y_train_all, X_test_all, y_test_all = [], [], [], []

    for state in df["state"].unique():
        sdf = df[df["state"] == state].sort_values("week_start")
        if len(sdf) < WINDOW_SIZE + 2:
            continue
        scaled = scaler.transform(sdf[all_cols].values)
        X_seq, y_seq = create_sequences(scaled, target_idx)
        dates = sdf["week_start"].values[WINDOW_SIZE:]
        tr = dates < np.datetime64(cutoff)
        te = ~tr
        if tr.sum() > 0:
            X_train_all.append(X_seq[tr]); y_train_all.append(y_seq[tr])
        if te.sum() > 0:
            X_test_all.append(X_seq[te]); y_test_all.append(y_seq[te])

    X_train = np.concatenate(X_train_all); y_train = np.concatenate(y_train_all)
    X_test = np.concatenate(X_test_all); y_test = np.concatenate(y_test_all)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, len(all_cols))),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_split=0.15,
                        epochs=80, batch_size=64, callbacks=[es], verbose=0)

    y_pred_sc = model.predict(X_test, verbose=0).flatten()
    dummy = np.zeros((len(y_pred_sc), len(all_cols)))
    dummy[:, target_idx] = y_pred_sc
    y_pred = scaler.inverse_transform(dummy)[:, target_idx].clip(min=0)
    dummy[:, target_idx] = y_test
    y_true = scaler.inverse_transform(dummy)[:, target_idx]

    return y_true, y_pred, history


def mlp_sliding_window(df, cutoff):
    """Fallback: MLP with flattened sliding-window features."""
    feature_cols = get_feature_cols(df)
    target = "weekly_new_cases"
    all_cols = feature_cols + [target]
    target_idx = len(feature_cols)

    scaler = MinMaxScaler()
    train_mask = df["week_start"] < cutoff
    scaler.fit(df.loc[train_mask, all_cols].values)

    X_train_all, y_train_all, X_test_all, y_test_all = [], [], [], []

    for state in df["state"].unique():
        sdf = df[df["state"] == state].sort_values("week_start")
        if len(sdf) < WINDOW_SIZE + 2:
            continue
        scaled = scaler.transform(sdf[all_cols].values)
        X_seq, y_seq = create_sequences(scaled, target_idx)

        # Flatten windows for MLP: (n, window, features) → (n, window*features)
        X_flat = X_seq.reshape(X_seq.shape[0], -1)

        dates = sdf["week_start"].values[WINDOW_SIZE:]
        tr = dates < np.datetime64(cutoff)
        te = ~tr
        if tr.sum() > 0:
            X_train_all.append(X_flat[tr]); y_train_all.append(y_seq[tr])
        if te.sum() > 0:
            X_test_all.append(X_flat[te]); y_test_all.append(y_seq[te])

    X_train = np.concatenate(X_train_all); y_train = np.concatenate(y_train_all)
    X_test = np.concatenate(X_test_all); y_test = np.concatenate(y_test_all)

    logger.info(f"MLP sequences — Train: {X_train.shape}, Test: {X_test.shape}")

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_train, y_train)
    logger.info(f"MLP training: {mlp.n_iter_} iterations")

    y_pred_sc = mlp.predict(X_test)

    # Inverse transform
    dummy = np.zeros((len(y_pred_sc), len(all_cols)))
    dummy[:, target_idx] = y_pred_sc
    y_pred = scaler.inverse_transform(dummy)[:, target_idx].clip(min=0)
    dummy[:, target_idx] = y_test
    y_true = scaler.inverse_transform(dummy)[:, target_idx]

    # Save loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mlp.loss_curve_, label="Train Loss")
    if hasattr(mlp, "validation_scores_"):
        ax.plot([-s for s in mlp.validation_scores_], label="Val Loss (neg R2)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("MLP Training Curve", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_FIG, "lstm_training_curve.png"), bbox_inches="tight")
    plt.close()

    joblib.dump(mlp, os.path.join(OUTPUTS_MODELS, "mlp_model.pkl"))

    return y_true, y_pred


def plot_results(y_true, y_pred, label="Neural Network"):
    if y_true is None:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#8e44ad")
    lims = [0, max(y_true.max(), y_pred.max()) * 1.1]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Weekly Cases")
    ax.set_ylabel("Predicted Weekly Cases")
    ax.set_title(f"{label}: Actual vs Predicted", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_FIG, "lstm_predictions.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def main():
    logger.info("=" * 60)
    logger.info("STEP 7: NEURAL NETWORK MODEL")
    logger.info("=" * 60)

    df, cutoff = load_data()

    # Try TensorFlow LSTM first
    y_true, y_pred, history = try_tensorflow(df, cutoff)

    if y_true is None:
        # Fallback to MLP
        y_true, y_pred = mlp_sliding_window(df, cutoff)
        label = "MLP (sliding window)"
    else:
        label = "LSTM"

    if y_true is not None:
        metrics = compute_metrics(y_true, y_pred)
        logger.info(f"{label} test metrics: {metrics}")
        save_metrics(metrics, "neural_network")
        plot_results(y_true, y_pred, label)

    logger.info("Neural network training complete.")


if __name__ == "__main__":
    main()
