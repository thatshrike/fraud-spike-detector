from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anomaly_detection import (
    detect_anomalies_isolationforest,
    detect_anomalies_zscore,
    evaluate_predictions,
)


def generate_ramping_fraud(
    num_days=90, seed=42, ramp_duration_hours=48, max_multiplier=5.0
):
    rng = np.random.default_rng(seed)
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(num_days * 24)]
    hours = np.arange(len(timestamps))

    # Baseline generation (same as generate_synthetic_transactions)
    baseline = 100 + 50 * np.sin((hours - 6) * (2 * np.pi / 24))
    noise = rng.normal(loc=0, scale=10, size=len(timestamps))
    transaction_count = np.maximum(0, baseline + noise).astype(float)
    is_anomaly = np.zeros(len(timestamps), dtype=bool)

    # Inject one gradual ramp
    # Choose a random start point for the ramp, leaving enough room for ramp_duration_hours
    start_idx = rng.integers(0, len(timestamps) - ramp_duration_hours)
    end_idx = start_idx + ramp_duration_hours

    # Ramp multiplier goes from 1.0 to max_multiplier linearly over ramp_duration_hours
    ramp_multipliers = np.linspace(1.0, max_multiplier, ramp_duration_hours)

    # Apply ramp
    transaction_count[start_idx:end_idx] *= ramp_multipliers

    # Convert to int
    transaction_count = transaction_count.astype(int)

    # Mark anomalies
    is_anomaly[start_idx:end_idx] = True

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "transaction_count": transaction_count,
            "is_anomaly": is_anomaly,
        }
    )
    return df


if __name__ == "__main__":
    print("Generating ramping fraud dataset...")
    df = generate_ramping_fraud(
        num_days=90, seed=42, ramp_duration_hours=48, max_multiplier=5.0
    )

    z_threshold = 1.8177
    iso_threshold = 0.0334

    print("\nRunning Z-Score detection...")
    df_zscore = detect_anomalies_zscore(df)
    df_zscore["predicted_anomaly"] = df_zscore["z_score"].abs() > z_threshold
    print("--- Z-Score Predictions on Ramping Fraud ---")
    evaluate_predictions(df_zscore)

    print("\nRunning IsolationForest detection...")
    df_iforest = detect_anomalies_isolationforest(df)
    df_iforest["predicted_anomaly"] = df_iforest["score"] < iso_threshold
    print("--- IsolationForest Predictions on Ramping Fraud ---")
    evaluate_predictions(df_iforest)

    print("\n--- Summary Comparison ---")
    print(
        "Recall on Sudden Spikes (Known): Z-Score ~0.77-0.94, IsolationForest ~0.95-1.00"
    )

    z_recall = (
        df_zscore["is_anomaly"] & df_zscore["predicted_anomaly"]
    ).sum() / df_zscore["is_anomaly"].sum()
    i_recall = (
        df_iforest["is_anomaly"] & df_iforest["predicted_anomaly"]
    ).sum() / df_iforest["is_anomaly"].sum()

    print(
        f"Recall on Gradual Ramp:        Z-Score {z_recall:.4f}, IsolationForest {i_recall:.4f}"
    )

    # Plotting
    plt.figure(figsize=(15, 6))

    # Just plot around the anomaly
    anomaly_indices = df.index[df["is_anomaly"]]
    start_plot = max(0, anomaly_indices[0] - 48)
    end_plot = min(len(df), anomaly_indices[-1] + 48)

    df_plot = df.iloc[start_plot:end_plot]
    df_z_plot = df_zscore.iloc[start_plot:end_plot]
    df_i_plot = df_iforest.iloc[start_plot:end_plot]

    plt.plot(
        df_plot["timestamp"],
        df_plot["transaction_count"],
        label="Transaction Count",
        color="blue",
        alpha=0.5,
    )

    plt.fill_between(
        df_plot["timestamp"],
        0,
        df_plot["transaction_count"],
        where=df_plot["is_anomaly"],
        color="red",
        alpha=0.2,
        label="Actual Ramp Window",
    )

    # Mark detections
    z_anomalies = df_z_plot[df_z_plot["predicted_anomaly"]]
    i_anomalies = df_i_plot[df_i_plot["predicted_anomaly"]]

    plt.scatter(
        z_anomalies["timestamp"],
        z_anomalies["transaction_count"],
        color="orange",
        marker="o",
        s=50,
        label="Z-Score Detected",
        zorder=5,
    )
    plt.scatter(
        i_anomalies["timestamp"],
        i_anomalies["transaction_count"],
        color="purple",
        marker="x",
        s=50,
        label="IsolationForest Detected",
        zorder=6,
    )

    plt.title("Detection of Gradual Ramping Fraud")
    plt.xlabel("Time")
    plt.ylabel("Transaction Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ramp_up_plot.png")
    print("\nSaved plot to ramp_up_plot.png")
