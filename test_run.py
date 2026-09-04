from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from anomaly_detection import (
    detect_anomalies_isolationforest,
    detect_anomalies_zscore,
    evaluate_predictions,
    run_cost_sweep,
)


def generate_synthetic_transactions(
    num_days=90, seed=42, num_spikes=5, spike_strength=5.0
):
    """
    Generate synthetic daily transaction data with organic seasonality and injected fraud spikes.
    
    Args:
        num_days (int): Number of days of data to generate.
        seed (int): Random seed for reproducibility.
        num_spikes (int): Number of sudden fraud spikes to inject.
        spike_strength (float): Multiplier for the transaction count during a spike.
        
    Returns:
        pd.DataFrame: DataFrame containing 'timestamp', 'transaction_count', and 'is_anomaly'.
    """
    rng = np.random.default_rng(seed)
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(num_days * 24)]
    hours = np.arange(len(timestamps))
    baseline = 100 + 50 * np.sin((hours - 6) * (2 * np.pi / 24))
    noise = rng.normal(loc=0, scale=10, size=len(timestamps))
    transaction_count = np.maximum(0, baseline + noise).astype(int)
    is_anomaly = np.zeros(len(timestamps), dtype=bool)

    placed_spikes = 0
    attempts = 0

    while placed_spikes < num_spikes and attempts < 1000:
        duration = rng.integers(2, 7)
        start_idx = rng.integers(0, len(timestamps) - duration + 1)
        end_idx = start_idx + duration

        if not np.any(is_anomaly[start_idx:end_idx]):
            is_anomaly[start_idx:end_idx] = True
            transaction_count[start_idx:end_idx] = (
                transaction_count[start_idx:end_idx] * spike_strength
            ).astype(int)
            placed_spikes += 1
        attempts += 1

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "transaction_count": transaction_count,
            "is_anomaly": is_anomaly,
        }
    )
    return df


if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_transactions(seed=123)

    print("Running anomaly detection...")
    # Add predicted_anomaly column using our zscore function
    df_predicted = detect_anomalies_zscore(df, window=24, threshold=3.0)

    print("Evaluating Z-Score predictions:")
    evaluate_predictions(df_predicted)

    print("\n-----------------------------------\n")

    print("Running IsolationForest anomaly detection...")
    df_iforest = detect_anomalies_isolationforest(df, contamination=0.02)
    print("Evaluating IsolationForest predictions:")
    evaluate_predictions(df_iforest)

    print("\n-----------------------------------\n")

    print("Running Cost Sweep Analysis...")
    df_sweep = run_cost_sweep(df_predicted, df_iforest, fn_cost=500, fp_cost=25)
