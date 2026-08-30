import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_transactions(num_days=90, seed=42, num_spikes=5, spike_strength=5.0):
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
        
    df = pd.DataFrame({
        'timestamp': timestamps,
        'transaction_count': transaction_count,
        'is_anomaly': is_anomaly
    })
    return df

