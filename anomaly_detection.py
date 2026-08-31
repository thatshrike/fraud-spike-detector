import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def detect_anomalies_zscore(df, window=24, threshold=3.0):
    """
    Detect anomalous spikes in a time-series of transaction counts using a rolling z-score.
    """
    df_out = df.copy()
    
    # Calculate rolling mean and standard deviation
    rolling = df_out['transaction_count'].rolling(window=window, min_periods=window)
    df_out['rolling_mean'] = rolling.mean()
    df_out['rolling_std'] = rolling.std()
    
    # Calculate z-score
    # Avoid division by zero warnings by temporarily replacing 0 with NaN or using np.where
    df_out['z_score'] = (df_out['transaction_count'] - df_out['rolling_mean']) / df_out['rolling_std']
    
    # If rolling_std is 0, set z_score to 0 instead of inf/NaN
    df_out.loc[df_out['rolling_std'] == 0, 'z_score'] = 0.0
    
    # Flag anomalies
    df_out['predicted_anomaly'] = df_out['z_score'].abs() > threshold
    
    # Set predicted_anomaly to False for the first `window` rows where we lack enough data
    df_out.loc[df_out['rolling_mean'].isna(), 'predicted_anomaly'] = False
    
    return df_out

def evaluate_predictions(df):
    """
    Evaluate predicted anomalies against ground truth.
    """
    if 'is_anomaly' not in df.columns or 'predicted_anomaly' not in df.columns:
        raise ValueError("DataFrame must contain 'is_anomaly' and 'predicted_anomaly' columns")
        
    y_true = df['is_anomaly'].astype(bool)
    y_pred = df['predicted_anomaly'].astype(bool)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()
    
    print("Evaluation Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Negatives (FN): {fn}")

