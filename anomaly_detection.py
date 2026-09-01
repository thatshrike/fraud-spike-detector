import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import IsolationForest

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

def detect_anomalies_isolationforest(df, contamination=0.02, random_state=42):
    """
    Detect anomalous spikes in transaction data using sklearn's IsolationForest.
    """
    df_out = df.copy()
    
    # Feature engineering
    rolling = df_out['transaction_count'].rolling(window=24, min_periods=1)
    df_out['rolling_mean_24h'] = rolling.mean()
    df_out['rolling_std_24h'] = rolling.std()
    df_out['deviation'] = df_out['transaction_count'] - df_out['rolling_mean_24h']
    
    # Fill NaNs resulting from rolling operations with 0
    df_out['rolling_mean_24h'] = df_out['rolling_mean_24h'].fillna(0)
    df_out['rolling_std_24h'] = df_out['rolling_std_24h'].fillna(0)
    df_out['deviation'] = df_out['deviation'].fillna(0)
    
    features = ['transaction_count', 'rolling_mean_24h', 'rolling_std_24h', 'deviation']
    X = df_out[features]
    
    # Initialize and fit the model
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)
    
    # Predict (-1 for anomaly, 1 for normal)
    preds = model.predict(X)
    df_out['predicted_anomaly'] = (preds == -1)
    
    # Add continuous score for threshold sweeping (lower = more anomalous)
    df_out['score'] = model.decision_function(X)
    
    return df_out

