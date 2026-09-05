import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def detect_anomalies_zscore(df, window=24, threshold=3.0):
    """
    Detect anomalous spikes in a time-series of transaction counts using a rolling z-score.
    """
    df_out = df.copy()

    # Calculate rolling mean and standard deviation
    rolling = df_out["transaction_count"].rolling(window=window, min_periods=window)
    df_out["rolling_mean"] = rolling.mean()
    df_out["rolling_std"] = rolling.std()

    # Calculate z-score
    # Avoid division by zero warnings by temporarily replacing 0 with NaN or using np.where
    df_out["z_score"] = (df_out["transaction_count"] - df_out["rolling_mean"]) / df_out[
        "rolling_std"
    ]

    # If rolling_std is 0, set z_score to 0 instead of inf/NaN
    df_out.loc[df_out["rolling_std"] == 0, "z_score"] = 0.0

    # Flag anomalies
    df_out["predicted_anomaly"] = df_out["z_score"].abs() > threshold

    # Set predicted_anomaly to False for the first `window` rows where we lack enough data
    df_out.loc[df_out["rolling_mean"].isna(), "predicted_anomaly"] = False

    return df_out


def evaluate_predictions(df):
    """
    Evaluate predicted anomalies against ground truth.
    """
    if "is_anomaly" not in df.columns or "predicted_anomaly" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'is_anomaly' and 'predicted_anomaly' columns"
        )

    y_true = df["is_anomaly"].astype(bool)
    y_pred = df["predicted_anomaly"].astype(bool)

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
    rolling = df_out["transaction_count"].rolling(window=24, min_periods=1)
    df_out["rolling_mean_24h"] = rolling.mean()
    df_out["rolling_std_24h"] = rolling.std()
    df_out["deviation"] = df_out["transaction_count"] - df_out["rolling_mean_24h"]

    # Fill NaNs resulting from rolling operations with 0
    df_out["rolling_mean_24h"] = df_out["rolling_mean_24h"].fillna(0)
    df_out["rolling_std_24h"] = df_out["rolling_std_24h"].fillna(0)
    df_out["deviation"] = df_out["deviation"].fillna(0)

    features = ["transaction_count", "rolling_mean_24h", "rolling_std_24h", "deviation"]
    X = df_out[features]

    # Initialize and fit the model
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)

    # Predict (-1 for anomaly, 1 for normal)
    preds = model.predict(X)
    df_out["predicted_anomaly"] = preds == -1

    # Add continuous score for threshold sweeping (lower = more anomalous)
    df_out["score"] = model.decision_function(X)

    return df_out


def run_cost_sweep(df_zscore, df_isoforest, fn_cost=500, fp_cost=25, thresholds=None):
    """
    Computes total business cost across a range of decision thresholds for z-score
    and IsolationForest detectors, and finds the threshold that minimizes cost.
    """
    if thresholds is None:
        z_min, z_max = df_zscore["z_score"].min(), df_zscore["z_score"].max()
        iso_min, iso_max = df_isoforest["score"].min(), df_isoforest["score"].max()

        z_thresholds = np.linspace(z_min, z_max, 50)
        iso_thresholds = np.linspace(iso_min, iso_max, 50)
    else:
        z_thresholds = thresholds
        iso_thresholds = thresholds

    results = []

    # Evaluate z-score
    for t in z_thresholds:
        pred = df_zscore["z_score"] > t
        y_true = df_zscore["is_anomaly"]

        fn = (y_true & ~pred).sum()
        fp = (~y_true & pred).sum()

        cost = fn * fn_cost + fp * fp_cost

        results.append(
            {
                "detector": "Z-Score",
                "threshold": t,
                "FP": fp,
                "FN": fn,
                "total_cost": cost,
            }
        )

    # Evaluate IsolationForest
    for t in iso_thresholds:
        pred = df_isoforest["score"] < t
        y_true = df_isoforest["is_anomaly"]

        fn = (y_true & ~pred).sum()
        fp = (~y_true & pred).sum()

        cost = fn * fn_cost + fp * fp_cost

        results.append(
            {
                "detector": "IsolationForest",
                "threshold": t,
                "FP": fp,
                "FN": fn,
                "total_cost": cost,
            }
        )

    df_results = pd.DataFrame(results)

    # Find optimums
    z_res = df_results[df_results["detector"] == "Z-Score"]
    iso_res = df_results[df_results["detector"] == "IsolationForest"]

    z_best = z_res.loc[z_res["total_cost"].idxmin()]
    iso_best = iso_res.loc[iso_res["total_cost"].idxmin()]

    # Print summary
    print("\n--- Cost Sweep Summary ---")
    print(
        f"Z-Score Best: Threshold = {z_best['threshold']:.4f}, Cost = {z_best['total_cost']}"
    )
    print(
        f"IsolationForest Best: Threshold = {iso_best['threshold']:.4f}, Cost = {iso_best['total_cost']}"
    )

    if z_best["total_cost"] < iso_best["total_cost"]:
        winner = "Z-Score"
    elif iso_best["total_cost"] < z_best["total_cost"]:
        winner = "IsolationForest"
    else:
        winner = "Tie"
    print(f"Overall Winner: {winner}")

    # Plot
    plt.figure(figsize=(10, 6))

    plt.plot(z_res["threshold"], z_res["total_cost"], label="Z-Score", color="blue")
    plt.scatter(
        [z_best["threshold"]],
        [z_best["total_cost"]],
        color="blue",
        marker="*",
        s=200,
        zorder=5,
    )

    plt.plot(
        iso_res["threshold"],
        iso_res["total_cost"],
        label="IsolationForest",
        color="orange",
    )
    plt.scatter(
        [iso_best["threshold"]],
        [iso_best["total_cost"]],
        color="orange",
        marker="*",
        s=200,
        zorder=5,
    )

    plt.xlabel("Threshold")
    plt.ylabel("Total Cost")
    plt.title("Cost Sweep by Detector")
    plt.legend()
    plt.grid(True)
    plt.savefig('cost_sweep.png')

    return df_results


def detect_anomalies_ensemble(df):
    """
    Detect anomalies using an ensemble of Z-Score and IsolationForest.
    A data point is flagged as an anomaly if EITHER detector flags it
    using the fixed thresholds.
    """
    z_threshold = 1.8177
    iso_threshold = 0.0334

    df_z = detect_anomalies_zscore(df)
    df_i = detect_anomalies_isolationforest(df)

    df_out = df.copy()
    z_pred = df_z["z_score"].abs() > z_threshold
    iso_pred = df_i["score"] < iso_threshold

    df_out["predicted_anomaly"] = z_pred | iso_pred

    return df_out
