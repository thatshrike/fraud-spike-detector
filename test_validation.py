import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from anomaly_detection import (
    detect_anomalies_ensemble,
    detect_anomalies_isolationforest,
    detect_anomalies_zscore,
    evaluate_predictions,
)
from test_run import generate_synthetic_transactions


def get_metrics(df):
    """
    Calculate precision, recall, and F1 score for predictions in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'is_anomaly' and 'predicted_anomaly' columns.
        
    Returns:
        tuple: (precision, recall, f1_score)
    """
    y_true = df["is_anomaly"].astype(bool)
    y_pred = df["predicted_anomaly"].astype(bool)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    return p, r, f


def format_metric(vals):
    """
    Format a list of metric values into a string showing the average and min-max range.
    
    Args:
        vals (list): List of numerical metric values.
        
    Returns:
        str: Formatted string (e.g., '0.9500 (0.9000-1.0000)').
    """
    avg = np.mean(vals)
    min_val = np.min(vals)
    max_val = np.max(vals)
    return f"{avg:.4f} ({min_val:.4f}-{max_val:.4f})"


def main():
    seeds = [42, 99, 7, 123]
    z_threshold = 1.8177
    iso_threshold = 0.0334

    z_metrics = {"p": [], "r": [], "f": []}
    i_metrics = {"p": [], "r": [], "f": []}
    e_metrics = {"p": [], "r": [], "f": []}

    print(f"Applying fixed Z-Score threshold: {z_threshold}")
    print(f"Applying fixed IsolationForest threshold: {iso_threshold}\n")

    for seed in seeds:
        print("=" * 50)
        print(f"Evaluating Seed: {seed}")
        print("=" * 50)
        test_df = generate_synthetic_transactions(
            num_days=90, seed=seed, num_spikes=5, spike_strength=5.0
        )

        # Z-Score
        df_zscore_test = detect_anomalies_zscore(test_df)
        df_zscore_test["predicted_anomaly"] = (
            df_zscore_test["z_score"].abs() > z_threshold
        )
        print("\n--- Z-Score Predictions ---")
        evaluate_predictions(df_zscore_test)

        # IsolationForest
        df_iforest_test = detect_anomalies_isolationforest(test_df)
        df_iforest_test["predicted_anomaly"] = df_iforest_test["score"] < iso_threshold
        print("\n--- IsolationForest Predictions ---")
        evaluate_predictions(df_iforest_test)

        # Ensemble
        df_ensemble_test = detect_anomalies_ensemble(test_df)
        print("\n--- Ensemble Predictions ---")
        evaluate_predictions(df_ensemble_test)

        z_p, z_r, z_f = get_metrics(df_zscore_test)
        z_metrics["p"].append(z_p)
        z_metrics["r"].append(z_r)
        z_metrics["f"].append(z_f)

        i_p, i_r, i_f = get_metrics(df_iforest_test)
        i_metrics["p"].append(i_p)
        i_metrics["r"].append(i_r)
        i_metrics["f"].append(i_f)

        e_p, e_r, e_f = get_metrics(df_ensemble_test)
        e_metrics["p"].append(e_p)
        e_metrics["r"].append(e_r)
        e_metrics["f"].append(e_f)
        print("\n")

    print("=" * 80)
    print("--- Multi-Seed Side-by-Side Comparison (Avg and Min-Max Range) ---")
    print(
        f"{'Detector':<15} | {'Precision (Avg, Range)':<25} | {'Recall (Avg, Range)':<25} | {'F1 (Avg, Range)':<25}"
    )
    print("-" * 97)
    print(
        f"{'Z-Score':<15} | {format_metric(z_metrics['p']):<25} | {format_metric(z_metrics['r']):<25} | {format_metric(z_metrics['f']):<25}"
    )
    print(
        f"{'IsolationForest':<15} | {format_metric(i_metrics['p']):<25} | {format_metric(i_metrics['r']):<25} | {format_metric(i_metrics['f']):<25}"
    )
    print(
        f"{'Ensemble':<15} | {format_metric(e_metrics['p']):<25} | {format_metric(e_metrics['r']):<25} | {format_metric(e_metrics['f']):<25}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
