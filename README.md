# Fraud-Spike Detector

**Razorpay Buildathon — AI Risk Manager Track**

A comparison of two anomaly detection approaches (rolling z-score and IsolationForest) for catching fraud spikes in transaction data, with an explicit cost-based analysis of when each model is the right choice — not just which one scores higher on paper.

**Headline finding:** neither model is unconditionally better. Z-score wins when a missed fraud case costs less than roughly 15–25x a false alarm; IsolationForest wins above that ratio. The "right" model depends on a business assumption, not a leaderboard score — and we measured where that crossover actually sits, rather than asserting a winner.

---

## Problem

Merchants lose money to fraud spikes — short, unnatural surges in transaction activity that look different from organic growth. Catching them is a threshold problem, not a pure classification problem: catch too little and fraud slips through, catch too aggressively and legitimate transactions get flagged, creating customer friction. Most detector comparisons stop at precision/recall. This project goes one step further and asks: given that friction and fraud cost different amounts, which detector — and which threshold — actually minimizes business loss?

## Approach

1. **Synthetic data generation** — hourly transaction counts over 90 days, following a realistic daily seasonality pattern, with a known number of fraud spikes injected at random, non-overlapping windows. Because the data is synthetic, we have exact ground truth for every anomaly, which lets us measure precision/recall directly rather than guessing.
2. **Rolling z-score baseline** — flags any point more than *k* standard deviations from its trailing rolling mean. Simple, interpretable, deliberately naive.
3. **IsolationForest comparison** — an unsupervised ensemble model trained on engineered features (raw count, rolling mean, rolling std, deviation from rolling mean), flagging the most "isolated" points as anomalies.
4. **Cost-curve sweep** — instead of picking a single threshold and reporting one F1 score, we swept each detector across a full range of thresholds, computed total business cost (`FN × fraud_cost + FP × friction_cost`) at each point, and found where each detector's cost is minimized — for several different assumed cost ratios.
5. **Held-out validation** — the cost-optimal z-score threshold was applied, unchanged, to three independently generated datasets (different random seeds) never used in threshold selection, to check whether performance was a fluke of one dataset or a stable operating point.
6. **Failure-mode analysis** — an honest accounting of where and why each detector breaks, including a debugging incident encountered while building this (see Failure Recovery below).

## Results

### Detection performance (threshold = 1.8177 for z-score, throughout)

| Dataset | Precision | Recall | F1 |
|---|---|---|---|
| Tuning set (seed=42) | 0.68 | 0.77 | 0.72 |
| Held-out (seed=99) | 0.73 | 0.86 | 0.79 |
| Held-out (seeds 7, 123 — avg/range) | 0.52–0.73 | 0.85–0.94 | 0.64–0.79 |

Z-score's performance at its cost-optimal threshold was stable across four independently generated datasets — recall consistently in the 0.77–0.94 range, precision 0.52–0.73. This is a genuine stability check, not a train/test split in the strict ML sense: seed=42 was used both to select the threshold (via the cost sweep) and to re-score at that threshold; seeds 99, 7, and 123 are the true held-out evaluations.

IsolationForest (contamination=0.02, fixed) achieved **recall = 1.00 on every one of the four datasets**, but precision stayed low (0.27–0.32) throughout. This consistency is worth reading carefully rather than celebrating outright — see Failure Modes below.

### Cost-sensitivity sweep

| Assumed cost ratio (fraud : friction) | Optimal detector |
|---|---|
| 5:1 | Z-Score |
| 15:1 | Z-Score |
| 26:1 | IsolationForest |
| 32:1 | IsolationForest |
| 40:1 | IsolationForest |

The crossover point sits between roughly **15:1 and 25:1**. Below that ratio, z-score's precision (fewer false alarms) matters more than IsolationForest's recall; above it, catching more fraud outweighs the extra friction. In a real deployment, this ratio should come from actual cost data, not be assumed — using either model "by default" without first estimating this ratio risks optimizing for the wrong objective.

## Failure Modes and Known Vulnerabilities

1. **Rolling z-score under-detects short spikes at conservative thresholds.** At threshold=3.0 (a common textbook default), recall was only 0.30 — the rolling window used to compute "normal" behavior gets partially contaminated by the spike itself, especially for short-duration spikes. Lowering the threshold to the cost-optimal value (1.8177) fixed most of this, but it's a reminder that a detector's default settings can be badly miscalibrated for the actual cost structure of the problem.

2. **IsolationForest's flagged-anomaly rate is a hyperparameter artifact, not a data-driven signal.** With `contamination` fixed at 0.02, the model flags close to that fraction of any dataset it sees, roughly independent of the data's true anomaly rate. Its perfect recall (1.00) across all four datasets is consistent with this: a model that reliably flags ~2% of everything will tend to catch every true anomaly if the true anomaly rate is close to or below 2% — but this is closer to casting a wide net than precise detection, and the low precision (0.27–0.32) confirms it.

3. **No single model is unconditionally best.** See the cost-sweep results above — this is arguably the most important finding of the project, more actionable than either model's raw F1 score.

4. **Adversarial vulnerability — gradual ramping evades both detectors.** Both models are tuned to catch sudden, discrete spikes. A fraud pattern that ramps up gradually would likely evade both: z-score's rolling baseline would slowly absorb the drift, and IsolationForest's engineered features (which include rolling mean and deviation) would adapt alongside a slow-moving pattern rather than flagging it as anomalous. Neither model was tested against this attack pattern; a real deployment would need a complementary trend-detection method.

## Failure Recovery (a real one, not hypothetical)

While adding the cost-sweep function to `anomaly_detection.py` mid-project, a new import (`run_cost_sweep`) failed with no clear error — the function existed in the file, but Colab's Python session had cached the module from before the edit. Diagnosed by checking the file contents directly against the import error, then resolved via a session restart (equivalent to `importlib.reload()` for a live session). Left in the commit history as its own commit rather than hidden, since it's a real, common gotcha when iterating on a `.py` module from a notebook.

## Project Structure

```
fraud-spike-detector/
├── test_run.py            # synthetic data generation + plotting, and main orchestration
├── anomaly_detection.py   # both detectors, evaluation metrics, cost-sweep analysis
├── test_validation.py     # held-out validation across multiple random seeds
├── README.md
└── .gitignore
```

## Running it

```bash
git clone https://github.com/thatshrike/fraud-spike-detector.git
cd fraud-spike-detector
pip install pandas numpy matplotlib scikit-learn
python test_run.py          # generates data, runs both detectors, evaluates, runs cost sweep
python test_validation.py   # runs held-out validation across seeds 99, 7, 123
```

## Limitations

- All results are on synthetic data with a known, injected anomaly structure — real transaction fraud is likely messier and less cleanly separable.
- The cost sweep uses assumed cost values (`fraud_cost`, `friction_cost`) as inputs, not values derived from real business data — the crossover point (15–25x) is specific to the cost model used here and would shift with different assumptions.
- Held-out evaluation used a small number of true anomalies per dataset (~20), so precision/recall estimates carry meaningful sampling variance; the reported ranges reflect that, but a larger evaluation population would tighten them further.
- IsolationForest's `contamination` parameter was fixed rather than tuned or estimated from data — see Failure Modes above.