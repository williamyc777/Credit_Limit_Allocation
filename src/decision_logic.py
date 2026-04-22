"""
Decision logic for credit actions — NOT driven by a fixed 0.5 PD cutoff.

Course guidance: there is little reason to treat 0.5 as the right threshold;
prefer an explicit decision rule tied to business objectives.

This script:
  1) Rebuilds the same train/test split as train_model.py (random_state=42).
  2) Loads the saved best model and scores the TEST split only.
  3) Scans candidate thresholds on P(default):
        - "Approve" / extend exposure if PD <= t  (low predicted risk)
        - "Decline" / restrict if PD > t
  4) On the test set, evaluates a SIMPLE retrospective profit proxy for each t:
        If approve:  (1 - y) * revenue_per_good - y * loss_if_default
        If decline:  0  (neutral; you can add a fixed cost of decline if desired)
  5) Reports the threshold that maximizes total proxy profit vs. t = 0.5.

Adjust REVENUE_IF_GOOD and LOSS_IF_DEFAULT to match your story (or later replace
with full simulation using loan_amnt and interest).
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CLEAN_DATA_PATH = os.path.join(PROJECT_ROOT, "output", "clean_data.csv")


def load_clean_data():
    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(f"Missing {CLEAN_DATA_PATH}; run: python src/preprocess.py")
    return pd.read_csv(CLEAN_DATA_PATH, low_memory=False)


def load_best_bundle():
    with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "best_model_name.txt"), "r", encoding="utf-8") as f:
        name = f.read().strip()
    return model, scaler, name


def predict_proba_test(X_test, model, scaler, best_name):
    if best_name == "logistic_regression":
        return model.predict_proba(scaler.transform(X_test))[:, 1]
    return model.predict_proba(X_test)[:, 1]


def total_profit_proxy(
    y_true: np.ndarray,
    pd_scores: np.ndarray,
    threshold: float,
    revenue_if_good: float,
    loss_if_default: float,
) -> float:
    """
    Approve loans with PD <= threshold. Retrospective realized cashflow on test labels.
    """
    approve = pd_scores <= threshold
    profit = 0.0
    for i in range(len(y_true)):
        if not approve[i]:
            continue
        if y_true[i] == 0:
            profit += revenue_if_good
        else:
            profit -= loss_if_default
    return profit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--revenue",
        type=float,
        default=1.0,
        help="Retrospective revenue proxy if loan is good (y=0) and approved",
    )
    parser.add_argument(
        "--loss",
        type=float,
        default=5.0,
        help="Retrospective loss if loan defaults (y=1) and approved",
    )
    args = parser.parse_args()

    df = load_clean_data()
    X = df.drop(columns=["default"])
    y = df["default"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model, scaler, best_name = load_best_bundle()
    p_test = predict_proba_test(X_test, model, scaler, best_name)

    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        prof = total_profit_proxy(
            y_test, p_test, t, args.revenue, args.loss
        )
        rows.append({"threshold": t, "profit_proxy": prof})

    res = pd.DataFrame(rows)
    best_idx = res["profit_proxy"].idxmax()
    best_t = res.loc[best_idx, "threshold"]
    best_profit = res.loc[best_idx, "profit_proxy"]

    t05 = 0.5
    profit_05 = total_profit_proxy(
        y_test, p_test, t05, args.revenue, args.loss
    )

    print("=" * 70)
    print("DECISION LOGIC (test-set retrospective; NOT for deployment without validation)")
    print("=" * 70)
    print(f"Best model: {best_name}")
    print(
        f"Rule: APPROVE if P(default) <= t; revenue_if_good={args.revenue}, "
        f"loss_if_default={args.loss}"
    )
    print()
    print(
        f"  Profit proxy at t = 0.5 (arbitrary; sklearn default for .predict): {profit_05:,.2f}"
    )
    print(
        f"  Best t* over grid [0.01, 0.99]:  t* = {best_t:.3f}  |  profit proxy = {best_profit:,.2f}"
    )
    print()
    print(
        "Takeaway: 0.5 is not special; an explicit objective (here: simple profit proxy) "
        "picks a different operating point. For the project, the PRIMARY use of PD is "
        "continuous scoring + counterfactual limits + expected profit simulation — not "
        "a single hard threshold."
    )
    print("=" * 70)

    out_path = os.path.join(OUTPUT_DIR, "threshold_profit_scan.csv")
    res.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
