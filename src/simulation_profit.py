"""
Counterfactual credit limit (loan_amnt) simulation + simple expected profit optimization.

Uses the same trained best model to re-score PD when loan_amnt is scaled.
For each loan, compares a grid of multipliers on the original amount and picks
the multiplier that maximizes a one-period expected profit proxy:

    E[profit] ≈ A * ( (1 - PD) * margin  -  PD * lgd )

where A = counterfactual loan amount (proxy for limit/exposure), PD = model
P(default) after changing A, margin = simplified revenue rate on good loans,
lgd = loss rate on default (e.g. fraction of balance lost).

This connects instructor guidance: change loan_amnt → recompute PD → evaluate profit,
not a fixed 0.5 threshold.

Outputs:
  - output/simulation_portfolio_summary.csv  (1 row: aggregates)
  - output/simulation_per_loan_sample.csv    (first N rows detail; N configurable)
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CLEAN_PATH = os.path.join(PROJECT_ROOT, "output", "clean_data.csv")


def load_bundle():
    with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "best_model_name.txt"), "r", encoding="utf-8") as f:
        name = f.read().strip()
    return model, scaler, feature_cols, name


def predict_pd(X: pd.DataFrame, model, scaler, name: str) -> np.ndarray:
    if name == "logistic_regression":
        return model.predict_proba(scaler.transform(X))[:, 1]
    return model.predict_proba(X)[:, 1]


def run_simulation(
    df: pd.DataFrame,
    model,
    scaler,
    feature_cols: list,
    name: str,
    multipliers: list,
    margin: float,
    lgd: float,
):
    X = df[feature_cols]
    orig_amt = X["loan_amnt"].values.astype(float)
    n = len(df)
    k = len(multipliers)

    # PD matrix: n x k, profit matrix: n x k
    pd_mat = np.zeros((n, k))
    amt_mat = np.zeros((n, k))

    for j, m in enumerate(multipliers):
        Xm = X.copy()
        new_amt = orig_amt * m
        Xm["loan_amnt"] = new_amt
        pd_col = predict_pd(Xm, model, scaler, name)
        pd_mat[:, j] = pd_col
        amt_mat[:, j] = new_amt

    # E[profit] per row, per multiplier (same formula as in proposal narrative)
    profit_mat = amt_mat * ((1.0 - pd_mat) * margin - pd_mat * lgd)

    best_j = np.argmax(profit_mat, axis=1)
    best_profit = profit_mat[np.arange(n), best_j]
    baseline_j = multipliers.index(1.0) if 1.0 in multipliers else 0
    baseline_profit = profit_mat[:, baseline_j]
    best_pd = pd_mat[np.arange(n), best_j]
    best_amt = amt_mat[np.arange(n), best_j]
    best_mult = np.array(multipliers)[best_j]

    return {
        "orig_amt": orig_amt,
        "default": df["default"].values,
        "pd_mat": pd_mat,
        "profit_mat": profit_mat,
        "best_mult": best_mult,
        "best_amt": best_amt,
        "best_pd": best_pd,
        "best_profit": best_profit,
        "baseline_profit": baseline_profit,
        "pd_baseline": pd_mat[:, baseline_j],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--multipliers",
        type=str,
        default="0.7,0.85,1.0,1.1,1.2,1.3",
        help="Comma-separated multipliers on original loan_amnt",
    )
    p.add_argument("--margin", type=float, default=0.10, help="Simplified good-case margin on balance")
    p.add_argument("--lgd", type=float, default=0.50, help="Loss given default as fraction of balance")
    p.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="If >0, only use first N rows of clean_data (faster); 0 = all rows",
    )
    p.add_argument(
        "--sample-out",
        type=int,
        default=5000,
        help="Rows to write in simulation_per_loan_sample.csv (capped for repo size)",
    )
    args = p.parse_args()

    multipliers = [float(x) for x in args.multipliers.split(",") if x.strip()]
    if 1.0 not in multipliers:
        multipliers.append(1.0)
        multipliers = sorted(set(multipliers))

    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(f"Missing {CLEAN_PATH}; run preprocess first.")

    df = pd.read_csv(CLEAN_PATH, low_memory=False)
    if args.sample_rows > 0:
        df = df.iloc[: args.sample_rows].copy()

    model, scaler, feature_cols, name = load_bundle()

    print("=" * 70)
    print("CREDIT LIMIT COUNTERFACTUALS + EXPECTED PROFIT (SIMPLE PROXY)")
    print("=" * 70)
    print(f"Best model: {name}")
    print(f"Rows: {len(df):,} | multipliers: {multipliers}")
    print(f"margin={args.margin}, lgd={args.lgd}")
    print("Running re-scoring (this may take a few minutes on full data)...")

    res = run_simulation(df, model, scaler, feature_cols, name, multipliers, args.margin, args.lgd)

    total_baseline = res["baseline_profit"].sum()
    total_opt = res["best_profit"].sum()
    mean_baseline = res["baseline_profit"].mean()
    mean_opt = res["best_profit"].mean()

    n_changed = (res["best_mult"] != 1.0).sum()

    summary = pd.DataFrame(
        [
            {
                "n_loans": len(df),
                "sum_expected_profit_baseline_mult1": total_baseline,
                "sum_expected_profit_optimized": total_opt,
                "delta_total": total_opt - total_baseline,
                "mean_profit_baseline": mean_baseline,
                "mean_profit_optimized": mean_opt,
                "n_loans_best_neq_1.0x": n_changed,
                "margin": args.margin,
                "lgd": args.lgd,
                "multipliers": str(multipliers),
            }
        ]
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_sum = os.path.join(OUTPUT_DIR, "simulation_portfolio_summary.csv")
    summary.to_csv(out_sum, index=False)
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Saved: {out_sum}")

    n_out = min(args.sample_out, len(df))
    sample = pd.DataFrame(
        {
            "loan_amnt_orig": res["orig_amt"][:n_out],
            "default": res["default"][:n_out],
            "PD_at_baseline_1.0x": res["pd_baseline"][:n_out],
            "optimal_multiplier": res["best_mult"][:n_out],
            "loan_amnt_opt": res["best_amt"][:n_out],
            "PD_at_opt": res["best_pd"][:n_out],
            "E_profit_baseline_1.0x": res["baseline_profit"][:n_out],
            "E_profit_opt": res["best_profit"][:n_out],
        }
    )
    out_samp = os.path.join(OUTPUT_DIR, "simulation_per_loan_sample.csv")
    sample.to_csv(out_samp, index=False)
    print(f"Saved: {out_samp}  ({n_out} rows)")

    print()
    print("Note: margin/lgd are stylized; tune with your business case. Primary point:")
    print("  PD is recomputed for each counterfactual loan_amnt, then expected profit")
    print("  drives the preferred limit on this grid — not a 0.5 label threshold.")
    print("=" * 70)


if __name__ == "__main__":
    main()
