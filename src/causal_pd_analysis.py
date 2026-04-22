"""
Causal-style what-if analysis:
Change loan_amnt and observe PD change using the best saved model.
"""

import os
import pickle
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CLEAN_DATA_PATH = os.path.join(PROJECT_ROOT, "output", "clean_data.csv")


def load_best_model_bundle():
    with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(OUTPUT_DIR, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)

    with open(os.path.join(OUTPUT_DIR, "best_model_name.txt"), "r", encoding="utf-8") as f:
        best_model_name = f.read().strip()

    return model, scaler, feature_cols, best_model_name


def estimate_pd_change_with_loan_amnt(loan_amnt_new, row, feature_cols, model, scaler, best_model_name):
    X = row[feature_cols].copy()
    X["loan_amnt"] = loan_amnt_new
    X_df = X.to_frame().T
    X_df = X_df[feature_cols]

    if best_model_name == "logistic_regression":
        X_input = scaler.transform(X_df)
    else:
        X_input = X_df

    pd_new = model.predict_proba(X_input)[:, 1][0]
    return pd_new


def demo_causal_effect():
    df = pd.read_csv(CLEAN_DATA_PATH)
    model, scaler, feature_cols, best_model_name = load_best_model_bundle()

    sample = df.head(5)

    print("What-if analysis: changing loan_amnt and observing PD")
    print("=" * 60)
    print(f"Using best model: {best_model_name}")
    print("=" * 60)

    for idx, row in sample.iterrows():
        loan_amnt_orig = row["loan_amnt"]
        print(f"\nSample row {idx} | original loan_amnt = ${loan_amnt_orig:,.0f}")

        for mult in [0.8, 1.0, 1.2]:
            amnt = loan_amnt_orig * mult
            pd_new = estimate_pd_change_with_loan_amnt(
                amnt, row, feature_cols, model, scaler, best_model_name
            )
            print(f"loan_amnt: ${amnt:,.0f} -> PD: {pd_new:.4f}")


if __name__ == "__main__":
    demo_causal_effect()
