"""
Multi-Model PD Prediction
Trains and compares:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. HistGradientBoosting (sklearn — strong tabular baseline, large-sample friendly)

Compares ROC-AUC, PR-AUC; exports pd_predictions, models, and evaluation figures
(ROC, PR, calibration for best model).
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from evaluation_plots import save_all_evaluation_figures

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA_PATH = os.path.join(PROJECT_ROOT, "output", "clean_data.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def load_clean_data():
    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(
            f"Cannot find cleaned data: {CLEAN_DATA_PATH}\n"
            f"Please run: python src/preprocess.py first"
        )
    return pd.read_csv(CLEAN_DATA_PATH)


DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
}


def evaluate_model(model, X_test_input, y_test, model_key: str, test_scores: dict):
    y_pred = model.predict(X_test_input)
    y_pred_prob = model.predict_proba(X_test_input)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    pr_auc = average_precision_score(y_test, y_pred_prob)
    test_scores[model_key] = y_pred_prob

    print("=" * 60)
    print(f"Model: {DISPLAY.get(model_key, model_key)}")
    print("=" * 60)
    print(f"AUC: {auc:.4f}  |  PR-AUC (avg precision): {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(
        "Note: .predict() uses sklearn's default 0.5 on P(default) — not a business rule.\n"
        "      For decision logic / profit-style thresholds, run: python src/decision_logic.py\n"
    )

    return auc, pr_auc


def train_models():
    df = load_clean_data()

    X = df.drop(columns=["default"])
    y = df["default"]

    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    test_scores = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.transform(X)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=10,
            learning_rate=0.06,
            l2_regularization=0.5,
            min_samples_leaf=40,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
        ),
    }

    model_results = {}
    full_pd_output = pd.DataFrame(
        {
            "loan_amnt": df["loan_amnt"].values,
            "default": df["default"].values,
        }
    )

    # Logistic
    lr_model = models["logistic_regression"]
    lr_model.fit(X_train_scaled, y_train)
    lr_auc, lr_pr = evaluate_model(
        lr_model, X_test_scaled, y_test, "logistic_regression", test_scores
    )
    full_pd_output["PD_logistic_regression"] = lr_model.predict_proba(X_full_scaled)[:, 1]
    model_results["logistic_regression"] = {
        "model": lr_model,
        "auc": lr_auc,
        "pr_auc": lr_pr,
        "needs_scaling": True,
    }

    # Decision Tree
    dt_model = models["decision_tree"]
    dt_model.fit(X_train, y_train)
    dt_auc, dt_pr = evaluate_model(
        dt_model, X_test, y_test, "decision_tree", test_scores
    )
    full_pd_output["PD_decision_tree"] = dt_model.predict_proba(X)[:, 1]
    model_results["decision_tree"] = {
        "model": dt_model,
        "auc": dt_auc,
        "pr_auc": dt_pr,
        "needs_scaling": False,
    }

    # Random Forest
    rf_model = models["random_forest"]
    rf_model.fit(X_train, y_train)
    rf_auc, rf_pr = evaluate_model(
        rf_model, X_test, y_test, "random_forest", test_scores
    )
    full_pd_output["PD_random_forest"] = rf_model.predict_proba(X)[:, 1]
    model_results["random_forest"] = {
        "model": rf_model,
        "auc": rf_auc,
        "pr_auc": rf_pr,
        "needs_scaling": False,
    }

    # HistGradientBoosting
    hgb = models["hist_gradient_boosting"]
    hgb.fit(X_train, y_train)
    h_auc, h_pr = evaluate_model(
        hgb, X_test, y_test, "hist_gradient_boosting", test_scores
    )
    full_pd_output["PD_hist_gradient_boosting"] = hgb.predict_proba(X)[:, 1]
    model_results["hist_gradient_boosting"] = {
        "model": hgb,
        "auc": h_auc,
        "pr_auc": h_pr,
        "needs_scaling": False,
    }

    best_model_name = max(model_results, key=lambda name: model_results[name]["auc"])
    best_model = model_results[best_model_name]["model"]

    if model_results[best_model_name]["needs_scaling"]:
        full_pd_output["PD_best_model"] = best_model.predict_proba(X_full_scaled)[:, 1]
    else:
        full_pd_output["PD_best_model"] = best_model.predict_proba(X)[:, 1]

    full_pd_output["PD"] = full_pd_output["PD_best_model"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_pd_output.to_csv(os.path.join(OUTPUT_DIR, "pd_predictions.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "all_models.pkl"), "wb") as f:
        pickle.dump(
            {name: result["model"] for name, result in model_results.items()},
            f,
        )

    with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(OUTPUT_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    with open(os.path.join(OUTPUT_DIR, "best_model_name.txt"), "w", encoding="utf-8") as f:
        f.write(best_model_name)

    metrics_df = pd.DataFrame(
        {
            "model": list(model_results.keys()),
            "auc": [model_results[name]["auc"] for name in model_results],
            "pr_auc": [model_results[name]["pr_auc"] for name in model_results],
        }
    ).sort_values(by="auc", ascending=False)

    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    # ROC / PR / calibration figures
    save_all_evaluation_figures(
        y_test.values,
        test_scores,
        best_model_name,
        OUTPUT_DIR,
        display_names=DISPLAY,
    )

    print("=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print()
    print(f"Best model: {best_model_name}")
    print(f"Saved: {OUTPUT_DIR}/pd_predictions.csv")
    print(f"Saved: {OUTPUT_DIR}/model_comparison.csv")
    print(f"Saved: {OUTPUT_DIR}/all_models.pkl")
    print(f"Saved: {OUTPUT_DIR}/best_model.pkl")
    print(f"Saved: {OUTPUT_DIR}/scaler.pkl")
    print(f"Saved: {OUTPUT_DIR}/feature_cols.pkl")
    print(f"Saved: {OUTPUT_DIR}/best_model_name.txt")
    print(f"Saved: {OUTPUT_DIR}/roc_curves_all_models.png, pr_curves_all_models.png, calibration_best_model.png")

    return model_results, best_model_name


if __name__ == "__main__":
    train_models()
