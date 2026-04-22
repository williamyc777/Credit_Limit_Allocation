"""
Multi-Model PD Prediction
Train 3 models:
1. Logistic Regression
2. Decision Tree
3. Random Forest

Compare AUC and export:
- pd_predictions.csv
- all_models.pkl
- scaler.pkl
- best_model.pkl
- best_model_name.txt
"""

import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA_PATH = os.path.join(PROJECT_ROOT, "output", "clean_data.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def load_clean_data():
    """Read cleaned data"""
    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(
            f"Cannot find cleaned data: {CLEAN_DATA_PATH}\n"
            f"Please run: python src/preprocess.py first"
        )
    return pd.read_csv(CLEAN_DATA_PATH)


def evaluate_model(model, X_test_input, y_test, model_name):
    """Print evaluation results and return AUC"""
    y_pred = model.predict(X_test_input)
    y_pred_prob = model.predict_proba(X_test_input)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)

    print("=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)
    print(f"AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    return auc, y_pred_prob


def train_models():
    """Train 3 models and save results"""
    df = load_clean_data()

    X = df.drop(columns=["default"])
    y = df["default"]

    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # scaler only for Logistic Regression
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
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
    }

    model_results = {}
    full_pd_output = pd.DataFrame({
        "loan_amnt": df["loan_amnt"].values,
        "default": df["default"].values
    })

    # Train Logistic Regression
    lr_model = models["logistic_regression"]
    lr_model.fit(X_train_scaled, y_train)
    lr_auc, _ = evaluate_model(
        lr_model,
        X_test_scaled,
        y_test,
        "Logistic Regression"
    )
    full_pd_output["PD_logistic_regression"] = lr_model.predict_proba(X_full_scaled)[:, 1]
    model_results["logistic_regression"] = {
        "model": lr_model,
        "auc": lr_auc,
        "needs_scaling": True
    }

    # Train Decision Tree
    dt_model = models["decision_tree"]
    dt_model.fit(X_train, y_train)
    dt_auc, _ = evaluate_model(
        dt_model,
        X_test,
        y_test,
        "Decision Tree"
    )
    full_pd_output["PD_decision_tree"] = dt_model.predict_proba(X)[:, 1]
    model_results["decision_tree"] = {
        "model": dt_model,
        "auc": dt_auc,
        "needs_scaling": False
    }

    # Train Random Forest
    rf_model = models["random_forest"]
    rf_model.fit(X_train, y_train)
    rf_auc, _ = evaluate_model(
        rf_model,
        X_test,
        y_test,
        "Random Forest"
    )
    full_pd_output["PD_random_forest"] = rf_model.predict_proba(X)[:, 1]
    model_results["random_forest"] = {
        "model": rf_model,
        "auc": rf_auc,
        "needs_scaling": False
    }

    # choose best model by AUC
    best_model_name = max(model_results, key=lambda name: model_results[name]["auc"])
    best_model = model_results[best_model_name]["model"]

    # also create one final PD column using best model
    if model_results[best_model_name]["needs_scaling"]:
        full_pd_output["PD_best_model"] = best_model.predict_proba(X_full_scaled)[:, 1]
    else:
        full_pd_output["PD_best_model"] = best_model.predict_proba(X)[:, 1]

    # Backward compatibility: single PD column for notebooks / simulators expecting "PD"
    full_pd_output["PD"] = full_pd_output["PD_best_model"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # save prediction csv
    full_pd_output.to_csv(os.path.join(OUTPUT_DIR, "pd_predictions.csv"), index=False)

    # save all models
    with open(os.path.join(OUTPUT_DIR, "all_models.pkl"), "wb") as f:
        pickle.dump(
            {name: result["model"] for name, result in model_results.items()},
            f
        )

    # save best model only
    with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    # save scaler
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # save feature columns
    with open(os.path.join(OUTPUT_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    # save best model name
    with open(os.path.join(OUTPUT_DIR, "best_model_name.txt"), "w", encoding="utf-8") as f:
        f.write(best_model_name)

    # save metrics summary
    metrics_df = pd.DataFrame({
        "model": list(model_results.keys()),
        "auc": [model_results[name]["auc"] for name in model_results]
    }).sort_values(by="auc", ascending=False)

    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

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

    return model_results, best_model_name


if __name__ == "__main__":
    train_models()
