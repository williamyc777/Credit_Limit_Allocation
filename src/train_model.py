"""
Baseline PD Model - Logistic Regression
DS&AI Project - William (Tech Lead + Modeling)

输出每个 loan 的 PD，供 simulation 使用。
模型与 scaler 会保存，便于后续做因果分析：改变 loan_amnt 估计信用额度调整对 PD 的影响。
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DATA_PATH = os.path.join(PROJECT_ROOT, "output", "clean_data.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def load_clean_data():
    """读取清洗后的数据"""
    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(
            f"未找到清洗数据: {CLEAN_DATA_PATH}\n请先运行: python src/preprocess.py"
        )
    df = pd.read_csv(CLEAN_DATA_PATH)
    return df


def train_baseline_model():
    """训练 baseline PD 模型"""
    df = load_clean_data()
    
    X = df.drop(columns=["default"])
    y = df["default"]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 标准化（Logistic 对量纲敏感）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 评估
    y_pred_prob_test = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob_test)
    
    print("=" * 50)
    print("Baseline PD Model - Logistic Regression")
    print("=" * 50)
    print(f"AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, model.predict(X_test_scaled)))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, model.predict(X_test_scaled)))
    
    # 对全量数据预测 PD（供 simulation 使用）
    X_full_scaled = scaler.transform(X)
    pd_full = model.predict_proba(X_full_scaled)[:, 1]
    
    # 保存模型和 scaler（便于因果分析：改变 loan_amnt 看 PD 变化）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    # 保存每个 loan 的 PD（含 loan_amnt 便于后续分析）
    output_df = pd.DataFrame({
        "loan_amnt": df["loan_amnt"].values,
        "default": df["default"].values,
        "PD": pd_full,
    })
    output_df.to_csv(os.path.join(OUTPUT_DIR, "pd_predictions.csv"), index=False)
    print(f"\n已保存: {OUTPUT_DIR}/pd_predictions.csv (每个 loan 的 PD)")
    print(f"已保存: {OUTPUT_DIR}/model.pkl, scaler.pkl (用于因果分析)")
    
    return model, scaler, auc, pd_full, X.columns.tolist()


def estimate_pd_with_different_loan_amnt(
    loan_amnt_new: float,
    other_features: dict,
    model_path: str = None,
    scaler_path: str = None,
):
    """
    老师建议：用 model 估计改变 loan_amnt（credit limit proxy）对 PD 的影响。
    示例用法 - 给定其他特征不变，只改 loan_amnt，看 PD 如何变化。
    """
    model_path = model_path or os.path.join(OUTPUT_DIR, "model.pkl")
    scaler_path = scaler_path or os.path.join(OUTPUT_DIR, "scaler.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # other_features 需包含除 loan_amnt 外的所有特征（与训练时一致）
    # 此处为示意，实际使用时需根据 feature 顺序构建 X
    pass  # 可后续扩展


if __name__ == "__main__":
    train_baseline_model()
