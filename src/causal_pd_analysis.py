"""
老师建议的因果分析：估计改变 loan_amnt（credit limit proxy）对 PD 的影响

用法：给定一个 loan 的特征，改变 loan_amnt，看模型预测的 PD 如何变化。
可用于后续优化 credit limit 以最大化预期利润。
"""

import os
import pickle
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CLEAN_DATA_PATH = os.path.join(PROJECT_ROOT, "output", "clean_data.csv")


def load_model_and_scaler():
    with open(os.path.join(OUTPUT_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def estimate_pd_change_with_loan_amnt(loan_amnt_new: float, row: pd.Series, feature_cols: list):
    """
    给定一行数据，将 loan_amnt 改为新值，预测 PD 变化。
    row: 来自 clean_data 的一行（含所有特征）
    """
    model, scaler = load_model_and_scaler()
    
    X = row[feature_cols].copy()
    X["loan_amnt"] = loan_amnt_new
    X = X.reindex(feature_cols).values.reshape(1, -1)
    
    X_scaled = scaler.transform(X)
    pd_new = model.predict_proba(X_scaled)[:, 1][0]
    return pd_new


def demo_causal_effect():
    """演示：对若干 loan，展示 loan_amnt 变化时 PD 的变化"""
    df = pd.read_csv(CLEAN_DATA_PATH)
    model, scaler = load_model_and_scaler()
    
    feature_cols = [c for c in df.columns if c != "default"]
    
    # 取前 5 个样本做演示
    sample = df.head(5)
    
    print("因果分析演示：改变 loan_amnt 对 PD 的影响")
    print("=" * 60)
    
    for idx, row in sample.iterrows():
        loan_amnt_orig = row["loan_amnt"]
        
        # 测试 ±20% loan_amnt
        for mult in [0.8, 1.0, 1.2]:
            amnt = loan_amnt_orig * mult
            pd_new = estimate_pd_change_with_loan_amnt(amnt, row, feature_cols)
            print(f"  loan_amnt: ${amnt:,.0f} -> PD: {pd_new:.4f}")
        print()


if __name__ == "__main__":
    demo_causal_effect()
