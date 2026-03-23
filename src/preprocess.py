"""
数据清洗 + Label 构建
DS&AI Project - William (Tech Lead + Modeling)

⚠️ 重要：只保留 Fully Paid 和 Charged Off，排除 Current 等未结清贷款（会污染 label）
⚠️ 重要：loan_amnt 必须保留 - 后续 simulation 和老师建议的因果分析（信用额度调整对 PD 的影响）都需要
"""

import pandas as pd
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "accepted_2007_to_2018Q4.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "clean_data.csv")


def load_data():
    """读取原始数据"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"未找到数据文件: {DATA_PATH}\n"
            "请从 Kaggle 下载: https://www.kaggle.com/datasets/wordsforthewise/lending-club\n"
            "下载后解压到 project/data/ 目录"
        )
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗主流程"""
    
    # 1. 保留关键字段（含 loan_amnt - 老师建议：可作为 credit limit 的 proxy，用于因果分析）
    cols = [
        "loan_amnt",      # 必须保留！simulation + 信用额度调整的因果效应分析
        "term",
        "int_rate",
        "grade",
        "emp_length",
        "home_ownership",
        "annual_inc",
        "dti",
        "fico_range_low",
        "loan_status",
    ]
    
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少列: {missing_cols}\n可用列: {list(df.columns[:20])}...")
    
    df = df[cols].copy()
    
    # 数值转换：int_rate 可能是 "12.5%" 格式
    if df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "").astype(float)
    
    # 2. 处理 label - 只保留已结清贷款，排除 Current（坑1）
    valid_statuses = ["Fully Paid", "Charged Off"]
    before_filter = len(df)
    df = df[df["loan_status"].isin(valid_statuses)]
    print(f"Label 过滤 (仅 Fully Paid / Charged Off): {before_filter} -> {len(df)} 行")
    
    df["default"] = (df["loan_status"] == "Charged Off").astype(int)
    df = df.drop(columns=["loan_status"])
    
    # 3. 处理缺失值
    before_dropna = len(df)
    df = df.dropna()
    print(f"剔除缺失值: {before_dropna} -> {len(df)} 行")
    
    # 4. 处理类别变量 (one-hot)
    cat_cols = ["term", "grade", "emp_length", "home_ownership"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    print(f"清洗后: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"违约率: {df['default'].mean():.2%}")
    
    return df


def main():
    df = load_data()
    df_clean = preprocess(df)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\n已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
