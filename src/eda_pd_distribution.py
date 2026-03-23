"""
PD 分布可视化 - 供 MBA 同学 / Report 使用
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PD_PATH = os.path.join(PROJECT_ROOT, "output", "pd_predictions.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "pd_distribution.png")


def plot_pd_distribution():
    if not os.path.exists(PD_PATH):
        print(f"请先运行 train_model.py 生成 {PD_PATH}")
        return
    
    df = pd.read_csv(PD_PATH)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.histplot(df["PD"], bins=50, ax=axes[0], kde=True)
    axes[0].set_title("PD Distribution")
    axes[0].set_xlabel("Probability of Default")
    
    sns.histplot(df, x="PD", hue="default", bins=50, alpha=0.6, ax=axes[1])
    axes[1].set_title("PD by Actual Default Status")
    axes[1].set_xlabel("Probability of Default")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"已保存: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    plot_pd_distribution()
