# DS&AI Project - PD Model 模块

William 负责：数据清洗、Label 构建、Baseline PD Model（Logistic Regression）

> **GitHub**：仓库不含 Kaggle 原始 CSV 与 `output/` 生成文件（体积过大）。克隆后请将 `accepted_2007_to_2018Q4.csv` 放入 `data/`，再运行 pipeline。

## 快速开始

### 1. 环境准备

```bash
cd project
pip install -r requirements.txt
```

### 2. 下载数据

从 [Kaggle - Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) 下载，将 `accepted_2007_to_2018Q4.csv` 放入 `data/` 目录。

### 3. 执行流程

```bash
# Step 1: 数据清洗 + Label
python src/preprocess.py

# Step 2: 训练模型 + 输出 PD
python src/train_model.py

# Step 3 (可选): PD 分布图
python src/eda_pd_distribution.py
```

## 交付物

| 文件 | 说明 |
|------|------|
| `output/clean_data.csv` | 清洗后数据，全队可用 |
| `output/pd_predictions.csv` | 每个 loan 的 PD，simulation 用 |
| `output/model.pkl` | 训练好的 Logistic 模型 |
| `output/scaler.pkl` | 特征标准化器 |
| `output/pd_distribution.png` | PD 分布图 |

## 老师建议：因果分析

老师指出这是 **prescriptive analytics** 问题：调整信用额度会改变 PD。

- `loan_amnt` 作为 credit limit 的 proxy 已包含在模型中
- 可通过修改 `loan_amnt`，用同一模型估计「改变信用额度对 PD 的影响」
- 进而可优化 credit limit 以最大化预期利润（参考 Chapter 11）

模型和 scaler 已保存，便于后续实现该因果分析逻辑。
