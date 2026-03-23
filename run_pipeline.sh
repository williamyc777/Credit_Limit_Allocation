#!/bin/bash
# 完整流程：数据清洗 -> 训练模型 -> PD 分布图
cd "$(dirname "$0")"

echo "Step 1: 数据清洗..."
python src/preprocess.py

echo ""
echo "Step 2: 训练 PD 模型..."
python src/train_model.py

echo ""
echo "Step 3: 生成 PD 分布图..."
python src/eda_pd_distribution.py

echo ""
echo "完成！交付物在 output/ 目录"
