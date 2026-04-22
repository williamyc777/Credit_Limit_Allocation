#!/usr/bin/env bash
# 完整流程：数据清洗 -> 训练 -> EDA -> 决策阈值 -> 利润 simulation
# 使用与 sklearn 兼容的 Python（优先 Anaconda，避免 Apple Silicon 上系统 python 的 numpy 架构问题）
set -e
cd "$(dirname "$0")"

if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "/opt/anaconda3/bin/python" ]]; then
  PYTHON="/opt/anaconda3/bin/python"
elif [[ -x "$HOME/anaconda3/bin/python" ]]; then
  PYTHON="$HOME/anaconda3/bin/python"
elif [[ -x "$HOME/miniconda3/bin/python" ]]; then
  PYTHON="$HOME/miniconda3/bin/python"
else
  PYTHON="python3"
fi

echo "Using: $PYTHON"
"$PYTHON" -c "import sklearn; print('sklearn', sklearn.__version__)" 2>/dev/null || {
  echo "Error: need a working Python with scikit-learn. Set PYTHON=/path/to/python and retry."
  exit 1
}

echo "Step 1: 数据清洗..."
"$PYTHON" src/preprocess.py

echo ""
echo "Step 2: 训练 PD 模型..."
"$PYTHON" src/train_model.py

echo ""
echo "Step 3: 生成 PD 分布图..."
"$PYTHON" src/eda_pd_distribution.py

echo ""
echo "Step 4: 决策逻辑 / 阈值扫描 (非 0.5 中心)..."
"$PYTHON" src/decision_logic.py

echo ""
echo "Step 5: 反事实额度 + 期望利润 simulation..."
"$PYTHON" src/simulation_profit.py

echo ""
echo "完成！交付物在 output/ 目录"
