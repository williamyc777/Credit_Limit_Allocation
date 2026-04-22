#!/usr/bin/env bash
# Pack modeling outputs into one zip for GitHub / teammates.
# Run after: bash run_pipeline.sh
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ZIP="output/model_outputs_bundle.zip"
mkdir -p output

# Artifacts from src/train_model.py (multi-model) + preprocess + EDA plots
FILES=(
  output/clean_data.csv
  output/pd_predictions.csv
  output/scaler.pkl
  output/best_model.pkl
  output/all_models.pkl
  output/feature_cols.pkl
  output/best_model_name.txt
  output/model_comparison.csv
)

missing=()
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || missing+=("$f")
done
if ((${#missing[@]})); then
  echo "Missing files — run the pipeline first (e.g. bash run_pipeline.sh):"
  printf '  %s\n' "${missing[@]}"
  exit 1
fi

rm -f "$ZIP"
zip -j "$ZIP" "${FILES[@]}"
# Optional PNGs from eda_pd_distribution.py (include if present)
shopt -s nullglob
for png in output/PD_*_distribution.png output/pd_distribution_comparison.png; do
  [[ -f "$png" ]] && zip -j "$ZIP" "$png"
done
shopt -u nullglob

# Optional analysis outputs (if present)
for opt in \
  output/threshold_profit_scan.csv \
  output/simulation_portfolio_summary.csv \
  output/simulation_per_loan_sample.csv
do
  [[ -f "$opt" ]] && zip -j "$ZIP" "$opt"
done

echo "Created: $ZIP  ($(du -h "$ZIP" | awk '{print $1}'))"
echo "If du -h shows ~100MB or more, GitHub may reject the push — use GitHub Releases or cloud storage instead."
