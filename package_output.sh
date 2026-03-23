#!/usr/bin/env bash
# Pack modeling outputs into one zip for GitHub / teammates.
# Run after: bash run_pipeline.sh
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ZIP="output/model_outputs_bundle.zip"
mkdir -p output

FILES=(
  output/clean_data.csv
  output/pd_predictions.csv
  output/model.pkl
  output/scaler.pkl
  output/pd_distribution.png
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
echo "Created: $ZIP  ($(du -h "$ZIP" | awk '{print $1}'))"
echo "If du -h shows ~100MB or more, GitHub may reject the push — use GitHub Releases or cloud storage instead."
