# Complete project guide — Credit Limit Allocation

This document ties together **code**, **outputs**, and **final deliverables** for the course project.

## 1. One-time setup

1. Clone the repo and place Kaggle file `accepted_2007_to_2018Q4.csv` under `data/`.
2. Create environment: `pip install -r requirements.txt` (prefer **Anaconda** on Apple Silicon).
3. Run end-to-end: `bash run_pipeline.sh`  
   - Or set `PYTHON=/opt/anaconda3/bin/python` if needed.

## 2. Pipeline steps (automation)

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `src/preprocess.py` | Clean data, label, one-hots → `output/clean_data.csv` |
| 2 | `src/train_model.py` | LR, DT, RF, **HistGradientBoosting**; AUC/PR-AUC; `pd_predictions.csv`; **ROC/PR/calibration PNGs** |
| 3 | `src/eda_pd_distribution.py` | PD histograms per model |
| 4 | `src/decision_logic.py` | Threshold scan vs. profit proxy (not 0.5–centric) |
| 5 | `src/simulation_profit.py` | Counterfactual `loan_amnt` grid → re-score PD → expected profit |

Supporting: `src/causal_pd_analysis.py` (what-if demo), `creditallocation_eda.ipynb` (business EDA).

## 3. Key outputs (under `output/` — gitignored except optional zip)

| File | Role |
|------|------|
| `clean_data.csv` | Modeling table |
| `pd_predictions.csv` | Per-model PD columns + `PD` = best model |
| `model_comparison.csv` | AUC & PR-AUC per model |
| `best_model.pkl`, `scaler.pkl`, `feature_cols.pkl`, `best_model_name.txt` | Scoring & simulation |
| `roc_curves_all_models.png`, `pr_curves_all_models.png`, `calibration_best_model.png` | Slides / report |
| `threshold_profit_scan.csv` | Decision analysis |
| `simulation_portfolio_summary.csv`, `simulation_per_loan_sample.csv` | Prescriptive story |

## 4. Business narrative (for report)

1. **PD changes with exposure** — use `loan_amnt` as limit proxy; same model re-scores counterfactuals (`simulation_profit.py`, `causal_pd_analysis.py`).
2. **Do not center on 0.5** — use `decision_logic.py` + instructor feedback.
3. **Tune** `margin` / `lgd` in `simulation_profit.py` to match your financial assumptions.

## 5. Optional bundle for teammates

```bash
bash package_output.sh
```

## 6. Written submissions

- `deliverable3_project_update.tex` — status report (update numbers from `model_comparison.csv`).
- Final presentation: use figures in `output/` and notebook.
- `Report_of_edit.docx` — merge MBA narrative as appropriate.

## 7. What is *not* included (by design)

- Production deployment, real-time API.
- XGBoost/LightGBM **libraries** (optional upgrade; we include **sklearn HistGradientBoosting**).
- Regulatory sign-off, fairness deep-dive (discuss as limitation).
