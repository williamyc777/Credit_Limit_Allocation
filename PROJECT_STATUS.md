# Project status — Credit Limit Allocation

## Implemented (end-to-end in repo)

| Area | Artifacts |
|------|-----------|
| **Data** | `src/preprocess.py` → `output/clean_data.csv` (terminal labels, one-hot) |
| **Modeling** | `src/train_model.py` — **Logistic**, **Decision Tree**, **Random Forest**, **HistGradientBoosting**; **AUC + PR-AUC**; `pd_predictions.csv`, `best_model.pkl`, `model_comparison.csv` |
| **Model figures** | `output/roc_curves_all_models.png`, `pr_curves_all_models.png`, `calibration_best_model.png` (from `evaluation_plots.py`) |
| **Decision logic** | `src/decision_logic.py` — threshold grid vs. profit proxy (not 0.5–centric) |
| **Prescriptive** | `src/simulation_profit.py` — counterfactual `loan_amnt` multipliers, re-score PD, expected profit on a grid; portfolio + sample CSV |
| **Causal demo** | `src/causal_pd_analysis.py` — what-if `loan_amnt` → PD (uses best model) |
| **EDA** | `src/eda_pd_distribution.py`, `creditallocation_eda.ipynb` |
| **Automation** | `run_pipeline.sh` (auto-picks Anaconda python when present), `package_output.sh` |
| **Docs** | `README.md`, `docs/COMPLETE_PROJECT_GUIDE.md`, `deliverable3_project_update.tex`, `Report_of_edit.docx` (as provided) |

## Not in Git (regenerate locally)

- `data/accepted_2007_to_2018Q4.csv` (Kaggle)  
- `output/*` (gitignored except optional zip policy) — run `run_pipeline.sh` then optionally `bash package_output.sh`

## Optional improvements (not required to “close” the course project)

- **XGBoost / LightGBM** — add as extra model in `train_model` or separate script; tune runtime on 1.2M rows.  
- **Fairness / demographic** — LendingClub has limited protected attributes; discuss limitations in report.  
- **Calibration (extra)** — beyond the saved `calibration_best_model.png` isotonic, teams may add Platt scaling in report text.  
- **Deploy** — out of scope for class; keep as “future work.”  
- **Revenue / LGD** — replace stylized `margin`/`lgd` in `simulation_profit.py` with team’s business assumptions.  
- **XGBoost / LightGBM** (separate from sklearn HGB) — optional if you need a fifth comparison model.

## Quick verify

```bash
bash run_pipeline.sh
```

Set `PYTHON=/path/to/python` if the auto-detect fails.
