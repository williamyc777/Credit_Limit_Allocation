# Credit Limit Allocation 

**Status:** see [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for what is done vs.\ optional extensions.

This repository supports a **credit-line / lending decision** project built on **Lending Club** loan data. The goal is to estimate **probability of default (PD)** with a transparent baseline model, then use those estimates (and later **counterfactual changes** in exposure) to inform prescriptive decisions—e.g., how adjusting an amount analogous to a credit limit affects PD and expected outcomes.

---

## Project status (completed so far)

The following baseline pipeline is **implemented and runnable end-to-end**:

| Milestone | Description |
|-----------|-------------|
| **Data sourcing** | Uses Kaggle dataset [Lending Club — accepted loans](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (`accepted_2007_to_2018Q4.csv`). |
| **Data cleaning** | Selects core features; restricts to **terminal** outcomes only (`Fully Paid` vs `Charged Off`) so labels are not contaminated by ongoing loans (e.g. `Current`). |
| **Label construction** | Binary `default`: `1` = `Charged Off`, `0` = `Fully Paid`. |
| **Feature handling** | Missing values dropped (simple baseline); categoricals one-hot encoded (`term`, `grade`, `emp_length`, `home_ownership`). `int_rate` parsed if stored as strings with `%`. |
| **PD models (baseline + comparison)** | **Logistic regression**, **decision tree**, and **random forest**; compare **ROC-AUC** and **PR-AUC** (average precision; useful under imbalance); save **`best_model.pkl`**, `all_models.pkl`, `scaler.pkl` (for logistic), `feature_cols.pkl`, `model_comparison.csv`. Per-model PD columns in `pd_predictions.csv` plus **`PD`** / **`PD_best_model`** (aligned to the AUC-best model). |
| **Outputs** | `clean_data.csv`, `pd_predictions.csv` (multi-column PDs + `default` + `loan_amnt`), model artifacts as above, optional EDA figures (`PD_*_distribution.png`, `pd_distribution_comparison.png`) from `eda_pd_distribution.py`. |
| **Causal / prescriptive hook** | `loan_amnt` is retained as a **proxy for credit limit** in the feature set; the saved model can be used to **re-score** scenarios where only `loan_amnt` (or related exposure) changes—aligned with instructor feedback on **prescriptive analytics** and Chapter 11-style “what-if” on a single model. |
| **Simulation** | `simulation_profit.py` — grid of `loan_amnt` multipliers, **re-score PD**, **expected profit** proxy, portfolio summary + per-loan sample export. |
| **EDA notebook** | `creditallocation_eda.ipynb` — team exploratory analysis tied to **credit-limit optimization** (see below). |

**Note (decision logic):** We do **not** treat **0.5** as a meaningful business cutoff—see instructor feedback. `sklearn`’s `.predict()` uses 0.5 for class labels, but **prescriptive** decisions should follow **explicit rules** (e.g. expected profit, cost asymmetry, or limit simulation). After training, run **`python src/decision_logic.py`** to scan thresholds with a simple **retrospective profit proxy** on the test split and compare to an arbitrary 0.5. Primary project use of PD remains **continuous** scores + **counterfactual** `loan_amnt` and portfolio-level **simulation**.

---

## Exploratory analysis: `creditallocation_eda.ipynb`

The Jupyter notebook **`creditallocation_eda.ipynb`** (repository root) complements the Python pipeline. It reads only **`output/clean_data.csv`** and **`output/pd_predictions.csv`**—the same files produced by `preprocess.py` and `train_model.py` (no extra datasets).

**Purpose for the business problem**

- **PD is not fixed when limits change.** The course emphasis is *prescriptive* analytics: if we treat **`loan_amnt` as a proxy for credit limit**, then **changing that exposure should change predicted PD**. The notebook motivates that story with **data**, not only with the model.
- **Action variable focus:** It highlights **`loan_amnt`** with **quantile bins** and **default rates by bin**, so the team can see empirically how larger amounts relate to default frequency before running **simulations** that **re-score PD** under counterfactual loan amounts.
- **Risk context for decisions:** Plots and markdown cover **FICO**, **interest rate**, **DTI**, and key **categorical** risk tags (grade, term, home ownership), each tied to **why** those views matter when adjusting limits and expected profit.
- **Link to the model:** The notebook includes **PD vs. `loan_amnt`** and a **ROC / AUC** view using stored predictions, connecting exploratory patterns to the **baseline PD model** the pipeline trains.

**How to run**

1. Generate `output/clean_data.csv` and `output/pd_predictions.csv` (e.g. `bash run_pipeline.sh` or the `python src/...` steps).  
2. Open **`creditallocation_eda.ipynb`** with the **working directory set to the repository root** (the folder that contains `output/`).

A closing section in the notebook, **“Implications for Credit Limit Optimization,”** summarizes how EDA supports the later steps: **simulate alternative limits → recompute PD → optimize expected profit**.

---

## Repository layout

```
project/
├── creditallocation_eda.ipynb  # EDA: risk drivers, loan_amnt bins, PD linkage, optimization narrative
├── data/           # Place Kaggle CSV here (not committed; too large)
├── output/         # Generated artifacts (not committed)
├── src/
│   ├── preprocess.py           # Cleaning + labels
│   ├── train_model.py          # Logistic baseline + PD export
│   ├── eda_pd_distribution.py  # PD histograms
│   └── causal_pd_analysis.py   # Demo: PD vs. scaled loan_amnt
├── requirements.txt
├── run_pipeline.sh
├── package_output.sh     # zip outputs for sharing (optional)
└── README.md
```

**GitHub:** Raw Kaggle files and loose `output/*` (CSV, PKL, PNG) are **gitignored** to keep the repo small. After cloning, download the CSV locally and rerun the pipeline to regenerate outputs—or use the **bundle zip** (below) if a teammate committed it.

### Optional: share outputs as one zip on GitHub

You **can** commit a **single archive** so teammates get artifacts without rerunning the full pipeline:

```bash
bash run_pipeline.sh          # generate CSV / PKL / PNG under output/
bash package_output.sh        # creates output/model_outputs_bundle.zip
git add output/model_outputs_bundle.zip
git commit -m "Add bundled model outputs"
git push
```

Teammates unzip `model_outputs_bundle.zip` into their local `output/` folder (same filenames as the pipeline).

**GitHub limits:** Pushes fail if any file is **≥ ~100 MB**. Full `clean_data.csv` bundles often exceed that even when zipped—if `git push` is rejected, use **GitHub Releases** (attach the zip there), **Git LFS**, or cloud storage instead of committing the zip.

---

## Setup

```bash
cd Credit_Limit_Allocation   # or your clone folder name
pip install -r requirements.txt
```

Download `accepted_2007_to_2018Q4.csv` from Kaggle and save it under `data/`.

---

## Run the pipeline

**Option A — shell script**

```bash
bash run_pipeline.sh
```

**Option B — step by step**

```bash
python src/preprocess.py          # cleaning + labels → output/clean_data.csv
python src/train_model.py         # train, evaluate, export PD + model artifacts
python src/eda_pd_distribution.py # optional: output/pd_distribution.png
```

After training, you can run:

```bash
python src/causal_pd_analysis.py  # what-if: change loan_amnt → PD (uses best_model.pkl)
```

`causal_pd_analysis.py` requires training to have produced `best_model.pkl`, `scaler.pkl`, and `feature_cols.pkl` (run `train_model.py` after `preprocess.py`).

```bash
python src/decision_logic.py                 # optional: threshold scan vs. profit proxy (not 0.5-centric)
# Optional: python src/decision_logic.py --revenue 1.0 --loss 8.0
```
Outputs `output/threshold_profit_scan.csv`.

```bash
python src/simulation_profit.py
# Optional: python src/simulation_profit.py --margin 0.08 --lgd 0.45 --sample-rows 100000
```

**Counterfactual limits + expected profit (course prescriptive story):** `simulation_profit.py` scales `loan_amnt` by a grid of multipliers, **recomputes PD** with the saved best model for each counterfactual, and evaluates a stylized one-period **expected profit** per loan, then picks the multiplier with highest profit. Writes `output/simulation_portfolio_summary.csv` and a small `output/simulation_per_loan_sample.csv`. This is the bridge from **PD(model)** to **optimal limit on a grid** — complementing `decision_logic.py` (thresholding) and professor guidance on **changing exposure → new PD → new profit**.

> Use the same Python environment that can run `sklearn` (e.g. Anaconda on Apple Silicon) if the system `python3` shows NumPy arch errors.

---

## Generated artifacts

| Path | Purpose |
|------|---------|
| `output/clean_data.csv` | Clean modeling table |
| `output/pd_predictions.csv` | Per-loan columns: `loan_amnt`, `default`, `PD_logistic_regression`, `PD_decision_tree`, `PD_random_forest`, `PD_best_model`, and **`PD`** (copy of best) |
| `output/best_model.pkl` | AUC-selected model for deployment / `causal_pd_analysis.py` |
| `output/all_models.pkl` | Dict of all fitted models |
| `output/scaler.pkl` | `StandardScaler` (features used for logistic) |
| `output/feature_cols.pkl` | Column order for scoring |
| `output/best_model_name.txt` | Name of best model (e.g. `random_forest`) |
| `output/model_comparison.csv` | AUC by model |
| `output/PD_*_distribution.png`, `output/pd_distribution_comparison.png` | From `eda_pd_distribution.py` |
| `output/model_outputs_bundle.zip` | **Optional:** `package_output.sh` bundles key CSV/PKL/TXT; may be committed if under GitHub’s file-size limit |

---

## References

- Kaggle: [All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Course framing: prescriptive analytics, causal-style **counterfactual scoring** via feature perturbation (e.g. `loan_amnt`), and eventual **decision logic** / profit objectives.
