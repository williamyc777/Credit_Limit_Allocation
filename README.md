# Credit Limit Allocation — DS & AI for Business

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
| **Baseline PD model** | **Logistic regression** with **standardized** numeric/boolean features; stratified train/test split. |
| **Outputs** | `clean_data.csv`, per-loan **PD** in `pd_predictions.csv` (includes `loan_amnt` and realized `default` for downstream use), saved **`model.pkl`** and **`scaler.pkl`**, optional **PD distribution** plot. |
| **Causal / prescriptive hook** | `loan_amnt` is retained as a **proxy for credit limit** in the feature set; the saved model can be used to **re-score** scenarios where only `loan_amnt` (or related exposure) changes—aligned with instructor feedback on **prescriptive analytics** and Chapter 11-style “what-if” on a single model. |

**Note:** The default **0.5 classification threshold** under class imbalance yields **low recall on defaults**; **AUC** and **predicted PD** are still useful for ranking and for simulations that consume continuous PD. Improving recall (e.g. `class_weight`, threshold tuning, or richer models) is a natural next iteration—not required to unblock the rest of the workflow.

---

## Repository layout

```
project/
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
python src/causal_pd_analysis.py  # illustrative PD sensitivity to loan_amnt
```

---

## Generated artifacts

| Path | Purpose |
|------|---------|
| `output/clean_data.csv` | Clean modeling table |
| `output/pd_predictions.csv` | Per-loan PD (+ `loan_amnt`, `default`) |
| `output/model.pkl` | Fitted logistic model |
| `output/scaler.pkl` | Fitted `StandardScaler` (same feature order as training) |
| `output/pd_distribution.png` | EDA figure |
| `output/model_outputs_bundle.zip` | **Optional:** all of the above in one zip (`package_output.sh`); may be committed if under GitHub’s file-size limit |

---

## References

- Kaggle: [All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Course framing: prescriptive analytics, causal-style **counterfactual scoring** via feature perturbation (e.g. `loan_amnt`), and eventual **decision logic** / profit objectives.
