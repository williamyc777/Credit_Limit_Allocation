"""
EDA for PD distributions from multiple models.

This script:
1. Reads output/pd_predictions.csv
2. Plots PD histograms for:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Best Model
3. Saves figures to output/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
PD_PRED_PATH = os.path.join(OUTPUT_DIR, "pd_predictions.csv")


def load_pd_predictions():
    """Load pd_predictions.csv"""
    if not os.path.exists(PD_PRED_PATH):
        raise FileNotFoundError(
            f"Cannot find: {PD_PRED_PATH}\n"
            f"Please run: python src/train_model.py first"
        )
    return pd.read_csv(PD_PRED_PATH)


def plot_single_histogram(series, title, output_file, bins=30):
    """Plot one PD histogram"""
    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel("Predicted Probability of Default (PD)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def main():
    df = load_pd_predictions()

    model_columns = {
        "PD_logistic_regression": "Logistic Regression",
        "PD_decision_tree": "Decision Tree",
        "PD_random_forest": "Random Forest",
        "PD_best_model": "Best Model"
    }

    print("=" * 60)
    print("PD Distribution EDA")
    print("=" * 60)

    for col, model_name in model_columns.items():
        if col not in df.columns:
            print(f"Skipping {model_name}: column '{col}' not found.")
            continue

        print(f"\n{model_name}")
        print("-" * 40)
        print(df[col].describe())

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{col}_distribution.png"
        )

        plot_single_histogram(
            df[col],
            f"PD Distribution - {model_name}",
            output_file
        )

        print(f"Saved plot: {output_file}")

    # optional comparison overlay plot
    compare_cols = [col for col in model_columns if col in df.columns]

    if len(compare_cols) >= 2:
        plt.figure(figsize=(10, 6))

        for col in compare_cols:
            plt.hist(
                df[col],
                bins=30,
                alpha=0.4,
                label=model_columns[col],
                edgecolor="black"
            )

        plt.title("PD Distribution Comparison Across Models")
        plt.xlabel("Predicted Probability of Default (PD)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()

        comparison_file = os.path.join(OUTPUT_DIR, "pd_distribution_comparison.png")
        plt.savefig(comparison_file, dpi=300)
        plt.close()

        print(f"\nSaved comparison plot: {comparison_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
