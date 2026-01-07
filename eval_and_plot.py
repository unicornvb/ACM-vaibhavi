# src/eval_and_plot.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

ARTIFACT_DIR = "artifacts"
PREDICTIONS_FILE = "test_predictions.csv"


def plot_from_test_set():
    """Generate evaluation plots from test set predictions."""

    csv_path = os.path.join(ARTIFACT_DIR, PREDICTIONS_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{PREDICTIONS_FILE} not found in {ARTIFACT_DIR}")

    df = pd.read_csv(csv_path)

    # -----------------------------
    # Confusion Matrix (Classification)
    # -----------------------------
    labels = sorted(df["true_class"].unique())
    cm = confusion_matrix(df["true_class"], df["pred_class"], labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # -----------------------------
    # Regression Scatter Plot
    # -----------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(df["true_score"], df["pred_score"], alpha=0.6)
    mn = min(df["true_score"].min(), df["pred_score"].min())
    mx = max(df["true_score"].max(), df["pred_score"].max())
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("True Difficulty Score")
    plt.ylabel("Predicted Difficulty Score")
    plt.title("Regression: True vs Predicted (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "regression_scatter.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    plot_from_test_set()
    print("Evaluation plots saved to artifacts/")
