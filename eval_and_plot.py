# src/eval_and_plot.py
import joblib, os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix

ARTIFACT_DIR = 'artifacts'

def plot_from_holdout():
    hold_csv = os.path.join(ARTIFACT_DIR, 'holdout_predictions.csv')
    df = pd.read_csv(hold_csv)
    labels = sorted(df['true_class'].unique())
    cm = confusion_matrix(df['true_class'], df['pred_class'], labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(ARTIFACT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(df['true_score'], df['pred_score'], alpha=0.6)
    mn = min(df['true_score'].min(), df['pred_score'].min())
    mx = max(df['true_score'].max(), df['pred_score'].max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('True score')
    plt.ylabel('Pred score')
    plt.title('Regression: True vs Predicted')
    plt.savefig(os.path.join(ARTIFACT_DIR, 'regression_scatter.png'), dpi=150)
    plt.close()

if __name__=='__main__':
    plot_from_holdout()
    print("Plots saved to artifacts/")
