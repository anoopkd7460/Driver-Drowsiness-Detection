from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "model_evaluation"
    / "lstm_baseline_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "final_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load test predictions
print("=" * 70)
print("FINAL LSTM EVALUATION VISUALIZATIONS")
print("=" * 70)
print("\nLoading prediction results...")

df = pd.read_csv(PREDICTIONS_FILE)

print(f"Rows loaded: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Identify columns
required_columns = [
    "label",
    "predicted_label",
    "probability_drowsy",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

y_true = df["label"].astype(int).values
y_pred = df["predicted_label"].astype(int).values
y_prob = df["probability_drowsy"].astype(float).values

# Basic validation
print("\nClass distribution:")
print(pd.Series(y_true).value_counts().sort_index())

print("\nPrediction distribution:")
print(pd.Series(y_pred).value_counts().sort_index())

# 1. Confusion matrix
print("\nCreating confusion matrix...")

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)
print(f"\nTN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Not Drowsy", "Drowsy"], fontsize=11)
ax.set_yticklabels(["Not Drowsy", "Drowsy"], fontsize=11)

ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
ax.set_title(
    "LSTM Baseline — Confusion Matrix",
    fontsize=14,
    fontweight="bold",
)

for i in range(2):
    for j in range(2):
        value = cm[i, j]
        ax.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )

plt.tight_layout()

cm_path = OUTPUT_DIR / "confusion_matrix.png"

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {cm_path}")

# 2. ROC curve
print("\nCreating ROC curve...")

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

print(f"ROC-AUC: {roc_auc:.6f}")

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"LSTM Baseline (AUC = {roc_auc:.4f})",
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random Classifier",
)

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title(
    "LSTM Baseline — ROC Curve",
    fontsize=14,
    fontweight="bold",
)

ax.legend(loc="lower right")
ax.grid(True, alpha=0.25)

plt.tight_layout()

roc_path = OUTPUT_DIR / "roc_curve.png"

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {roc_path}")

# 3. Find training history
print("\nSearching for training history...")

possible_history_files = [
    PROJECT_ROOT / "results" / "lstm_baseline" / "history.csv",
    PROJECT_ROOT / "results" / "lstm_baseline" / "training_history.csv",
    PROJECT_ROOT / "results" / "lstm_baseline" / "history.json",
    PROJECT_ROOT / "results" / "lstm_baseline" / "training_history.csv",
]

history_file = None

for file_path in possible_history_files:
    if file_path.exists():
        history_file = file_path
        break

# Search recursively if the expected names weren't found
if history_file is None:
    search_dir = PROJECT_ROOT / "results" / "lstm_baseline"

    if search_dir.exists():
        candidates = list(search_dir.rglob("*.csv"))

        for candidate in candidates:
            try:
                temp = pd.read_csv(candidate)
                columns = set(temp.columns)

                if (
                    "loss" in columns
                    and (
                        "val_loss" in columns
                        or "accuracy" in columns
                        or "val_accuracy" in columns
                    )
                ):
                    history_file = candidate
                    break

            except Exception:
                continue

# Training curves
if history_file is None:
    print("\nWARNING:")
    print("Training history file was not found.")
    print("Confusion matrix and ROC curve were generated successfully.")
    print("Training curves require the original Keras training history.")

else:
    print(f"\nTraining history found:")
    print(history_file)

    history = pd.read_csv(history_file)

    print("\nHistory columns:")
    print(list(history.columns))

    # Loss curve
    if "loss" in history.columns:
        print("\nCreating loss curve...")

        epochs = np.arange(1, len(history) + 1)
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            epochs,
            history["loss"],
            linewidth=2,
            label="Training Loss",
        )

        if "val_loss" in history.columns:
            ax.plot(
                epochs,
                history["val_loss"],
                linewidth=2,
                label="Validation Loss",
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(
            "LSTM Baseline — Training and Validation Loss",
            fontsize=14,
            fontweight="bold",
        )

        ax.legend()
        ax.grid(True, alpha=0.25)

        plt.tight_layout()

        loss_path = OUTPUT_DIR / "training_validation_loss.png"

        plt.savefig(
            loss_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved: {loss_path}")

    # Accuracy curve
    accuracy_column = None

    if "accuracy" in history.columns:
        accuracy_column = "accuracy"
    elif "binary_accuracy" in history.columns:
        accuracy_column = "binary_accuracy"

    val_accuracy_column = None

    if "val_accuracy" in history.columns:
        val_accuracy_column = "val_accuracy"
    elif "val_binary_accuracy" in history.columns:
        val_accuracy_column = "val_binary_accuracy"

    if accuracy_column is not None:
        print("\nCreating accuracy curve...")

        epochs = np.arange(1, len(history) + 1)
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            epochs,
            history[accuracy_column],
            linewidth=2,
            label="Training Accuracy",
        )

        if val_accuracy_column is not None:
            ax.plot(
                epochs,
                history[val_accuracy_column],
                linewidth=2,
                label="Validation Accuracy",
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(
            "LSTM Baseline — Training and Validation Accuracy",
            fontsize=14,
            fontweight="bold",
        )

        ax.legend()
        ax.grid(True, alpha=0.25)

        plt.tight_layout()

        accuracy_path = OUTPUT_DIR / "training_validation_accuracy.png"

        plt.savefig(
            accuracy_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved: {accuracy_path}")

# Save numeric summary
summary = pd.DataFrame(
    {
        "metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives",
        ],
        "value": [
            (tn + tp) / (tn + fp + fn + tp),
            tp / (tp + fp) if (tp + fp) else 0,
            tp / (tp + fn) if (tp + fn) else 0,
            (
                2 * tp / (2 * tp + fp + fn)
                if (2 * tp + fp + fn)
                else 0
            ),
            roc_auc,
            tn,
            fp,
            fn,
            tp,
        ],
    }
)

summary_path = OUTPUT_DIR / "final_test_metrics.csv"

summary.to_csv(
    summary_path,
    index=False,
)

print(f"\nSaved: {summary_path}")

# Final output
print("\n" + "=" * 70)
print("FINAL EVALUATION VISUALIZATION COMPLETE")
print("=" * 70)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    if file.is_file():
        print(f"  - {file.name}")

print("\nOfficial test confusion matrix:")
print(cm)

print(f"\nOfficial test ROC-AUC: {roc_auc:.4f}")
print("=" * 70)