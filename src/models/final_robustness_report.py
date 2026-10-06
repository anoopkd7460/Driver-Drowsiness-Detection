import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
TEST_METRICS_FILE = "results/model_evaluation/test_metrics.csv"
REPEATED_RESULTS_FILE = "results/generalization/repeated_video_level_results.csv"
VIDEO_RESULTS_FILE = "results/generalization/video_wise_results.csv"
OUTPUT_DIR = "results/final_robustness"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper functions
def find_file(path):
    if os.path.exists(path):
        return path

    print(f"WARNING: File not found: {path}")
    return None


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name

    return None


def print_columns(df, title):
    print(f"\n{title}")

    for column in df.columns:
        print(f"  - {column}")


# Start
print("=" * 70)
print("FINAL ROBUSTNESS REPORT")
print("=" * 70)

# 1. Official test results
print("\n[1] Loading official test results...")

test_file = find_file(TEST_METRICS_FILE)
test_df = None

if test_file:
    test_df = pd.read_csv(test_file)

    print(f"Loaded {len(test_df)} model results.")
    print("\nOfficial Test Comparison:")

    print(
        test_df[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        ].to_string(index=False)
    )

    test_df.to_csv(
        os.path.join(OUTPUT_DIR, "model_comparison.csv"),
        index=False,
    )

# 2. Repeated video-level generalization
print("\n[2] Loading repeated video-level results...")

repeated_file = find_file(REPEATED_RESULTS_FILE)
repeated_df = None
robustness_summary = None

if repeated_file:
    repeated_df = pd.read_csv(repeated_file)

    print(f"Loaded {len(repeated_df)} experiments.")

    print_columns(
        repeated_df,
        "Available columns:",
    )

    print("\nRepeated Video-Level Results:")
    print(repeated_df.to_string(index=False))

    # Detect metric columns
    accuracy_col = find_column(
        repeated_df,
        ["accuracy", "val_accuracy"],
    )

    precision_col = find_column(
        repeated_df,
        ["precision", "val_precision"],
    )

    recall_col = find_column(
        repeated_df,
        ["recall", "val_recall"],
    )

    f1_col = find_column(
        repeated_df,
        ["f1", "f1_score", "val_f1"],
    )

    seed_col = find_column(
        repeated_df,
        ["seed", "random_seed"],
    )

    # Save original repeated results
    repeated_df.to_csv(
        os.path.join(OUTPUT_DIR, "repeated_video_results.csv"),
        index=False,
    )

    # Calculate aggregate statistics
    summary_rows = []

    metric_mapping = {
        "Accuracy": accuracy_col,
        "Precision": precision_col,
        "Recall": recall_col,
        "F1": f1_col,
    }

    for metric_name, column in metric_mapping.items():
        if column is None:
            print(f"WARNING: {metric_name} column not found.")
            continue

        values = pd.to_numeric(
            repeated_df[column],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            continue

        summary_rows.append(
            {
                "metric": metric_name,
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "minimum": values.min(),
                "maximum": values.max(),
            }
        )

    robustness_summary = pd.DataFrame(summary_rows)

    print("\nRepeated Video-Level Summary:")

    if len(robustness_summary) > 0:
        print(robustness_summary.to_string(index=False))

        robustness_summary.to_csv(
            os.path.join(OUTPUT_DIR, "robustness_summary.csv"),
            index=False,
        )

# 3. Video-wise error analysis
print("\n[3] Loading video-wise error analysis...")

video_file = find_file(VIDEO_RESULTS_FILE)
video_df = None

if video_file:
    video_df = pd.read_csv(video_file)

    print(f"Loaded {len(video_df)} video-level records.")

    print_columns(
        video_df,
        "Available columns:",
    )

    print("\nVideo-Level Results:")
    print(video_df.to_string(index=False))

    video_df.to_csv(
        os.path.join(OUTPUT_DIR, "video_error_summary.csv"),
        index=False,
    )

# 4. Final metrics summary
print("\n[4] Creating final metrics summary...")

final_rows = []

# Official test
if test_df is not None:
    for _, row in test_df.iterrows():
        final_rows.append(
            {
                "evaluation_type": "Official Test",
                "model": row["model"],
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "roc_auc": row["roc_auc"],
            }
        )

# Repeated validation aggregate
if robustness_summary is not None:
    aggregate = {
        "evaluation_type": "Repeated Video-Level Validation",
        "model": "LSTM Baseline",
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
    }

    for _, row in robustness_summary.iterrows():
        metric = row["metric"]

        if metric == "Accuracy":
            aggregate["accuracy"] = row["mean"]
        elif metric == "Precision":
            aggregate["precision"] = row["mean"]
        elif metric == "Recall":
            aggregate["recall"] = row["mean"]
        elif metric == "F1":
            aggregate["f1"] = row["mean"]

    final_rows.append(aggregate)

final_metrics_df = pd.DataFrame(final_rows)

final_metrics_file = os.path.join(
    OUTPUT_DIR,
    "final_metrics_summary.csv",
)

final_metrics_df.to_csv(
    final_metrics_file,
    index=False,
)

print("\nFinal Metrics Summary:")
print(final_metrics_df.to_string(index=False))

# 5. Model accuracy comparison
if test_df is not None:
    print("\n[5] Creating model accuracy comparison...")

    plt.figure(figsize=(9, 6))

    plt.bar(
        test_df["model"],
        test_df["accuracy"] * 100,
    )

    plt.ylabel("Accuracy (%)")
    plt.xlabel("Model")
    plt.title("Official Test Accuracy Comparison")
    plt.ylim(0, 105)
    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "model_accuracy_comparison.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

# 6. Repeated validation accuracy
if repeated_df is not None and accuracy_col is not None:
    print("\n[6] Creating robustness accuracy plot...")

    accuracy_values = (
        pd.to_numeric(
            repeated_df[accuracy_col],
            errors="coerce",
        )
        * 100
    )

    if seed_col is not None:
        x_values = repeated_df[seed_col].values
        x_label = "Random Seed"
    else:
        x_values = np.arange(1, len(repeated_df) + 1)
        x_label = "Repeated Video-Level Experiment"

    plt.figure(figsize=(9, 6))

    plt.plot(
        x_values,
        accuracy_values,
        marker="o",
    )

    plt.axhline(
        95,
        linestyle="--",
        label="95% Target",
    )

    plt.xlabel(x_label)
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Repeated Video-Level Validation")
    plt.ylim(90, 102)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "robustness_accuracy.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

# 7. Metric robustness
if repeated_df is not None:
    print("\n[7] Creating metric robustness plot...")

    available_metrics = []

    metric_mapping = {
        "Accuracy": accuracy_col,
        "Precision": precision_col,
        "Recall": recall_col,
        "F1": f1_col,
    }

    for metric_name, column in metric_mapping.items():
        if column is not None:
            available_metrics.append((metric_name, column))

    if available_metrics:
        plt.figure(figsize=(10, 6))

        for metric_name, column in available_metrics:
            values = (
                pd.to_numeric(
                    repeated_df[column],
                    errors="coerce",
                )
                * 100
            )

            x_values = np.arange(1, len(values) + 1)

            plt.plot(
                x_values,
                values,
                marker="o",
                label=metric_name,
            )

        plt.xlabel("Repeated Video-Level Experiment")
        plt.ylabel("Score (%)")
        plt.title("Robustness Across Video-Level Splits")
        plt.ylim(70, 102)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "metric_robustness.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

# 8. Difficult video analysis
if video_df is not None:
    print("\n[8] Identifying difficult videos...")

    video_accuracy_col = find_column(
        video_df,
        [
            "accuracy",
            "mean_accuracy",
            "video_accuracy",
        ],
    )

    if video_accuracy_col is not None:
        video_df[video_accuracy_col] = pd.to_numeric(
            video_df[video_accuracy_col],
            errors="coerce",
        )

        difficult_videos = (
            video_df
            .sort_values(
                video_accuracy_col,
                ascending=True,
            )
            .head(10)
        )

        print("\nMost difficult videos:")
        print(difficult_videos.to_string(index=False))

        difficult_videos.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "difficult_videos.csv",
            ),
            index=False,
        )

# 9. Final model selection
print("\n[9] Determining final model...")

final_model = None

if test_df is not None:
    ranked = test_df.sort_values(
        by=["f1", "accuracy"],
        ascending=False,
    )

    final_model = ranked.iloc[0]["model"]

if final_model is not None:
    decision = pd.DataFrame(
        [
            {
                "final_model": final_model,
                "reason": (
                    "Highest official-test F1; "
                    "accuracy used as secondary criterion."
                ),
            }
        ]
    )

    decision.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "final_model_selection.csv",
        ),
        index=False,
    )

    print(f"\nSelected final model: {final_model}")

# 10. Final console summary
print("\n" + "=" * 70)
print("ROBUSTNESS REPORT COMPLETE")
print("=" * 70)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(f" - {filename}")

print("\nNo model was retrained.")
print("No official test data was modified.")
print("No threshold tuning was performed.")

print("\n" + "=" * 70)