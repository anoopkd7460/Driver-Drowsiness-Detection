import os
import pandas as pd
import numpy as np

# PHASE 17 — VIDEO-WISE ERROR ANALYSIS

RESULTS_DIR = "results/phase16_generalization"

VIDEO_RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "video_wise_results.csv"
)

EXPERIMENT_RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "repeated_video_level_results.csv"
)

OUTPUT_DIR = "results/phase17_error_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 17 — VIDEO-WISE ERROR ANALYSIS")
print("=" * 70)

# 1. LOAD RESULTS

print("\n" + "=" * 70)
print("1. LOADING PHASE 16 RESULTS")
print("=" * 70)

video_df = pd.read_csv(VIDEO_RESULTS_FILE)
experiment_df = pd.read_csv(EXPERIMENT_RESULTS_FILE)

print("\nVideo-wise rows :", len(video_df))
print("Experiment rows :", len(experiment_df))

print("\nVideo result columns:")
print(list(video_df.columns))

# 2. BASIC VIDEO ANALYSIS

print("\n" + "=" * 70)
print("2. VIDEO-WISE PERFORMANCE")
print("=" * 70)

video_summary = (
    video_df
    .groupby("video_name")
    .agg(
        evaluations=("seed", "count"),
        total_sequences=("sequences", "sum"),
        total_drowsy_sequences=("drowsy_sequences", "sum"),
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        min_accuracy=("accuracy", "min"),
        max_accuracy=("accuracy", "max"),
        mean_drowsy_recall=("recall_drowsy", "mean"),
    )
    .reset_index()
)

video_summary = video_summary.sort_values("mean_accuracy")

print("\nVideos evaluated:", len(video_summary))
print("\nLowest-performing videos:")
print(video_summary.head(10).to_string(index=False))

# 3. SAVE VIDEO SUMMARY

video_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "video_performance_summary.csv"
    ),
    index=False
)

# 4. DIFFICULT VIDEOS

print("\n" + "=" * 70)
print("3. DIFFICULT VIDEO IDENTIFICATION")
print("=" * 70)

difficult_videos = video_summary[
    video_summary["mean_accuracy"] < 0.95
].copy()

print(
    "\nVideos with mean accuracy < 95%:",
    len(difficult_videos)
)

if len(difficult_videos) > 0:
    print(difficult_videos.to_string(index=False))
else:
    print("No video has mean accuracy below 95%.")

difficult_videos.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "difficult_videos.csv"
    ),
    index=False
)

# 5. DROWSY VIDEO ANALYSIS

print("\n" + "=" * 70)
print("4. DROWSY-VIDEO ANALYSIS")
print("=" * 70)

drowsy_videos = video_summary[
    video_summary["total_drowsy_sequences"] > 0
].copy()

drowsy_videos = drowsy_videos.sort_values(
    "mean_drowsy_recall"
)

print(
    "\nVideos containing drowsy sequences:",
    len(drowsy_videos)
)

print("\nDrowsy recall by video:")

print(
    drowsy_videos[
        [
            "video_name",
            "evaluations",
            "total_sequences",
            "total_drowsy_sequences",
            "mean_accuracy",
            "mean_drowsy_recall",
        ]
    ].to_string(index=False)
)

drowsy_videos.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "drowsy_video_analysis.csv"
    ),
    index=False
)

# 6. EXPERIMENT-LEVEL VARIABILITY

print("\n" + "=" * 70)
print("5. EXPERIMENT-LEVEL VARIABILITY")
print("=" * 70)

print("\nResults for every seed:")

print(
    experiment_df[
        [
            "seed",
            "train_videos",
            "validation_videos",
            "train_sequences",
            "validation_sequences",
            "train_drowsy",
            "validation_drowsy",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "fp",
            "fn",
        ]
    ].to_string(index=False)
)

# 7. FIND WORST EXPERIMENT

worst_idx = experiment_df["accuracy"].idxmin()
worst_experiment = experiment_df.loc[worst_idx]

print("\nWorst experiment:")

print(f"  Seed     : {worst_experiment['seed']}")
print(
    f"  Accuracy : "
    f"{worst_experiment['accuracy'] * 100:.2f}%"
)
print(
    f"  Precision: "
    f"{worst_experiment['precision'] * 100:.2f}%"
)
print(
    f"  Recall   : "
    f"{worst_experiment['recall'] * 100:.2f}%"
)
print(
    f"  F1       : "
    f"{worst_experiment['f1'] * 100:.2f}%"
)

# 8. WORST EXPERIMENT VIDEOS

worst_seed = worst_experiment["seed"]

worst_seed_videos = video_df[
    video_df["seed"] == worst_seed
].copy()

worst_seed_videos = worst_seed_videos.sort_values(
    "accuracy"
)

print("\nVideos used in worst experiment:")

print(
    worst_seed_videos[
        [
            "video_name",
            "sequences",
            "drowsy_sequences",
            "accuracy",
            "recall_drowsy",
        ]
    ].to_string(index=False)
)

worst_seed_videos.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "worst_experiment_videos.csv"
    ),
    index=False
)

# 9. IDENTIFY CONSISTENTLY DIFFICULT VIDEOS

print("\n" + "=" * 70)
print("6. CONSISTENTLY DIFFICULT VIDEOS")
print("=" * 70)

consistent_difficult = video_summary[
    (video_summary["evaluations"] >= 2)
    & (video_summary["mean_accuracy"] < 0.98)
].copy()

consistent_difficult = consistent_difficult.sort_values(
    "mean_accuracy"
)

print(
    "\nVideos evaluated multiple times "
    "with mean accuracy < 98%:"
)

if len(consistent_difficult) > 0:
    print(
        consistent_difficult.to_string(
            index=False
        )
    )
else:
    print("No consistently difficult video identified.")

consistent_difficult.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "consistently_difficult_videos.csv"
    ),
    index=False
)

# 10. CLASS IMBALANCE CHECK

print("\n" + "=" * 70)
print("7. VALIDATION CLASS DISTRIBUTION")
print("=" * 70)

experiment_df["drowsy_ratio"] = (
    experiment_df["validation_drowsy"]
    / experiment_df["validation_sequences"]
)

print(
    experiment_df[
        [
            "seed",
            "validation_sequences",
            "validation_drowsy",
            "drowsy_ratio",
            "accuracy",
            "recall",
            "f1",
        ]
    ].to_string(index=False)
)

experiment_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "experiment_analysis.csv"
    ),
    index=False
)

# 11. OVERALL INTERPRETATION

print("\n" + "=" * 70)
print("8. PHASE 17 INTERPRETATION")
print("=" * 70)

mean_accuracy = experiment_df["accuracy"].mean()
min_accuracy = experiment_df["accuracy"].min()
mean_recall = experiment_df["recall"].mean()

print(
    f"\nMean accuracy       : "
    f"{mean_accuracy * 100:.2f}%"
)

print(
    f"Minimum accuracy    : "
    f"{min_accuracy * 100:.2f}%"
)

print(
    f"Mean drowsy recall  : "
    f"{mean_recall * 100:.2f}%"
)

if min_accuracy >= 0.95:
    print(
        "\nAll repeated video-level "
        "experiments remained at or above "
        "95% accuracy."
    )
else:
    print(
        "\nAt least one video-level experiment "
        "fell below 95% accuracy."
    )

print("\nNo retraining was performed.")
print("No existing test set was modified.")
print("No mouth/yawning features were used.")

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\n" + "=" * 70)
print("PHASE 17 COMPLETED")
print("=" * 70)