from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_DIR = PROJECT_ROOT / "data" / "sequences"
OUTPUT_DIR = PROJECT_ROOT / "results" / "recording_shift"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SEQUENCE_DIR / "train_sequences.csv"
VAL_FILE = SEQUENCE_DIR / "val_sequences.csv"
TEST_FILE = SEQUENCE_DIR / "test_sequences.csv"

# CONFIGURATION
FEATURES = ["ear_mean", "pitch", "yaw", "roll"]
SEQUENCE_LENGTH = 30

# FEATURE COLUMN GENERATION
feature_columns = []

for feature in FEATURES:
    for t in range(1, SEQUENCE_LENGTH + 1):
        feature_columns.append(f"{feature}_t{t:02d}")

# LOAD DATA
print("=" * 70)
print("RECORDING DISTRIBUTION SHIFT ANALYSIS")
print("=" * 70)

print("\nLoading sequence data...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True,
)

print(f"Total sequences : {len(df):,}")
print(f"Unique videos   : {df['video_name'].nunique()}")

# CONVERT SEQUENCES TO FRAME-LIKE VALUES
print("\nPreparing feature distributions...")

records = []

for _, row in df.iterrows():
    video = row["video_name"]
    label = int(row["label"])
    split = row["split"]

    for feature in FEATURES:
        values = []

        for t in range(1, SEQUENCE_LENGTH + 1):
            column = f"{feature}_t{t:02d}"

            if column in row.index:
                value = row[column]

                if pd.notna(value):
                    values.append(float(value))

        if len(values) > 0:
            for value in values:
                records.append(
                    {
                        "video_name": video,
                        "label": label,
                        "split": split,
                        "feature": feature,
                        "value": value,
                    }
                )

long_df = pd.DataFrame(records)

print(
    f"Valid feature observations: "
    f"{len(long_df):,}"
)

# VIDEO-LEVEL SHIFT ANALYSIS
print("\n" + "-" * 70)
print("CALCULATING VIDEO-LEVEL DISTRIBUTION SHIFT")
print("-" * 70)

videos = sorted(long_df["video_name"].unique())
results = []

for video in videos:
    video_df = long_df[
        long_df["video_name"] == video
    ]

    other_df = long_df[
        long_df["video_name"] != video
    ]

    for feature in FEATURES:
        target = video_df[
            video_df["feature"] == feature
        ]["value"].dropna().values

        reference = other_df[
            other_df["feature"] == feature
        ]["value"].dropna().values

        if len(target) < 2 or len(reference) < 2:
            continue

        # KS TEST
        ks_stat, ks_p = ks_2samp(
            target,
            reference,
        )

        # STANDARDIZED MEAN DIFFERENCE
        target_mean = np.mean(target)
        reference_mean = np.mean(reference)

        target_std = np.std(
            target,
            ddof=1,
        )

        reference_std = np.std(
            reference,
            ddof=1,
        )

        pooled_std = np.sqrt(
            (target_std ** 2 + reference_std ** 2) / 2
        )

        if pooled_std > 0:
            effect_size = (
                target_mean - reference_mean
            ) / pooled_std
        else:
            effect_size = 0.0

        results.append(
            {
                "video_name": video,
                "feature": feature,
                "video_samples": len(target),
                "reference_samples": len(reference),
                "video_mean": target_mean,
                "reference_mean": reference_mean,
                "video_std": target_std,
                "reference_std": reference_std,
                "mean_difference": (
                    target_mean - reference_mean
                ),
                "standardized_effect": effect_size,
                "absolute_effect": abs(effect_size),
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
            }
        )

shift_df = pd.DataFrame(results)

# OVERALL VIDEO SHIFT SCORE
video_scores = (
    shift_df
    .groupby("video_name")
    .agg(
        mean_absolute_effect=(
            "absolute_effect",
            "mean",
        ),
        max_absolute_effect=(
            "absolute_effect",
            "max",
        ),
        mean_ks_statistic=(
            "ks_statistic",
            "mean",
        ),
        max_ks_statistic=(
            "ks_statistic",
            "max",
        ),
        significant_features=(
            "ks_pvalue",
            lambda x: int((x < 0.05).sum()),
        ),
    )
    .reset_index()
)

# ADD SEQUENCE INFORMATION
sequence_summary = (
    df
    .groupby("video_name")
    .agg(
        total_sequences=(
            "sequence_id",
            "count",
        ),
        drowsy_sequences=(
            "label",
            lambda x: int((x == 1).sum()),
        ),
        not_drowsy_sequences=(
            "label",
            lambda x: int((x == 0).sum()),
        ),
    )
    .reset_index()
)

video_scores = video_scores.merge(
    sequence_summary,
    on="video_name",
    how="left",
)

# RANK VIDEOS
video_scores = (
    video_scores
    .sort_values(
        "mean_absolute_effect",
        ascending=False,
    )
    .reset_index(drop=True)
)

video_scores["shift_rank"] = (
    np.arange(len(video_scores)) + 1
)

# PRINT OVERALL RESULTS
print("\n" + "-" * 70)
print("VIDEO SHIFT RANKING")
print("-" * 70)

print(
    video_scores[
        [
            "shift_rank",
            "video_name",
            "total_sequences",
            "drowsy_sequences",
            "mean_absolute_effect",
            "max_absolute_effect",
            "mean_ks_statistic",
            "significant_features",
        ]
    ].to_string(index=False)
)

# FEATURE-WISE SHIFT
print("\n" + "-" * 70)
print("FEATURE-WISE SHIFT")
print("-" * 70)

feature_summary = (
    shift_df
    .groupby("feature")
    .agg(
        mean_absolute_effect=(
            "absolute_effect",
            "mean",
        ),
        max_absolute_effect=(
            "absolute_effect",
            "max",
        ),
        mean_ks_statistic=(
            "ks_statistic",
            "mean",
        ),
        significant_video_count=(
            "ks_pvalue",
            lambda x: int((x < 0.05).sum()),
        ),
    )
    .reset_index()
)

feature_summary = feature_summary.sort_values(
    "mean_absolute_effect",
    ascending=False,
)

print(
    feature_summary.to_string(index=False)
)

# STRONGEST SHIFTS
print("\n" + "-" * 70)
print("STRONGEST VIDEO-FEATURE SHIFTS")
print("-" * 70)

strongest = (
    shift_df
    .sort_values(
        "absolute_effect",
        ascending=False,
    )
    .head(15)
)

print(
    strongest[
        [
            "video_name",
            "feature",
            "video_mean",
            "reference_mean",
            "standardized_effect",
            "ks_statistic",
            "ks_pvalue",
        ]
    ].to_string(index=False)
)

# P1042762 COMPARISON
target_video = "P1042762_720"

if target_video in video_scores["video_name"].values:
    print("\n" + "-" * 70)
    print(f"TARGET VIDEO: {target_video}")
    print("-" * 70)

    target_result = video_scores[
        video_scores["video_name"] == target_video
    ]

    print(
        target_result.to_string(index=False)
    )

# SAVE RESULTS
shift_output = (
    OUTPUT_DIR / "video_feature_shift.csv"
)

ranking_output = (
    OUTPUT_DIR / "video_shift_ranking.csv"
)

feature_output = (
    OUTPUT_DIR / "feature_shift_summary.csv"
)

strongest_output = (
    OUTPUT_DIR / "strongest_video_feature_shifts.csv"
)

shift_df.to_csv(
    shift_output,
    index=False,
)

video_scores.to_csv(
    ranking_output,
    index=False,
)

feature_summary.to_csv(
    feature_output,
    index=False,
)

strongest.to_csv(
    strongest_output,
    index=False,
)

# FINAL SUMMARY
print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print("\nSaved files:")
print(f"  {shift_output}")
print(f"  {ranking_output}")
print(f"  {feature_output}")
print(f"  {strongest_output}")

print(
    "\nThis analysis measures recording/video-specific "
    "distribution shift without creating artificial domain labels."
)