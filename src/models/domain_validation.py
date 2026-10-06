from pathlib import Path
import pandas as pd
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_DIR = PROJECT_ROOT / "data" / "sequences"
OUTPUT_DIR = PROJECT_ROOT / "results" / "domain_validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SEQUENCE_DIR / "train_sequences.csv"
VAL_FILE = SEQUENCE_DIR / "val_sequences.csv"
TEST_FILE = SEQUENCE_DIR / "test_sequences.csv"

# Load existing sequences
print("=" * 70)
print("DOMAIN LABEL VALIDATION")
print("=" * 70)
print("\nLoading existing temporal sequence files...")

files = {
    "train": TRAIN_FILE,
    "val": VAL_FILE,
    "test": TEST_FILE,
}

for split, path in files.items():
    if not path.exists():
        raise FileNotFoundError(f"{split} sequence file not found:\n{path}")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"Train sequences : {len(train_df):,}")
print(f"Val sequences   : {len(val_df):,}")
print(f"Test sequences  : {len(test_df):,}")
print(f"Total sequences : {len(df):,}")

# Required columns
required_columns = ["sequence_id", "video_name", "split", "label"]

missing = [column for column in required_columns if column not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Basic summary
print("\n" + "-" * 70)
print("DATASET SUMMARY")
print("-" * 70)

print(f"Unique videos   : {df['video_name'].nunique()}")
print(f"Total sequences : {len(df):,}")
print("\nClass distribution:")

class_counts = df["label"].value_counts().sort_index()

for label, count in class_counts.items():
    name = "Not Drowsy" if label == 0 else "Drowsy"
    percentage = count / len(df) * 100
    print(f"  {name:<15}: {count:>6,} ({percentage:6.2f}%)")

# Video-level domain summary
print("\n" + "-" * 70)
print("VIDEO-LEVEL DOMAIN SUMMARY")
print("-" * 70)

video_summary = (
    df.groupby("video_name")
    .agg(
        total_sequences=("sequence_id", "count"),
        not_drowsy_sequences=("label", lambda x: int((x == 0).sum())),
        drowsy_sequences=("label", lambda x: int((x == 1).sum())),
        splits=("split", lambda x: ",".join(sorted(x.unique()))),
    )
    .reset_index()
)

video_summary["drowsy_ratio"] = (
    video_summary["drowsy_sequences"] / video_summary["total_sequences"]
)
video_summary["drowsy_ratio_pct"] = video_summary["drowsy_ratio"] * 100

video_summary = video_summary.sort_values("total_sequences", ascending=False)

print(
    video_summary[
        [
            "video_name",
            "total_sequences",
            "not_drowsy_sequences",
            "drowsy_sequences",
            "drowsy_ratio_pct",
            "splits",
        ]
    ].to_string(index=False)
)

# Domain sample size
print("\n" + "-" * 70)
print("VIDEO SAMPLE SIZE")
print("-" * 70)

sequence_thresholds = [5, 10, 20, 30, 50]

for threshold in sequence_thresholds:
    count = (video_summary["total_sequences"] >= threshold).sum()
    print(f"Videos with >= {threshold:2d} sequences: {count}")

# Drowsy sample size
print("\n" + "-" * 70)
print("DROWSY SAMPLE SIZE PER VIDEO")
print("-" * 70)

drowsy_thresholds = [1, 3, 5, 10]

for threshold in drowsy_thresholds:
    count = (video_summary["drowsy_sequences"] >= threshold).sum()
    print(f"Videos with >= {threshold:2d} drowsy sequences: {count}")

# Class composition
print("\n" + "-" * 70)
print("CLASS COMPOSITION ACROSS VIDEOS")
print("-" * 70)

only_not_drowsy = video_summary[video_summary["drowsy_sequences"] == 0]
only_drowsy = video_summary[video_summary["not_drowsy_sequences"] == 0]
mixed = video_summary[
    (video_summary["drowsy_sequences"] > 0)
    & (video_summary["not_drowsy_sequences"] > 0)
]

print(f"Only Not Drowsy : {len(only_not_drowsy)} videos")
print(f"Only Drowsy     : {len(only_drowsy)} videos")
print(f"Both classes    : {len(mixed)} videos")

# Split-wise video distribution
print("\n" + "-" * 70)
print("SPLIT-WISE VIDEO DISTRIBUTION")
print("-" * 70)

split_summary = (
    df.groupby(["split", "video_name"])
    .agg(
        sequences=("sequence_id", "count"),
        drowsy=("label", lambda x: int((x == 1).sum())),
        not_drowsy=("label", lambda x: int((x == 0).sum())),
    )
    .reset_index()
)

split_summary["drowsy_ratio_pct"] = (
    split_summary["drowsy"] / split_summary["sequences"] * 100
)

print(split_summary.to_string(index=False))

# Video size statistics
print("\n" + "-" * 70)
print("VIDEO SEQUENCE COUNT STATISTICS")
print("-" * 70)

sizes = video_summary["total_sequences"]

print(f"Mean   : {sizes.mean():.2f}")
print(f"Median : {sizes.median():.2f}")
print(f"Std    : {sizes.std():.2f}")
print(f"Min    : {sizes.min()}")
print(f"Max    : {sizes.max()}")

if sizes.min() > 0:
    print(f"Max/Min ratio : {sizes.max() / sizes.min():.2f}")

# Drowsy ratio statistics
print("\n" + "-" * 70)
print("DROWSY RATIO VARIABILITY")
print("-" * 70)

ratios = video_summary["drowsy_ratio_pct"]

print(f"Mean   : {ratios.mean():.2f}%")
print(f"Median : {ratios.median():.2f}%")
print(f"Std    : {ratios.std():.2f}%")
print(f"Min    : {ratios.min():.2f}%")
print(f"Max    : {ratios.max():.2f}%")

# Check video-split leakage
print("\n" + "-" * 70)
print("VIDEO-SPLIT LEAKAGE CHECK")
print("-" * 70)

video_split_counts = df.groupby("video_name")["split"].nunique()
leaked_videos = video_split_counts[video_split_counts > 1]

print(f"Videos appearing in multiple splits: {len(leaked_videos)}")

if len(leaked_videos) > 0:
    print("\nLEAKED VIDEOS:")
    for video in leaked_videos.index:
        print(
            f"  {video}: "
            f"{df[df['video_name'] == video]['split'].unique()}"
        )
else:
    print("No video leakage detected.")

# Save results
video_output = OUTPUT_DIR / "video_domain_summary.csv"
split_output = OUTPUT_DIR / "split_domain_summary.csv"

video_summary.to_csv(video_output, index=False)
split_summary.to_csv(split_output, index=False)

# Final message
print("\n" + "=" * 70)
print("DOMAIN VALIDATION COMPLETED")
print("=" * 70)

print("\nSaved files:")
print(f"  {video_output}")
print(f"  {split_output}")

print(
    "\nNext decision: determine whether video/recording identity "
    "is a scientifically defensible domain label for GRL."
)