# ============================================================
# PHASE 7: NITYMED ANNOTATION ANALYSIS & METADATA CREATION
# ============================================================
# Purpose:
# This phase reads the NITYMED annotation JSON file and converts
# frame-level annotations into a clean CSV dataset.
#
# Our project has only 2 classes:
#   0 = Not Drowsy  -> alert
#   1 = Drowsy      -> microsleep
#
# Yawning is intentionally ignored because our research project
# uses only eye movement + head movement and does not use
# mouth-related features.
#
# Output:
#   data/features/annotation_metadata.csv
#
# CSV columns:
#   video_name
#   frame_number
#   driver_state
#   label
#   image_path
# ============================================================

import json
import re
from pathlib import Path

import pandas as pd

# PHASE 7: NITYMED ANNOTATION ANALYSIS & METADATA CREATION

# 1. PATH CONFIGURATION

ANNOTATION_FILE = Path(
    "dataset/NITYMED/classification_frames/annotations_all.json"
)

OUTPUT_FILE = Path(
    "data/features/annotation_metadata.csv"
)

# 2. LABEL MAPPING

LABEL_MAP = {
    "alert": 0,
    "microsleep": 1
}

IGNORED_STATES = {
    "yawning"
}

# 3. LOAD JSON

print("=" * 60)
print("PHASE 7: NITYMED ANNOTATION ANALYSIS")
print("=" * 60)

print("\nLoading annotation file...")

if not ANNOTATION_FILE.exists():
    raise FileNotFoundError(
        f"\nAnnotation file not found:\n{ANNOTATION_FILE}\n\n"
        "Please put your JSON file at:\n"
        "dataset/NITYMED/annotations/annotations.json"
    )

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    annotations = json.load(f)

print(f"Total annotations in JSON: {len(annotations):,}")

# 4. PARSE ANNOTATION KEYS

records = []

pattern = re.compile(
    r"classification_frames[\\/]+(.+?)[\\/]+frame(\d+)\.jpg$",
    re.IGNORECASE
)

for image_path, annotation in annotations.items():
    match = pattern.search(image_path)

    if match is None:
        continue

    video_name = match.group(1)
    frame_number = int(match.group(2))
    driver_state = annotation.get("driver_state")

    if driver_state is None:
        continue

    driver_state = driver_state.lower().strip()

    if driver_state in IGNORED_STATES:
        continue

    if driver_state not in LABEL_MAP:
        continue

    label = LABEL_MAP[driver_state]

    records.append({
        "video_name": video_name,
        "frame_number": frame_number,
        "driver_state": driver_state,
        "label": label,
        "image_path": image_path
    })

# 5. CREATE DATAFRAME

df = pd.DataFrame(records)

if df.empty:
    raise ValueError(
        "No valid alert/microsleep annotations were found."
    )

# 6. REMOVE DUPLICATES

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=[
        "video_name",
        "frame_number"
    ]
)

after_duplicates = len(df)
duplicates_removed = before_duplicates - after_duplicates

# 7. SORT DATA

df = df.sort_values(
    by=[
        "video_name",
        "frame_number"
    ]
).reset_index(drop=True)

# 8. SAVE CSV

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# 9. DISPLAY BASIC STATISTICS

print("\n" + "=" * 60)
print("ANNOTATION SUMMARY")
print("=" * 60)

print(f"Valid alert/microsleep frames : {len(df):,}")

print(
    f"Unique videos                 : "
    f"{df['video_name'].nunique():,}"
)

print(
    f"Duplicates removed            : "
    f"{duplicates_removed:,}"
)

# 10. CLASS DISTRIBUTION

print("\nClass Distribution:")
print("-" * 40)

class_counts = df["driver_state"].value_counts()

for state, count in class_counts.items():
    label = LABEL_MAP[state]
    percentage = count / len(df) * 100

    print(
        f"{state:12} -> "
        f"Label {label} -> "
        f"{count:7,} frames "
        f"({percentage:.2f}%)"
    )

# 11. VIDEO DISTRIBUTION

print("\nAnnotations per Video:")
print("-" * 40)

video_counts = (
    df.groupby("video_name")
    .size()
    .sort_values(ascending=False)
)

print(video_counts.to_string())

# 12. FRAME RANGE PER VIDEO

print("\nFrame Range per Video:")
print("-" * 40)

video_frame_stats = (
    df.groupby("video_name")["frame_number"]
    .agg(
        min_frame="min",
        max_frame="max",
        annotated_frames="count"
    )
    .reset_index()
)

print(video_frame_stats.to_string(index=False))

# 13. FINAL OUTPUT

print("\n" + "=" * 60)
print("PHASE 7 COMPLETED")
print("=" * 60)

print(
    f"\nAnnotation metadata saved to:\n"
    f"{OUTPUT_FILE}"
)

print("\nNext step:")
print(
    "We will analyze whether these annotations contain "
    "continuous 30-frame sequences suitable for LSTM."
)