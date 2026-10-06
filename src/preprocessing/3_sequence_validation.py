# PHASE 8: NITYMED CONTINUOUS SEQUENCE VALIDATION

# Purpose:
# This phase checks whether the annotated NITYMED frames contain
# continuous temporal sequences suitable for LSTM training.

# Dataset:
#   NITYMED

# Classes:
#   0 = Not Drowsy -> alert
#   1 = Drowsy     -> microsleep

# Sequence:
#   30 consecutive frames

# NITYMED videos are approximately 25 FPS, therefore:

#   30 frames / 25 FPS = 1.2 seconds
#
# A valid sequence requires:
#   1. Same video
#   2. Consecutive frame numbers
#   3. Same class label
#
# This gives us clean temporal sequences for the initial
# LSTM baseline.
#
# Output:
#   data/features/sequence_summary.csv
#   data/features/sequence_metadata.csv
# ============================================================


from pathlib import Path

import pandas as pd

# PHASE 8: NITYMED CONTINUOUS SEQUENCE VALIDATION

# 1. PATH CONFIGURATION

INPUT_FILE = Path(
    "data/features/annotation_metadata.csv"
)

SUMMARY_FILE = Path(
    "data/features/sequence_summary.csv"
)

SEQUENCE_FILE = Path(
    "data/features/sequence_metadata.csv"
)

# 2. SEQUENCE CONFIGURATION

SEQUENCE_LENGTH = 30
FPS = 25
SEQUENCE_DURATION = SEQUENCE_LENGTH / FPS

# 3. LOAD ANNOTATION METADATA

print("=" * 60)
print("PHASE 8: NITYMED CONTINUOUS SEQUENCE VALIDATION")
print("=" * 60)

print("\nLoading annotation metadata...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n\n"
        "Please run Phase 7 first."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Total annotated frames: {len(df):,}")

# 4. SORT DATA

df = df.sort_values(
    by=[
        "video_name",
        "frame_number"
    ]
).reset_index(drop=True)

# 5. FIND CONTINUOUS RUNS

df["previous_frame"] = (
    df.groupby("video_name")["frame_number"]
    .shift(1)
)

df["previous_label"] = (
    df.groupby("video_name")["label"]
    .shift(1)
)

df["new_run"] = (
    (df["frame_number"] != df["previous_frame"] + 1)
    | (df["label"] != df["previous_label"])
)

df["run_id"] = (
    df.groupby("video_name")["new_run"]
    .cumsum()
)

# 6. CALCULATE RUN STATISTICS

run_stats = (
    df.groupby(
        [
            "video_name",
            "run_id",
            "label"
        ]
    )
    .agg(
        start_frame=("frame_number", "min"),
        end_frame=("frame_number", "max"),
        run_length=("frame_number", "count")
    )
    .reset_index()
)

# 7. FIND RUNS CAPABLE OF 30-FRAME SEQUENCES

valid_runs = run_stats[
    run_stats["run_length"] >= SEQUENCE_LENGTH
].copy()

print("\n" + "=" * 60)
print("CONTINUOUS SEQUENCE RESULTS")
print("=" * 60)

print(
    f"\nSequence length       : "
    f"{SEQUENCE_LENGTH} frames"
)

print(
    f"FPS                   : "
    f"{FPS}"
)

print(
    f"Sequence duration     : "
    f"{SEQUENCE_DURATION:.2f} seconds"
)

print(
    f"\nTotal continuous runs : "
    f"{len(run_stats):,}"
)

print(
    f"Runs >= 30 frames    : "
    f"{len(valid_runs):,}"
)

# 8. CALCULATE NUMBER OF POSSIBLE WINDOWS

valid_runs["possible_sequences"] = (
    valid_runs["run_length"]
    - SEQUENCE_LENGTH
    + 1
)

total_sequences = int(
    valid_runs["possible_sequences"].sum()
)

print(
    f"Possible 30-frame sequences: "
    f"{total_sequences:,}"
)

# 9. CLASS-WISE SEQUENCE COUNT

print("\nSequence Distribution by Class:")
print("-" * 40)

class_sequence_stats = (
    valid_runs
    .groupby("label")
    .agg(
        valid_runs=("run_length", "count"),
        total_sequences=("possible_sequences", "sum"),
        longest_run=("run_length", "max")
    )
    .reset_index()
)

for _, row in class_sequence_stats.iterrows():
    label = int(row["label"])

    if label == 0:
        class_name = "Not Drowsy"
    else:
        class_name = "Drowsy"

    print(
        f"{class_name:12} -> "
        f"Runs: {int(row['valid_runs']):6,} | "
        f"Sequences: {int(row['total_sequences']):7,} | "
        f"Longest run: {int(row['longest_run']):5,}"
    )

# 10. CREATE SEQUENCE METADATA

sequence_records = []
sequence_id = 0

for _, row in valid_runs.iterrows():
    video_name = row["video_name"]
    label = int(row["label"])
    start_frame = int(row["start_frame"])
    end_frame = int(row["end_frame"])
    run_length = int(row["run_length"])

    max_start = (
        end_frame
        - SEQUENCE_LENGTH
        + 1
    )

    for sequence_start in range(
        start_frame,
        max_start + 1
    ):
        sequence_end = (
            sequence_start
            + SEQUENCE_LENGTH
            - 1
        )

        sequence_id += 1

        sequence_records.append({
            "sequence_id": sequence_id,
            "video_name": video_name,
            "start_frame": sequence_start,
            "end_frame": sequence_end,
            "label": label,
            "sequence_length": SEQUENCE_LENGTH,
            "fps": FPS,
            "duration_seconds": SEQUENCE_DURATION
        })

# 11. SAVE SEQUENCE METADATA

sequence_df = pd.DataFrame(sequence_records)

SEQUENCE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

sequence_df.to_csv(
    SEQUENCE_FILE,
    index=False
)

# 12. SAVE SUMMARY

video_summary = (
    valid_runs
    .groupby("video_name")
    .agg(
        valid_runs=("run_length", "count"),
        total_sequences=("possible_sequences", "sum"),
        longest_run=("run_length", "max")
    )
    .reset_index()
)

video_summary.to_csv(
    SUMMARY_FILE,
    index=False
)

# 13. VIDEO-LEVEL STATISTICS

videos_with_sequences = (
    valid_runs["video_name"]
    .nunique()
)

total_videos = (df["video_name"].nunique())

print(
    f"\nVideos with >=30-frame runs: "
    f"{videos_with_sequences:,}"
)

print(
    f"Total annotated videos      : "
    f"{total_videos:,}"
)

# 14. DISPLAY LONGEST RUNS

print("\nTop 20 Longest Continuous Runs:")
print("-" * 60)

top_runs = (
    valid_runs
    .sort_values(
        "run_length",
        ascending=False
    )
    .head(20)
)

print(
    top_runs[
        [
            "video_name",
            "label",
            "start_frame",
            "end_frame",
            "run_length",
            "possible_sequences"
        ]
    ].to_string(index=False)
)

# 15. FINAL OUTPUT

print("\n" + "=" * 60)
print("PHASE 8 COMPLETED")
print("=" * 60)

print(
    f"\nSequence metadata saved to:\n"
    f"{SEQUENCE_FILE}"
)

print(
    f"\nSequence summary saved to:\n"
    f"{SUMMARY_FILE}"
)

print("\nNext step:")

print(
    "If sufficient continuous sequences exist, "
    "we will start MediaPipe landmark extraction "
    "and calculate eye + head movement features."
)