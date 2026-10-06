# STEP 1: VIDEO DATASET INSPECTION

# Purpose:
# This phase inspects the video-based drowsiness dataset
# before extracting frames or building the ML model.
#
# For every video, we calculate:
#   - FPS
#   - Total number of frames
#   - Duration
#   - Video resolution
#   - Approximate number of 30-frame sequences
#
# IMPORTANT:
# We are NOT extracting frames in this phase.
#
# This step helps us understand whether the dataset contains
# enough genuine continuous temporal information for:
#
#       Video → Frames → Features → LSTM + Attention
#
# Output:
#       data/features/video_metadata.csv

import os
import cv2
import pandas as pd

# STEP 1: VIDEO DATASET INSPECTION

DATASET_PATH = "dataset/NITYMED"
OUTPUT_PATH = "data/features/video_metadata.csv"
SEQUENCE_LENGTH = 30

# Check dataset path.

if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset folder not found!")
    print(f"Expected path: {DATASET_PATH}")

    print("\nPlease make sure your videos are placed like:")
    print("dataset/video_dataset/male/")
    print("dataset/video_dataset/female/")

    raise SystemExit

# Supported video formats.

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv"
)

# Find all videos.

video_files = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(VIDEO_EXTENSIONS):
            video_files.append(
                os.path.join(root, file)
            )

video_files.sort()

print("\n" + "=" * 70)
print("VIDEO DATASET INSPECTION")
print("=" * 70)

print(f"\nDataset path: {DATASET_PATH}")
print(f"Total videos found: {len(video_files)}")

# Process each video.

records = []

for index, video_path in enumerate(video_files, start=1):
    print(
        f"\n[{index}/{len(video_files)}] "
        f"Processing: {os.path.basename(video_path)}"
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("  ERROR: Could not open video.")

        records.append({
            "video_id": index,
            "video_path": video_path,
            "filename": os.path.basename(video_path),
            "gender_folder": os.path.basename(
                os.path.dirname(video_path)
            ),
            "fps": None,
            "total_frames": None,
            "duration_seconds": None,
            "width": None,
            "height": None,
            "approx_30_frame_sequences": None,
            "status": "ERROR"
        })

        continue

    # Get video information.

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # Calculate duration.

    if fps > 0:
        duration = total_frames / fps
    else:
        duration = 0

    # Approximate number of 30-frame sequences.

    if total_frames >= SEQUENCE_LENGTH:
        num_sequences = (
            total_frames // SEQUENCE_LENGTH
        )
    else:
        num_sequences = 0

    # Get folder name.

    gender_folder = os.path.basename(
        os.path.dirname(video_path)
    )

    # Save record.

    records.append({
        "video_id": index,
        "video_path": video_path,
        "filename": os.path.basename(video_path),
        "gender_folder": gender_folder,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_seconds": round(duration, 2),
        "width": width,
        "height": height,
        "approx_30_frame_sequences": num_sequences,
        "status": "OK"
    })

    cap.release()

# Create dataframe.

metadata_df = pd.DataFrame(records)

# Create output directory.

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

# Save metadata.

metadata_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# Summary.

print("\n" + "=" * 70)
print("VIDEO DATASET INSPECTION COMPLETE")
print("=" * 70)

print(
    f"\nTotal videos       : "
    f"{len(metadata_df)}"
)

print(
    f"Successfully opened: "
    f"{(metadata_df['status'] == 'OK').sum()}"
)

print(
    f"Failed videos      : "
    f"{(metadata_df['status'] == 'ERROR').sum()}"
)

# Folder distribution.

if len(metadata_df) > 0:
    print("\nVideos by folder:")

    print(
        metadata_df["gender_folder"].value_counts()
    )

# Total frames.

valid_df = metadata_df[
    metadata_df["status"] == "OK"
]

if len(valid_df) > 0:
    print("\nTotal frames across videos:")

    print(
        f"{valid_df['total_frames'].sum():,}"
    )

    print("\nTotal approximate 30-frame sequences:")

    print(
        f"{valid_df['approx_30_frame_sequences'].sum():,}"
    )

    print("\nFPS distribution:")

    print(
        valid_df["fps"]
        .value_counts()
        .sort_index()
    )

    print("\nResolution distribution:")

    print(
        valid_df[
            ["width", "height"]
        ].value_counts()
    )

# Show video table.

print("\nVideo metadata:")

print(
    metadata_df[
        [
            "video_id",
            "filename",
            "gender_folder",
            "fps",
            "total_frames",
            "duration_seconds",
            "width",
            "height",
            "approx_30_frame_sequences",
            "status"
        ]
    ].to_string(index=False)
)

# Output.

print("\nMetadata saved to:")
print(OUTPUT_PATH)
print("=" * 70)