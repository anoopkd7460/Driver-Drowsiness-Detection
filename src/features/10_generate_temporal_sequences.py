import os
import numpy as np
import pandas as pd

INPUT_FILE = "data/features/modeling_frame_features.csv"
SPLIT_DIR = "data/splits"
OUTPUT_DIR = "data/sequences"

SEQUENCE_LENGTH = 30
STEP_SIZE = 30

FEATURES = [
    "ear_mean",
    "pitch",
    "yaw",
    "roll",
]


# Print section heading
def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_split(filename):
    path = os.path.join(SPLIT_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")

    return pd.read_csv(path)


def generate_sequences(df, split_videos, split_name):
    """
    Generate non-overlapping temporal sequences.

    IMPORTANT:
    - Sequences are generated only inside videos belonging
      to the specified split.
    - Missing frames are never interpolated.
    - A sequence must contain consecutive frame numbers.
    - All four features must be valid.
    - A sequence cannot cross a video boundary.
    """

    split_df = df[df["video_name"].isin(split_videos)].copy()
    split_df = split_df.sort_values(
        ["video_name", "frame_number"]
    ).reset_index(drop=True)

    sequences = []
    sequence_counter = 0
    videos_processed = 0
    videos_with_sequences = 0

    for video_name, video_df in split_df.groupby("video_name", sort=True):
        videos_processed += 1
        video_df = video_df.sort_values("frame_number").reset_index(drop=True)

        # Keep rows where all required features are available
        valid_mask = video_df[FEATURES].notna().all(axis=1)
        video_df = video_df.loc[valid_mask].reset_index(drop=True)

        if len(video_df) < SEQUENCE_LENGTH:
            continue

        # Find continuous frame runs
        frame_numbers = (
            video_df["frame_number"]
            .astype(int)
            .to_numpy()
        )

        run_start = 0

        for i in range(1, len(frame_numbers) + 1):
            end_of_run = (
                i == len(frame_numbers)
                or frame_numbers[i] != frame_numbers[i - 1] + 1
            )

            if not end_of_run:
                continue

            run_df = video_df.iloc[run_start:i].copy()
            run_length = len(run_df)

            # Generate non-overlapping 30-frame windows
            if run_length >= SEQUENCE_LENGTH:
                videos_with_sequences += 1
                start = 0

                while start + SEQUENCE_LENGTH <= run_length:
                    seq_df = run_df.iloc[start:start + SEQUENCE_LENGTH]

                    sequence_frames = (
                        seq_df["frame_number"]
                        .astype(int)
                        .to_numpy()
                    )

                    # Final safety check
                    if len(sequence_frames) != SEQUENCE_LENGTH:
                        start += STEP_SIZE
                        continue

                    expected_frames = np.arange(
                        sequence_frames[0],
                        sequence_frames[0] + SEQUENCE_LENGTH
                    )

                    if not np.array_equal(sequence_frames, expected_frames):
                        start += STEP_SIZE
                        continue

                    # Check sequence label
                    labels = seq_df["label"].astype(int).unique()

                    if len(labels) != 1:
                        start += STEP_SIZE
                        continue

                    label = int(labels[0])

                    sequence_id = (
                        f"{split_name}_{sequence_counter:06d}"
                    )

                    sequences.append({
                        "sequence_id": sequence_id,
                        "video_name": video_name,
                        "split": split_name,
                        "start_frame": int(sequence_frames[0]),
                        "end_frame": int(sequence_frames[-1]),
                        "num_frames": SEQUENCE_LENGTH,
                        "label": label,
                    })

                    # Store flattened temporal features
                    for feature in FEATURES:
                        values = seq_df[feature].to_numpy(dtype=np.float32)

                        for timestep, value in enumerate(values, start=1):
                            sequences[-1][
                                f"{feature}_t{timestep:02d}"
                            ] = float(value)

                    sequence_counter += 1
                    start += STEP_SIZE

            run_start = i

    result = pd.DataFrame(sequences)

    return result, videos_processed, videos_with_sequences


def print_sequence_summary(sequence_df, split_name):
    section(f"{split_name.upper()} SEQUENCE SUMMARY")

    if len(sequence_df) == 0:
        print("No sequences generated.")
        return

    print(f"Total sequences: {len(sequence_df):,}")
    print(f"Unique videos: {sequence_df['video_name'].nunique()}")

    print("\nClass distribution:")

    counts = (
        sequence_df["label"]
        .value_counts()
        .sort_index()
    )

    for label, count in counts.items():
        name = "Not Drowsy" if label == 0 else "Drowsy"
        percentage = (count / len(sequence_df)) * 100

        print(
            f"  {label} = {name:12s} "
            f"{count:,} ({percentage:.2f}%)"
        )

    print("\nSequence length:")
    print(
        sequence_df["num_frames"]
        .value_counts()
        .sort_index()
        .to_string()
    )


def main():
    section("PHASE 12 — TEMPORAL SEQUENCE GENERATION")

    # CONFIGURATION
    print("\nConfiguration:")
    print(f"  Sequence length : {SEQUENCE_LENGTH} frames")
    print("  FPS             : 25")
    print(f"  Time window     : {SEQUENCE_LENGTH / 25:.2f} seconds")
    print(f"  Step size       : {STEP_SIZE} frames")
    print("  Window type     : Non-overlapping")

    print("\nFeatures:")
    for feature in FEATURES:
        print(f"  - {feature}")

    # CHECK FILES
    if not os.path.exists(INPUT_FILE):
        print("\nERROR: Modeling dataset not found:")
        print(INPUT_FILE)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # LOAD MODELING DATA
    section("1. LOADING MODELING DATA")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Videos: {df['video_name'].nunique()}")

    # LOAD FIXED SPLITS
    section("2. LOADING FIXED VIDEO SPLITS")

    train_split = load_split("train_videos.csv")
    val_split = load_split("val_videos.csv")
    test_split = load_split("test_videos.csv")

    train_videos = set(train_split["video_name"])
    val_videos = set(val_split["video_name"])
    test_videos = set(test_split["video_name"])

    print(f"Train videos      : {len(train_videos)}")
    print(f"Validation videos : {len(val_videos)}")
    print(f"Test videos       : {len(test_videos)}")

    # SPLIT LEAKAGE CHECK
    section("3. PRE-SEQUENCE LEAKAGE CHECK")

    print(f"Train ∩ Validation : {len(train_videos & val_videos)}")
    print(f"Train ∩ Test       : {len(train_videos & test_videos)}")
    print(f"Validation ∩ Test  : {len(val_videos & test_videos)}")

    if (
        train_videos & val_videos
        or train_videos & test_videos
        or val_videos & test_videos
    ):
        print("\nERROR: Video leakage detected.")
        return

    print("\nNo video overlap detected.")

    # GENERATE TRAIN SEQUENCES
    section("4. GENERATING TRAIN SEQUENCES")

    train_sequences, train_processed, train_with_seq = generate_sequences(
        df, train_videos, "train"
    )

    print(f"Videos processed       : {train_processed}")
    print(f"Videos with sequences  : {train_with_seq}")
    print(f"Sequences generated    : {len(train_sequences):,}")

    # GENERATE VALIDATION SEQUENCES
    section("5. GENERATING VALIDATION SEQUENCES")

    val_sequences, val_processed, val_with_seq = generate_sequences(
        df, val_videos, "validation"
    )

    print(f"Videos processed       : {val_processed}")
    print(f"Videos with sequences  : {val_with_seq}")
    print(f"Sequences generated    : {len(val_sequences):,}")

    # GENERATE TEST SEQUENCES
    section("6. GENERATING TEST SEQUENCES")

    test_sequences, test_processed, test_with_seq = generate_sequences(
        df, test_videos, "test"
    )

    print(f"Videos processed       : {test_processed}")
    print(f"Videos with sequences  : {test_with_seq}")
    print(f"Sequences generated    : {len(test_sequences):,}")

    # PRINT SUMMARIES
    print_sequence_summary(train_sequences, "train")
    print_sequence_summary(val_sequences, "validation")
    print_sequence_summary(test_sequences, "test")

    # SAVE
    section("7. SAVING TEMPORAL SEQUENCES")

    train_file = os.path.join(OUTPUT_DIR, "train_sequences.csv")
    val_file = os.path.join(OUTPUT_DIR, "val_sequences.csv")
    test_file = os.path.join(OUTPUT_DIR, "test_sequences.csv")

    train_sequences.to_csv(train_file, index=False)
    val_sequences.to_csv(val_file, index=False)
    test_sequences.to_csv(test_file, index=False)

    print(f"Train sequences      : {train_file}")
    print(f"Validation sequences : {val_file}")
    print(f"Test sequences       : {test_file}")

    # SEQUENCE SUMMARY CSV
    section("8. SAVING SEQUENCE SUMMARY")

    summary_rows = []

    for split_name, sequence_df in [
        ("train", train_sequences),
        ("validation", val_sequences),
        ("test", test_sequences),
    ]:
        if len(sequence_df) == 0:
            summary_rows.append({
                "split": split_name,
                "sequences": 0,
                "videos": 0,
                "not_drowsy_sequences": 0,
                "drowsy_sequences": 0,
            })
            continue

        not_drowsy = (sequence_df["label"] == 0).sum()
        drowsy = (sequence_df["label"] == 1).sum()

        summary_rows.append({
            "split": split_name,
            "sequences": len(sequence_df),
            "videos": sequence_df["video_name"].nunique(),
            "not_drowsy_sequences": int(not_drowsy),
            "drowsy_sequences": int(drowsy),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(OUTPUT_DIR, "sequence_summary.csv")

    summary_df.to_csv(summary_file, index=False)

    print(summary_df.to_string(index=False))
    print("\nSummary saved:")
    print(summary_file)

    # FINAL LEAKAGE CHECK
    section("9. FINAL SEQUENCE LEAKAGE CHECK")

    train_sequence_videos = set(train_sequences["video_name"])
    val_sequence_videos = set(val_sequences["video_name"])
    test_sequence_videos = set(test_sequences["video_name"])

    print(
        "Train ∩ Validation:",
        len(train_sequence_videos & val_sequence_videos)
    )
    print(
        "Train ∩ Test:",
        len(train_sequence_videos & test_sequence_videos)
    )
    print(
        "Validation ∩ Test:",
        len(val_sequence_videos & test_sequence_videos)
    )

    # FINAL
    section("PHASE 12 COMPLETED")

    print("\nTemporal sequence generation completed.")

    print("\nSequence configuration:")
    print(f"  {SEQUENCE_LENGTH} frames × {len(FEATURES)} features")
    print(f"  Temporal duration: {SEQUENCE_LENGTH / 25:.2f} seconds")

    print("\nIMPORTANT:")
    print("Sequences were generated AFTER the video-wise split.")
    print("No missing frames were interpolated.")
    print("No sequence crosses a video boundary.")
    print("No mouth/yawning features are used.")

    print("\nNext step:")
    print(
        "Review sequence counts and class balance "
        "before building the LSTM baseline."
    )


if __name__ == "__main__":
    main()