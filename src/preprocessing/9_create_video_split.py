import os
import random
import pandas as pd
import numpy as np

INPUT_FILE = "data/features/modeling_frame_features.csv"
OUTPUT_DIR = "data/splits"

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_videos.csv")
VAL_FILE = os.path.join(OUTPUT_DIR, "val_videos.csv")
TEST_FILE = os.path.join(OUTPUT_DIR, "test_videos.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "split_summary.csv")

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEQUENCE_LENGTH = 30

FEATURES = [
    "ear_mean",
    "pitch",
    "yaw",
    "roll",
]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def get_continuous_runs(video_df):
    """Find continuous runs where frame numbers increase by exactly 1."""

    video_df = video_df.sort_values(
        "frame_number"
    ).reset_index(drop=True)

    if len(video_df) == 0:
        return []

    frames = (
        video_df["frame_number"]
        .astype(int)
        .to_numpy()
    )

    runs = []
    start = 0

    for i in range(1, len(frames) + 1):
        end_of_run = (
            i == len(frames)
            or frames[i] != frames[i - 1] + 1
        )

        if end_of_run:
            run = video_df.iloc[start:i].copy()
            runs.append(run)
            start = i

    return runs


def count_usable_sequences(video_df):
    """
    Count non-overlapping 30-frame sequences.

    A sequence is usable only when:
    - all 4 selected features are valid
    - frames are consecutive
    - all 30 labels are identical
    """

    valid_df = video_df[
        video_df[FEATURES]
        .notna()
        .all(axis=1)
    ].copy()

    if len(valid_df) < SEQUENCE_LENGTH:
        return 0, 0, 0

    runs = get_continuous_runs(valid_df)

    total_sequences = 0
    not_drowsy_sequences = 0
    drowsy_sequences = 0

    for run in runs:
        run_length = len(run)

        if run_length < SEQUENCE_LENGTH:
            continue

        for start in range(
            0,
            run_length - SEQUENCE_LENGTH + 1,
            SEQUENCE_LENGTH
        ):
            sequence = run.iloc[
                start:start + SEQUENCE_LENGTH
            ]

            labels = (
                sequence["label"]
                .astype(int)
                .unique()
            )

            if len(labels) != 1:
                continue

            label = int(labels[0])
            total_sequences += 1

            if label == 0:
                not_drowsy_sequences += 1
            elif label == 1:
                drowsy_sequences += 1

    return (
        total_sequences,
        not_drowsy_sequences,
        drowsy_sequences
    )


def get_video_information(df):
    video_info = []

    for video_name, group in df.groupby(
        "video_name",
        sort=True
    ):
        total_frames = len(group)

        not_drowsy_frames = int(
            (group["label"] == 0).sum()
        )

        drowsy_frames = int(
            (group["label"] == 1).sum()
        )

        (
            usable_sequences,
            not_drowsy_sequences,
            drowsy_sequences
        ) = count_usable_sequences(group)

        video_info.append({
            "video_name": video_name,
            "total_frames": total_frames,
            "not_drowsy_frames": not_drowsy_frames,
            "drowsy_frames": drowsy_frames,
            "usable_sequences": usable_sequences,
            "not_drowsy_sequences": not_drowsy_sequences,
            "drowsy_sequences": drowsy_sequences,
            "has_drowsy": drowsy_sequences > 0,
            "has_usable_sequences": usable_sequences > 0,
        })

    return pd.DataFrame(video_info)


def choose_split_counts(n):
    """
    Determine train/validation/test counts.

    For drowsy-containing videos we need enough videos
    for validation and test.
    """

    if n < 3:
        raise ValueError("At least 3 videos are required.")

    train_n = int(round(n * TRAIN_RATIO))
    val_n = int(round(n * VAL_RATIO))

    train_n = max(train_n, 1)
    val_n = max(val_n, 1)

    test_n = n - train_n - val_n

    if test_n < 1:
        test_n = 1
        train_n = n - val_n - test_n

    return train_n, val_n, test_n


def stratified_sequence_aware_split(video_info):
    random.seed(RANDOM_SEED)

    drowsy_videos = video_info[
        video_info["drowsy_sequences"] > 0
    ].copy()

    non_drowsy_videos = video_info[
        video_info["drowsy_sequences"] == 0
    ].copy()

    print(
        f"\nUsable drowsy videos: "
        f"{len(drowsy_videos)}"
    )

    print(
        f"Usable non-drowsy-only videos: "
        f"{len(non_drowsy_videos)}"
    )

    # Sort drowsy videos by usable drowsy sequences.
    drowsy_videos = drowsy_videos.sort_values(
        "drowsy_sequences",
        ascending=False
    ).reset_index(drop=True)

    # Deterministic shuffle before distribution.
    drowsy_records = drowsy_videos.to_dict("records")
    random.shuffle(drowsy_records)

    drowsy_records = sorted(
        drowsy_records,
        key=lambda x: (
            -x["drowsy_sequences"],
            x["video_name"]
        )
    )

    d_train_n, d_val_n, d_test_n = choose_split_counts(
        len(drowsy_records)
    )

    d_train = []
    d_val = []
    d_test = []

    # Greedy balancing by drowsy sequence count.
    split_buckets = {
        "train": d_train,
        "validation": d_val,
        "test": d_test
    }

    target_counts = {
        "train": d_train_n,
        "validation": d_val_n,
        "test": d_test_n
    }

    current_drowsy_sequences = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    # Start by guaranteeing one drowsy video in validation/test.
    ordered_splits = [
        "validation",
        "test",
        "train"
    ]

    used = 0

    for split_name in ordered_splits:
        if used >= len(drowsy_records):
            break

        record = drowsy_records[used]

        split_buckets[split_name].append(
            record["video_name"]
        )

        current_drowsy_sequences[split_name] += (
            record["drowsy_sequences"]
        )

        used += 1

    remaining = drowsy_records[used:]

    # Assign remaining videos to the split with the
    # lowest normalized drowsy sequence load.
    for record in remaining:
        available_splits = [
            split_name
            for split_name in [
                "train",
                "validation",
                "test"
            ]
            if len(split_buckets[split_name]) <
            target_counts[split_name]
        ]

        selected_split = min(
            available_splits,
            key=lambda split_name: (
                current_drowsy_sequences[split_name]
                / max(
                    len(split_buckets[split_name]),
                    1
                )
            )
        )

        split_buckets[selected_split].append(
            record["video_name"]
        )

        current_drowsy_sequences[selected_split] += (
            record["drowsy_sequences"]
        )

    # Distribute non-drowsy-only videos according to ratios.
    non_drowsy_names = (
        non_drowsy_videos["video_name"]
        .tolist()
    )

    random.shuffle(non_drowsy_names)

    n_train, n_val, n_test = choose_split_counts(
        len(non_drowsy_names)
    )

    n_train_names = non_drowsy_names[:n_train]

    n_val_names = non_drowsy_names[
        n_train:n_train + n_val
    ]

    n_test_names = non_drowsy_names[
        n_train + n_val:
    ]

    train_videos = (
        split_buckets["train"]
        + n_train_names
    )

    val_videos = (
        split_buckets["validation"]
        + n_val_names
    )

    test_videos = (
        split_buckets["test"]
        + n_test_names
    )

    random.shuffle(train_videos)
    random.shuffle(val_videos)
    random.shuffle(test_videos)

    return (
        train_videos,
        val_videos,
        test_videos
    )


def create_split_dataframe(
    video_info,
    video_list,
    split_name
):
    split_df = video_info[
        video_info["video_name"].isin(video_list)
    ].copy()

    split_df.insert(
        0,
        "split",
        split_name
    )

    return split_df


def main():
    section("PHASE 11 — SEQUENCE-AWARE VIDEO SPLIT")

    # Load data.
    if not os.path.exists(INPUT_FILE):
        print("\nERROR: Input file not found:")
        print(INPUT_FILE)
        return

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("\nLoading modeling dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total rows  : {len(df):,}")
    print(f"Total videos: {df['video_name'].nunique()}")

    # Video-level sequence analysis.
    section("1. VIDEO-LEVEL USABLE SEQUENCE ANALYSIS")

    video_info = get_video_information(df)

    print(
        f"\nTotal unique videos: "
        f"{len(video_info)}"
    )

    print("\nPer-video sequence statistics:")

    print(
        video_info[
            [
                "video_name",
                "usable_sequences",
                "not_drowsy_sequences",
                "drowsy_sequences",
            ]
        ]
        .sort_values(
            "drowsy_sequences",
            ascending=False
        )
        .to_string(index=False)
    )

    video_info.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "video_sequence_availability.csv"
        ),
        index=False
    )

    # Create split.
    section("2. CREATING SEQUENCE-AWARE VIDEO SPLIT")

    print(f"Random seed : {RANDOM_SEED}")

    print(
        f"Target ratio: "
        f"{TRAIN_RATIO:.0%} / "
        f"{VAL_RATIO:.0%} / "
        f"{TEST_RATIO:.0%}"
    )

    (
        train_videos,
        val_videos,
        test_videos
    ) = stratified_sequence_aware_split(video_info)

    print(f"\nTrain videos: {len(train_videos)}")
    print(f"Validation videos: {len(val_videos)}")
    print(f"Test videos: {len(test_videos)}")

    # Leakage check.
    section("3. SPLIT LEAKAGE CHECK")

    train_set = set(train_videos)
    val_set = set(val_videos)
    test_set = set(test_videos)

    print(
        f"Train ∩ Validation: "
        f"{len(train_set & val_set)}"
    )

    print(
        f"Train ∩ Test      : "
        f"{len(train_set & test_set)}"
    )

    print(
        f"Validation ∩ Test : "
        f"{len(val_set & test_set)}"
    )

    if (
        train_set & val_set
        or train_set & test_set
        or val_set & test_set
    ):
        print("\nERROR: Video leakage detected!")
        return

    print("\nNo video overlap detected.")

    # Create split dataframes.
    train_df = create_split_dataframe(
        video_info,
        train_videos,
        "train"
    )

    val_df = create_split_dataframe(
        video_info,
        val_videos,
        "validation"
    )

    test_df = create_split_dataframe(
        video_info,
        test_videos,
        "test"
    )

    # Save splits.
    section("4. SAVING SPLIT FILES")

    train_df.to_csv(
        TRAIN_FILE,
        index=False
    )

    val_df.to_csv(
        VAL_FILE,
        index=False
    )

    test_df.to_csv(
        TEST_FILE,
        index=False
    )

    print(f"Train file      : {TRAIN_FILE}")
    print(f"Validation file : {VAL_FILE}")
    print(f"Test file       : {TEST_FILE}")

    # Sequence distribution.
    section("5. USABLE SEQUENCE DISTRIBUTION")

    summary_rows = []

    for split_name, split_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        total_sequences = int(
            split_df["usable_sequences"].sum()
        )

        not_drowsy_sequences = int(
            split_df["not_drowsy_sequences"].sum()
        )

        drowsy_sequences = int(
            split_df["drowsy_sequences"].sum()
        )

        drowsy_video_count = int(
            (
                split_df["drowsy_sequences"] > 0
            ).sum()
        )

        summary_rows.append({
            "split": split_name,
            "videos": len(split_df),
            "videos_with_drowsy_sequences":
                drowsy_video_count,
            "usable_sequences":
                total_sequences,
            "not_drowsy_sequences":
                not_drowsy_sequences,
            "drowsy_sequences":
                drowsy_sequences,
        })

    summary_df = pd.DataFrame(summary_rows)

    print(
        summary_df.to_string(index=False)
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # Class balance check.
    section("6. DROWSY SEQUENCE BALANCE CHECK")

    for _, row in summary_df.iterrows():
        split_name = row["split"]
        drowsy_count = row["drowsy_sequences"]
        total_count = row["usable_sequences"]

        if total_count > 0:
            drowsy_percentage = (
                drowsy_count / total_count * 100
            )
        else:
            drowsy_percentage = 0

        print(
            f"{split_name:12s}: "
            f"{int(drowsy_count):4d} drowsy / "
            f"{int(total_count):4d} total "
            f"({drowsy_percentage:.2f}%)"
        )

    # Final video check.
    section("7. FINAL VIDEO ASSIGNMENT CHECK")

    total_split_videos = (
        len(train_set)
        + len(val_set)
        + len(test_set)
    )

    original_video_count = (
        video_info["video_name"].nunique()
    )

    print(
        f"Original videos : "
        f"{original_video_count}"
    )

    print(
        f"Split videos    : "
        f"{total_split_videos}"
    )

    if total_split_videos != original_video_count:
        print("\nERROR: Some videos are missing.")
        return

    print("\nAll videos are assigned exactly once.")

    # Final.
    section("PHASE 11 UPDATED — COMPLETED")

    print("\nGenerated files:")
    print(f"  - {TRAIN_FILE}")
    print(f"  - {VAL_FILE}")
    print(f"  - {TEST_FILE}")
    print(f"  - {SUMMARY_FILE}")
    print(
        "  - "
        f"{OUTPUT_DIR}\\video_sequence_availability.csv"
    )

    print("\nIMPORTANT:")
    print("The split is performed at VIDEO level.")
    print(
        "Usable 30-frame sequences are considered "
        "before assigning videos."
    )
    print("No individual sequence is moved between splits.")
    print(
        "Temporal sequence generation should now be "
        "rerun using this updated split."
    )


if __name__ == "__main__":
    main()