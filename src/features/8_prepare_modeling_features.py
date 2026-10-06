'''Convert raw extracted feature table to modeling-ready table'''

import os
import pandas as pd
import numpy as np

INPUT_FILE = "data/features/frame_features.csv"

OUTPUT_DIR = "data/features"
OUTPUT_FILE = os.path.join(OUTPUT_DIR,"modeling_frame_features.csv")

SELECTED_FEATURES = [
    "ear_mean",
    "pitch",
    "yaw",
    "roll",
]

REQUIRED_COLUMNS = [
    "video_name",
    "frame_number",
    "driver_state",
    "label",
    "image_exists",
    "face_detected",
    "eye_validity",
] + SELECTED_FEATURES


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("PHASE 10C — FINAL FEATURE SELECTION")

    # LOAD DATA
    if not os.path.exists(INPUT_FILE):
        print("ERROR: Input file not found:")
        print(INPUT_FILE)
        return

    print("\nLoading feature dataset...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Input rows    : {len(df):,}")
    print(f"Input columns : {len(df.columns)}")

    # CHECK REQUIRED COLUMNS
    section("1. COLUMN VALIDATION")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("\nERROR: Missing required columns:")

        for column in missing_columns:
            print(f"  - {column}")

        return

    print("All required columns are present.")

    # SELECT FINAL COLUMNS
    final_columns = [
        "video_name",
        "frame_number",
        "driver_state",
        "label",
        "image_exists",
        "face_detected",
        "eye_validity",
    ] + SELECTED_FEATURES

    modeling_df = df[final_columns].copy()

    # NUMERIC CONVERSION
    section("2. NUMERIC FEATURE VALIDATION")

    for feature in SELECTED_FEATURES:
        modeling_df[feature] = pd.to_numeric(
            modeling_df[feature],
            errors="coerce"
        )

    modeling_df["label"] = pd.to_numeric(
        modeling_df["label"],
        errors="coerce"
    )

    modeling_df["frame_number"] = pd.to_numeric(
        modeling_df["frame_number"],
        errors="coerce"
    )

    # Replace infinite values with NaN
    modeling_df = modeling_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print("\nSelected features:")

    for feature in SELECTED_FEATURES:
        print(f"  - {feature}")

    # FEATURE AVAILABILITY
    section("3. FEATURE AVAILABILITY")

    availability_rows = []

    for feature in SELECTED_FEATURES:
        valid_count = modeling_df[feature].notna().sum()
        missing_count = modeling_df[feature].isna().sum()

        availability = (
            valid_count / len(modeling_df)
        ) * 100

        print(f"\n{feature}")
        print(f"  Valid   : {valid_count:,}")
        print(f"  Missing : {missing_count:,}")
        print(f"  Rate    : {availability:.2f}%")

        availability_rows.append({
            "feature": feature,
            "valid_count": valid_count,
            "missing_count": missing_count,
            "validity_percentage": availability,
        })

    availability_df = pd.DataFrame(availability_rows)

    availability_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "modeling_feature_availability.csv"
        ),
        index=False
    )

    # VALID MODELING ROWS
    section("4. VALID MODELING ROW ANALYSIS")

    valid_feature_mask = modeling_df[
        SELECTED_FEATURES
    ].notna().all(axis=1)

    valid_rows = valid_feature_mask.sum()
    invalid_rows = len(modeling_df) - valid_rows

    print(f"Rows with all 4 features valid : {valid_rows:,}")
    print(f"Rows with missing feature      : {invalid_rows:,}")

    print(
        f"Complete-feature rate           : "
        f"{(valid_rows / len(modeling_df)) * 100:.2f}%"
    )

    # IMPORTANT:
    # DO NOT INTERPOLATE FEATURES
    print("\nMissing feature handling:")
    print("  - No interpolation")
    print("  - No forward filling")
    print("  - No backward filling")
    print("  - No artificial facial measurements")

    # Keep all rows in this frame-level dataset.
    # Temporal sequence generation will later select
    # sequences where all required frames/features are valid.

    # SORT DATA
    section("5. SORTING")

    modeling_df = modeling_df.sort_values(
        by=[
            "video_name",
            "frame_number",
        ]
    ).reset_index(drop=True)

    print("Sorted by video_name and frame_number.")

    # DUPLICATE CHECK
    section("6. DUPLICATE FRAME CHECK")

    duplicate_mask = modeling_df.duplicated(
        subset=[
            "video_name",
            "frame_number",
        ],
        keep=False
    )

    duplicate_count = duplicate_mask.sum()

    print(
        f"Duplicate video/frame rows : "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:
        print("\nWARNING:")
        print(
            "Duplicate video/frame combinations detected."
        )

        duplicate_df = modeling_df.loc[
            duplicate_mask
        ]

        duplicate_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "duplicate_video_frames.csv"
            ),
            index=False
        )
    else:
        print("No duplicate video/frame combinations found.")

    # CLASS DISTRIBUTION
    section("7. CLASS DISTRIBUTION")

    class_distribution = (
        modeling_df["label"]
        .value_counts()
        .sort_index()
    )

    total = len(modeling_df)

    for label, count in class_distribution.items():
        if label == 0:
            name = "Not Drowsy"
        elif label == 1:
            name = "Drowsy"
        else:
            name = "Unknown"

        percentage = (
            count / total
        ) * 100

        print(
            f"  {label} = {name:12s} "
            f"{count:,} ({percentage:.2f}%)"
        )

    class_distribution.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "modeling_class_distribution.csv"
        )
    )

    # VIDEO COUNT
    section("8. VIDEO INFORMATION")

    video_count = modeling_df[
        "video_name"
    ].nunique()

    print(f"Unique videos: {video_count}")

    videos_per_class = (
        modeling_df.groupby("label")[
            "video_name"
        ].nunique()
    )

    print("\nVideos represented by class:")

    for label, count in videos_per_class.items():
        name = (
            "Not Drowsy"
            if label == 0
            else "Drowsy"
        )

        print(
            f"  {label} = {name}: {count}"
        )

    # FINAL FEATURE SUMMARY
    section("9. FINAL FEATURE SET")

    print("\nFinal modeling features:")

    for i, feature in enumerate(
        SELECTED_FEATURES,
        start=1
    ):
        print(f"  {i}. {feature}")

    print("\nFeature groups:")
    print("  Eye behavior:")
    print("    - ear_mean")

    print("  Head pose:")
    print("    - pitch")
    print("    - yaw")
    print("    - roll")

    # SAVE MODELING DATASET
    section("10. SAVING MODELING DATASET")

    modeling_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved successfully:")
    print(OUTPUT_FILE)

    # FINAL SUMMARY
    section("PHASE 10C COMPLETED")

    print("\nInput dataset:")
    print(f"  {INPUT_FILE}")

    print("\nOutput dataset:")
    print(f"  {OUTPUT_FILE}")

    print("\nRows:")
    print(f"  {len(modeling_df):,}")

    print("\nColumns:")

    for column in modeling_df.columns:
        print(f"  - {column}")

    print("\nIMPORTANT:")
    print(
        "The original frame_features.csv has NOT been modified."
    )

    print("\nNext step:")
    print(
        "Phase 11 — Video/Subject-wise Train/Validation/Test Split"
    )


if __name__ == "__main__":
    main()