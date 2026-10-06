# src/features/feature_quality_analysis.py

import os
import numpy as np
import pandas as pd

FEATURE_FILE = "data/features/frame_features.csv"
OUTPUT_DIR = "data/features"

QUALITY_REPORT = os.path.join(OUTPUT_DIR, "feature_quality_summary.csv")
CLASS_COMPARISON_REPORT = os.path.join(OUTPUT_DIR, "feature_class_comparison.csv")
VIDEO_QUALITY_REPORT = os.path.join(OUTPUT_DIR, "video_quality_summary.csv")

# Continuous features only
FEATURES = [
    "ear_left", "ear_right", "ear_mean",
    "eye_openness_left", "eye_openness_right", "eye_openness_mean",
    "pitch", "yaw", "roll"
]


# Print section heading
def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_percentage(numerator, denominator):
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def analyze_feature_quality(df):
    print_section("NaN / INFINITE VALUE ANALYSIS")

    quality_rows = []

    for feature in FEATURES:
        series = pd.to_numeric(df[feature], errors="coerce")
        nan_count = series.isna().sum()
        infinite_count = np.isinf(series.dropna()).sum()
        valid_values = series.replace([np.inf, -np.inf], np.nan).dropna()

        if len(valid_values) > 0:
            min_value = valid_values.min()
            max_value = valid_values.max()
            mean_value = valid_values.mean()
            std_value = valid_values.std()
            unique_count = valid_values.nunique()
        else:
            min_value = np.nan
            max_value = np.nan
            mean_value = np.nan
            std_value = np.nan
            unique_count = 0

        print(f"\n{feature}")
        print(f"  NaN       : {nan_count:,}")
        print(f"  Infinite  : {infinite_count:,}")
        print(f"  Unique    : {unique_count:,}")

        if len(valid_values) > 0:
            print(f"  Min       : {min_value:.4f}")
            print(f"  Max       : {max_value:.4f}")
            print(f"  Mean      : {mean_value:.4f}")
            print(f"  Std       : {std_value:.4f}")
        else:
            print("  No valid values")

        quality_rows.append({
            "feature": feature,
            "total_rows": len(df),
            "nan_count": nan_count,
            "infinite_count": infinite_count,
            "valid_count": len(valid_values),
            "validity_percentage": safe_percentage(len(valid_values), len(df)),
            "unique_count": unique_count,
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "std": std_value,
        })

    return pd.DataFrame(quality_rows)


def analyze_class_distribution(df):
    print_section("DROWSY vs NOT DROWSY FEATURE ANALYSIS")

    rows = []

    for feature in FEATURES:
        for label, label_name in [(0, "Not Drowsy"), (1, "Drowsy")]:
            subset = pd.to_numeric(
                df.loc[df["label"] == label, feature],
                errors="coerce"
            )
            subset = subset.replace([np.inf, -np.inf], np.nan).dropna()

            if len(subset) == 0:
                count = 0
                mean_value = np.nan
                median_value = np.nan
                std_value = np.nan
            else:
                count = len(subset)
                mean_value = subset.mean()
                median_value = subset.median()
                std_value = subset.std()

            print(f"\n{feature} | {label_name}")
            print(f"  Count  : {count:,}")

            if count > 0:
                print(f"  Mean   : {mean_value:.4f}")
                print(f"  Median : {median_value:.4f}")
                print(f"  Std    : {std_value:.4f}")
            else:
                print("  No valid values")

            rows.append({
                "feature": feature,
                "label": label,
                "class": label_name,
                "count": count,
                "mean": mean_value,
                "median": median_value,
                "std": std_value,
            })

    return pd.DataFrame(rows)


def analyze_outliers(df):
    print_section("OUTLIER ANALYSIS — IQR METHOD")

    rows = []

    for feature in FEATURES:
        series = pd.to_numeric(df[feature], errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan).dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = outlier_mask.sum()
        outlier_percentage = safe_percentage(outlier_count, len(series))

        print(f"\n{feature}")
        print(f"  Lower bound : {lower_bound:.4f}")
        print(f"  Upper bound : {upper_bound:.4f}")
        print(f"  Outliers    : {outlier_count:,}")
        print(f"  Percentage  : {outlier_percentage:.2f}%")

        rows.append({
            "feature": feature,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage,
        })

    return pd.DataFrame(rows)


def analyze_video_quality(df):
    print_section("VIDEO-WISE QUALITY ANALYSIS")

    video_rows = []

    for video_name, group in df.groupby("video_name", dropna=False):
        total_frames = len(group)

        image_available = (group["image_exists"] == 1).sum()
        face_detected = (group["face_detected"] == 1).sum()

        # eye_validity == 1 means valid eye features
        eye_valid = (group["eye_validity"] == 1).sum()

        image_availability_percentage = safe_percentage(image_available, total_frames)
        face_detection_percentage = safe_percentage(face_detected, total_frames)
        eye_validity_percentage = safe_percentage(eye_valid, total_frames)

        video_rows.append({
            "video_name": video_name,
            "total_frames": total_frames,
            "images_available": image_available,
            "images_missing": total_frames - image_available,
            "faces_detected": face_detected,
            "faces_not_detected": total_frames - face_detected,
            "eye_valid": eye_valid,
            "eye_invalid": total_frames - eye_valid,
            "image_availability_percentage": image_availability_percentage,
            "face_detection_percentage": face_detection_percentage,
            "eye_validity_percentage": eye_validity_percentage,
        })

    video_df = pd.DataFrame(video_rows)

    if len(video_df) > 0:
        lowest_quality = video_df.sort_values(
            by="face_detection_percentage",
            ascending=True
        ).head(10)

        print("\nVideos with lowest face detection:")
        print(
            lowest_quality[
                [
                    "video_name",
                    "total_frames",
                    "image_availability_percentage",
                    "face_detection_percentage",
                    "eye_validity_percentage",
                ]
            ].to_string(index=False)
        )

    return video_df


def main():
    print("=" * 70)
    print("FEATURE QUALITY ANALYSIS")
    print("=" * 70)

    # Check feature file
    if not os.path.exists(FEATURE_FILE):
        print("\nERROR: Feature file not found:")
        print(FEATURE_FILE)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading feature dataset...")
    df = pd.read_csv(FEATURE_FILE)

    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    # BASIC DATASET INFORMATION
    print_section("BASIC DATASET INFORMATION")

    class_counts = df["label"].value_counts().sort_index()
    total = len(df)

    not_drowsy = class_counts.get(0, 0)
    drowsy = class_counts.get(1, 0)

    print("\nClass distribution:")
    print(
        f"  0 = Not Drowsy     {not_drowsy:,} "
        f"({safe_percentage(not_drowsy, total):.2f}%)"
    )
    print(
        f"  1 = Drowsy          {drowsy:,} "
        f"({safe_percentage(drowsy, total):.2f}%)"
    )

    # IMAGE AVAILABILITY
    print_section("IMAGE AVAILABILITY")

    images_available = (df["image_exists"] == 1).sum()
    images_missing = total - images_available

    print(f"Images available : {images_available:,}")
    print(f"Images missing   : {images_missing:,}")
    print(f"Availability rate: {safe_percentage(images_available, total):.2f}%")
    print(f"Missing rate     : {safe_percentage(images_missing, total):.2f}%")

    # FACE DETECTION
    print_section("FACE DETECTION QUALITY")

    face_detected = (df["face_detected"] == 1).sum()
    face_not_detected = total - face_detected

    print(f"Face detected     : {face_detected:,}")
    print(f"Face not detected : {face_not_detected:,}")
    print(f"Detection rate    : {safe_percentage(face_detected, total):.2f}%")

    # EYE VALIDITY
    print_section("EYE VALIDITY")

    eye_valid = (df["eye_validity"] == 1).sum()
    eye_invalid = (df["eye_validity"] == 0).sum()

    print(f"Valid eye features   : {eye_valid:,}")
    print(f"Invalid eye features : {eye_invalid:,}")
    print(f"Validity rate        : {safe_percentage(eye_valid, total):.2f}%")

    print("\nEye validity values:")
    print(
        df["eye_validity"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # FEATURES
    print_section("FEATURES BEING ANALYZED")

    for feature in FEATURES:
        print(f"  - {feature}")

    print(
        "\nNOTE: eye_validity is excluded from continuous "
        "feature/outlier analysis because it is a binary quality flag."
    )

    # FEATURE QUALITY
    quality_df = analyze_feature_quality(df)
    quality_df.to_csv(QUALITY_REPORT, index=False)

    # CLASS COMPARISON
    class_df = analyze_class_distribution(df)
    class_df.to_csv(CLASS_COMPARISON_REPORT, index=False)

    # OUTLIERS
    outlier_df = analyze_outliers(df)

    # Add outlier information to main quality report
    if len(outlier_df) > 0:
        quality_df = quality_df.merge(
            outlier_df[
                [
                    "feature",
                    "q1",
                    "q3",
                    "iqr",
                    "lower_bound",
                    "upper_bound",
                    "outlier_count",
                    "outlier_percentage",
                ]
            ],
            on="feature",
            how="left"
        )

        quality_df.to_csv(QUALITY_REPORT, index=False)

    # VIDEO QUALITY
    video_df = analyze_video_quality(df)
    video_df.to_csv(VIDEO_QUALITY_REPORT, index=False)

    print("\nVideo quality report saved:")
    print(VIDEO_QUALITY_REPORT)

    # FINAL SUMMARY
    print_section("PHASE 10A COMPLETED")

    print("\nFeature quality report:")
    print(QUALITY_REPORT)

    print("\nClass comparison report:")
    print(CLASS_COMPARISON_REPORT)

    print("\nVideo quality report:")
    print(VIDEO_QUALITY_REPORT)

    print("\nCorrect eye validity:")
    print(
        f"  Valid   : {eye_valid:,} "
        f"({safe_percentage(eye_valid, total):.2f}%)"
    )
    print(
        f"  Invalid : {eye_invalid:,} "
        f"({safe_percentage(eye_invalid, total):.2f}%)"
    )

    print("\nNext step:")
    print("Review pitch/yaw/roll distributions before starting Phase 10B.")


if __name__ == "__main__":
    main()