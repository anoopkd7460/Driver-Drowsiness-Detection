import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURE_FILE = "data/features/frame_features.csv"
OUTPUT_DIR = "data/features/phase10b"

FEATURES = [
    "ear_left", "ear_right", "ear_mean",
    "eye_openness_left", "eye_openness_right", "eye_openness_mean",
    "pitch", "yaw", "roll"
]


# Print section heading
def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("FEATURE DISTRIBUTION & RELATIONSHIP ANALYSIS")

    # Check feature file
    if not os.path.exists(FEATURE_FILE):
        print(f"ERROR: Feature file not found: {FEATURE_FILE}")
        return

    # Create output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading feature dataset...")
    df = pd.read_csv(FEATURE_FILE)
    print(f"Total rows: {len(df):,}")

    # PREPARE DATA
    feature_df = df[FEATURES + ["label"]].copy()

    # Convert features to numeric
    for feature in FEATURES:
        feature_df[feature] = pd.to_numeric(feature_df[feature], errors="coerce")

    # Replace infinite values
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    # 1. DESCRIPTIVE STATISTICS
    section("1. DESCRIPTIVE STATISTICS")

    stats = feature_df[FEATURES].describe().T

    # Add missing value information
    stats["missing"] = feature_df[FEATURES].isna().sum()
    stats["missing_percentage"] = stats["missing"] / len(feature_df) * 100

    print(stats[[
        "count", "mean", "std", "min", "25%", "50%",
        "75%", "max", "missing", "missing_percentage"
    ]].round(4).to_string())

    # Save statistics
    stats.to_csv(os.path.join(OUTPUT_DIR, "feature_descriptive_statistics.csv"))

    # 2. CORRELATION MATRIX
    section("2. FEATURE CORRELATION MATRIX")

    # Calculate correlation
    correlation = feature_df[FEATURES].corr(method="pearson")
    print(correlation.round(3).to_string())

    # Save correlation matrix
    correlation.to_csv(os.path.join(OUTPUT_DIR, "feature_correlation_matrix.csv"))

    # 3. HIGH CORRELATION PAIRS
    section("3. HIGHLY CORRELATED FEATURE PAIRS")

    pairs = []

    # Check feature pairs
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            f1 = FEATURES[i]
            f2 = FEATURES[j]
            corr_value = correlation.loc[f1, f2]

            # Check high correlation
            if abs(corr_value) >= 0.80:
                pairs.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "correlation": corr_value,
                    "absolute_correlation": abs(corr_value),
                })

                print(f"{f1:25s} <-> {f2:25s} : {corr_value:.4f}")

    high_corr_df = pd.DataFrame(pairs)

    # Save high correlation pairs
    high_corr_df.to_csv(
        os.path.join(OUTPUT_DIR, "high_correlation_pairs.csv"),
        index=False
    )

    if len(high_corr_df) == 0:
        print("No feature pairs with |correlation| >= 0.80")

    # 4. CLASS-WISE FEATURE DISTRIBUTION
    section("4. CLASS-WISE FEATURE DISTRIBUTION")

    class_rows = []

    # Calculate class-wise statistics
    for feature in FEATURES:
        for label, class_name in [(0, "Not Drowsy"), (1, "Drowsy")]:
            values = feature_df.loc[feature_df["label"] == label, feature].dropna()

            if len(values) == 0:
                continue

            class_rows.append({
                "feature": feature,
                "label": label,
                "class": class_name,
                "count": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
            })

    class_distribution_df = pd.DataFrame(class_rows)

    print(class_distribution_df.round(4).to_string(index=False))

    # Save class-wise statistics
    class_distribution_df.to_csv(
        os.path.join(OUTPUT_DIR, "class_wise_feature_distribution.csv"),
        index=False
    )

    # 5. STANDARDIZED CLASS SEPARATION
    section("5. CLASS SEPARATION ANALYSIS")

    separation_rows = []

    # Compare both classes
    for feature in FEATURES:
        not_drowsy = feature_df.loc[feature_df["label"] == 0, feature].dropna()
        drowsy = feature_df.loc[feature_df["label"] == 1, feature].dropna()

        if len(not_drowsy) == 0 or len(drowsy) == 0:
            continue

        mean_0 = not_drowsy.mean()
        mean_1 = drowsy.mean()
        std_0 = not_drowsy.std()
        std_1 = drowsy.std()

        # Calculate pooled standard deviation
        pooled_std = np.sqrt((std_0 ** 2 + std_1 ** 2) / 2)

        # Calculate class separation
        if pooled_std != 0:
            effect_size = (mean_0 - mean_1) / pooled_std
        else:
            effect_size = np.nan

        separation_rows.append({
            "feature": feature,
            "not_drowsy_mean": mean_0,
            "drowsy_mean": mean_1,
            "mean_difference": mean_0 - mean_1,
            "pooled_std": pooled_std,
            "standardized_separation": effect_size,
            "absolute_separation": abs(effect_size),
        })

    # Sort by class separation
    separation_df = pd.DataFrame(separation_rows).sort_values(
        "absolute_separation",
        ascending=False
    )

    print(separation_df.round(4).to_string(index=False))

    # Save separation results
    separation_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_class_separation.csv"),
        index=False
    )

    # 6. DISTRIBUTION PLOTS
    section("6. GENERATING DISTRIBUTION PLOTS")

    # Create plots for each feature
    for feature in FEATURES:
        values_0 = feature_df.loc[feature_df["label"] == 0, feature].dropna()
        values_1 = feature_df.loc[feature_df["label"] == 1, feature].dropna()

        if len(values_0) == 0 or len(values_1) == 0:
            continue

        plt.figure(figsize=(9, 5))

        plt.hist(
            values_0,
            bins=50,
            alpha=0.6,
            label="Not Drowsy",
            density=True
        )

        plt.hist(
            values_1,
            bins=50,
            alpha=0.6,
            label="Drowsy",
            density=True
        )

        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.title(f"{feature}: Drowsy vs Not Drowsy")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()

        # Save plot
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"{feature}_distribution.png"),
            dpi=200
        )

        plt.close()

    # 7. CORRELATION HEATMAP
    section("7. GENERATING CORRELATION HEATMAP")

    plt.figure(figsize=(11, 9))

    plt.imshow(
        correlation,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(FEATURES)),
        FEATURES,
        rotation=90
    )

    plt.yticks(range(len(FEATURES)), FEATURES)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()

    # Save heatmap
    plt.savefig(
        os.path.join(OUTPUT_DIR, "feature_correlation_heatmap.png"),
        dpi=200
    )

    plt.close()

    # 8. HEAD POSE SUMMARY
    section("8. HEAD POSE FEATURE SUMMARY")

    pose_features = ["pitch", "yaw", "roll"]
    pose_summary = feature_df[pose_features].describe().T

    print(pose_summary.round(4).to_string())

    # Save head pose summary
    pose_summary.to_csv(os.path.join(OUTPUT_DIR, "head_pose_summary.csv"))

    # COMPLETE
    section("PHASE 10B COMPLETED")

    print("\nOutputs saved in:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")

    for filename in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {filename}")


if __name__ == "__main__":
    main()