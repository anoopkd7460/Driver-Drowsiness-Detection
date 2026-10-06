import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Phase 18 - Challenging Video Feature Analysis
DATA_FILE = "data/features/modeling_frame_features.csv"
TARGET_VIDEO = "P1042762_720"
FEATURES = ["ear_mean","pitch","yaw","roll"]
OUTPUT_DIR = "results/phase18_challenging_video"

os.makedirs(OUTPUT_DIR,exist_ok=True)

print("="*70)
print("PHASE 18 — CHALLENGING VIDEO FEATURE ANALYSIS")
print("="*70)

print("\nTarget video:")
print(" ",TARGET_VIDEO)

print("\nFeatures:")
for feature in FEATURES:
    print(" ",feature)

print("\nMouth features   : NO")
print("Yawning features : NO")
print("Model retraining  : NO")

# 1. Load data
print("\n"+"="*70)
print("1. LOADING MODELING FEATURES")
print("="*70)

df = pd.read_csv(DATA_FILE)

print("\nTotal rows:",len(df))
print("Total videos:",df["video_name"].nunique())

# 2. Target video check
if TARGET_VIDEO not in set(df["video_name"]):
    raise ValueError(f"{TARGET_VIDEO} not found in dataset.")

target = df[df["video_name"] == TARGET_VIDEO].copy()
others = df[df["video_name"] != TARGET_VIDEO].copy()

print("\nTarget video rows:",len(target))
print("Other video rows:",len(others))

# 3. Data quality
print("\n"+"="*70)
print("2. DATA QUALITY")
print("="*70)


def quality_report(data,name):
    print(f"\n{name}:")
    print("  Rows:",len(data))
    print("  Videos:",data["video_name"].nunique())

    for feature in FEATURES:
        valid = data[feature].notna().sum()
        missing = data[feature].isna().sum()
        print(
            f"  {feature}: valid={valid}, "
            f"missing={missing}, "
            f"validity={valid/len(data)*100:.2f}%"
        )


quality_report(target,"TARGET VIDEO")
quality_report(others,"OTHER VIDEOS")

# 4. Class distribution
print("\n"+"="*70)
print("3. CLASS DISTRIBUTION")
print("="*70)


def class_report(data,name):
    counts = data["label"].value_counts().sort_index()
    print(f"\n{name}:")

    for label in [0,1]:
        count = int(counts.get(label,0))
        percentage = count/len(data)*100
        label_name = "Not Drowsy" if label == 0 else "Drowsy"

        print(
            f"  {label_name}: "
            f"{count} ({percentage:.2f}%)"
        )


class_report(target,"TARGET VIDEO")
class_report(others,"OTHER VIDEOS")

# 5. Feature statistics
print("\n"+"="*70)
print("4. FEATURE STATISTICS")
print("="*70)

stats_rows = []

for feature in FEATURES:
    target_values = target[feature].dropna()
    other_values = others[feature].dropna()

    target_mean = target_values.mean()
    target_std = target_values.std()
    target_median = target_values.median()

    other_mean = other_values.mean()
    other_std = other_values.std()
    other_median = other_values.median()

    print(f"\n{feature}:")
    print(f"  Target mean   : {target_mean:.6f}")
    print(f"  Other mean    : {other_mean:.6f}")
    print(f"  Target std    : {target_std:.6f}")
    print(f"  Other std     : {other_std:.6f}")
    print(f"  Target median : {target_median:.6f}")
    print(f"  Other median  : {other_median:.6f}")

    stats_rows.append({
        "feature":feature,
        "target_mean":target_mean,
        "target_std":target_std,
        "target_median":target_median,
        "target_min":target_values.min(),
        "target_max":target_values.max(),
        "other_mean":other_mean,
        "other_std":other_std,
        "other_median":other_median,
        "other_min":other_values.min(),
        "other_max":other_values.max()
    })

stats_df = pd.DataFrame(stats_rows)

stats_df.to_csv(
    os.path.join(OUTPUT_DIR,"feature_statistics.csv"),
    index=False
)

# 6. Target video class-wise feature analysis
print("\n"+"="*70)
print("5. TARGET VIDEO CLASS-WISE FEATURE ANALYSIS")
print("="*70)

class_stats = []

for label in [0,1]:
    label_name = "Not Drowsy" if label == 0 else "Drowsy"
    subset = target[target["label"] == label]

    print(f"\n{label_name}:")

    for feature in FEATURES:
        values = subset[feature].dropna()

        if len(values) == 0:
            continue

        print(
            f"  {feature}: "
            f"mean={values.mean():.6f}, "
            f"std={values.std():.6f}, "
            f"median={values.median():.6f}"
        )

        class_stats.append({
            "label":label,
            "label_name":label_name,
            "feature":feature,
            "count":len(values),
            "mean":values.mean(),
            "std":values.std(),
            "median":values.median(),
            "min":values.min(),
            "max":values.max()
        })

class_stats_df = pd.DataFrame(class_stats)

class_stats_df.to_csv(
    os.path.join(OUTPUT_DIR,"target_class_feature_statistics.csv"),
    index=False
)

# 7. Distribution shift - KS test
print("\n"+"="*70)
print("6. DISTRIBUTION SHIFT — KS TEST")
print("="*70)

print("\nComparing target video against all other videos.")

ks_rows = []

for feature in FEATURES:
    target_values = target[feature].dropna().values
    other_values = others[feature].dropna().values

    statistic,p_value = ks_2samp(target_values,other_values)

    print(f"\n{feature}:")
    print(f"  KS statistic : {statistic:.6f}")
    print(f"  p-value      : {p_value:.6e}")

    ks_rows.append({
        "feature":feature,
        "ks_statistic":statistic,
        "p_value":p_value
    })

ks_df = pd.DataFrame(ks_rows)

ks_df.to_csv(
    os.path.join(OUTPUT_DIR,"distribution_shift_ks_test.csv"),
    index=False
)

# 8. Temporal variability
print("\n"+"="*70)
print("7. TEMPORAL VARIABILITY")
print("="*70)

temporal_rows = []

for video_name,video_data in df.groupby("video_name"):
    video_data = video_data.sort_values("frame_number")
    row = {"video_name":video_name}

    for feature in FEATURES:
        values = video_data[feature].dropna().values

        if len(values) < 2:
            row[f"{feature}_temporal_std"] = np.nan
            row[f"{feature}_mean_abs_change"] = np.nan
            continue

        differences = np.diff(values)
        row[f"{feature}_temporal_std"] = np.std(values)
        row[f"{feature}_mean_abs_change"] = np.mean(np.abs(differences))

    temporal_rows.append(row)

temporal_df = pd.DataFrame(temporal_rows)

target_temporal = temporal_df[
    temporal_df["video_name"] == TARGET_VIDEO
]

other_temporal = temporal_df[
    temporal_df["video_name"] != TARGET_VIDEO
]

print("\nTarget video temporal behavior:")
print(target_temporal.to_string(index=False))

print("\nOther-video temporal averages:")

for feature in FEATURES:
    std_col = f"{feature}_temporal_std"
    change_col = f"{feature}_mean_abs_change"

    print(f"\n{feature}:")
    print(
        "  Target temporal std:",
        target_temporal[std_col].iloc[0]
    )
    print(
        "  Other mean temporal std:",
        other_temporal[std_col].mean()
    )
    print(
        "  Target mean abs change:",
        target_temporal[change_col].iloc[0]
    )
    print(
        "  Other mean abs change:",
        other_temporal[change_col].mean()
    )

temporal_df.to_csv(
    os.path.join(OUTPUT_DIR,"temporal_variability_by_video.csv"),
    index=False
)

# 9. Frame continuity
print("\n"+"="*70)
print("8. FRAME CONTINUITY")
print("="*70)

target_sorted = target.sort_values("frame_number")
frames = target_sorted["frame_number"].values

if len(frames) > 1:
    gaps = np.diff(frames)
    discontinuities = np.sum(gaps != 1)

    print("\nFrame discontinuities:",discontinuities)
    print("Largest frame gap:",gaps.max())
else:
    print("\nNot enough frames.")

# 10. Sequence-level analysis
print("\n"+"="*70)
print("9. SEQUENCE-LEVEL ANALYSIS")
print("="*70)


def build_sequence_statistics(video_data):
    video_data = (
        video_data
        .sort_values("frame_number")
        .reset_index(drop=True)
    )

    sequences = []
    start = 0

    while start+30 <= len(video_data):
        chunk = video_data.iloc[start:start+30]

        frames = chunk["frame_number"].values
        expected = np.arange(frames[0],frames[0]+30)

        if not np.array_equal(frames,expected):
            start += 1
            continue

        labels = chunk["label"].values

        if not np.all(labels == labels[0]):
            start += 30
            continue

        row = {
            "video_name":video_data.iloc[start]["video_name"],
            "start_frame":int(frames[0]),
            "end_frame":int(frames[-1]),
            "label":int(labels[0])
        }

        for feature in FEATURES:
            values = chunk[feature].values

            row[f"{feature}_mean"] = np.mean(values)
            row[f"{feature}_std"] = np.std(values)
            row[f"{feature}_min"] = np.min(values)
            row[f"{feature}_max"] = np.max(values)
            row[f"{feature}_range"] = np.max(values)-np.min(values)
            row[f"{feature}_mean_abs_change"] = np.mean(np.abs(np.diff(values)))

        sequences.append(row)
        start += 30

    return pd.DataFrame(sequences)


target_sequences = build_sequence_statistics(target)

print("\nTarget video sequences:",len(target_sequences))

if len(target_sequences) > 0:
    print("\nTarget sequence class distribution:")
    print(
        target_sequences["label"]
        .value_counts()
        .sort_index()
    )

target_sequences.to_csv(
    os.path.join(OUTPUT_DIR,"target_sequence_statistics.csv"),
    index=False
)

# 11. Extreme value check
print("\n"+"="*70)
print("10. EXTREME VALUE CHECK")
print("="*70)

for feature in FEATURES:
    values = target[feature].dropna()

    if len(values) == 0:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3-q1
    lower = q1-1.5*iqr
    upper = q3+1.5*iqr

    extreme = (values < lower)|(values > upper)

    print(f"\n{feature}:")
    print(f"  Q1       : {q1:.6f}")
    print(f"  Q3       : {q3:.6f}")
    print(f"  IQR      : {iqr:.6f}")
    print(
        f"  Outliers : "
        f"{extreme.sum()} "
        f"({extreme.mean()*100:.2f}%)"
    )

# 12. Save final report
report = {
    "target_video":TARGET_VIDEO,
    "target_rows":len(target),
    "other_rows":len(others),
    "target_sequences":len(target_sequences),
    "target_drowsy_frames":int(np.sum(target["label"] == 1)),
    "target_not_drowsy_frames":int(np.sum(target["label"] == 0))
}

pd.DataFrame([report]).to_csv(
    os.path.join(OUTPUT_DIR,"phase18_summary.csv"),
    index=False
)

# Final
print("\n"+"="*70)
print("PHASE 18 COMPLETED")
print("="*70)

print("\nChallenging video analyzed:",TARGET_VIDEO)
print("\nNo model retraining.")
print("No test set modification.")
print("No mouth/yawning features.")
print("\nResults saved to:")
print(OUTPUT_DIR)

print("\n"+"="*70)