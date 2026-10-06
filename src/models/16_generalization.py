import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Phase 16 — Video-Level Generalization / Robustness
SEEDS = [7, 21, 42, 84, 123]

SEQUENCE_LENGTH = 30
NUM_FEATURES = 4

FEATURES = [
    "ear_mean",
    "pitch",
    "yaw",
    "roll",
]

DATA_FILE = "data/features/modeling_frame_features.csv"
OUTPUT_DIR = "results/phase16_generalization"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 16 — VIDEO-LEVEL GENERALIZATION")
print("=" * 70)

print("\nConfiguration:")
print("  Sequence length :", SEQUENCE_LENGTH)
print("  Features        :", NUM_FEATURES)
print("  Feature names   :", ", ".join(FEATURES))
print("  Seeds            :", SEEDS)
print("  Evaluation unit  : VIDEO")
print("  Mouth features   : NO")
print("  Yawning features : NO")

# Reproducibility
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Load frame-level data
print("\n" + "=" * 70)
print("1. LOADING FRAME FEATURES")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nRows :", len(df))
print("Columns :", len(df))
print("Unique videos :", df["video_name"].nunique())

# Valid frame filter
required_columns = ["video_name", "frame_number", "label"] + FEATURES

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

valid_df = df.dropna(subset=FEATURES).copy()

print("\nFrames with all features valid :", len(valid_df))
print("Videos with valid frames :", valid_df["video_name"].nunique())

# Create continuous 30-frame sequences
print("\n" + "=" * 70)
print("2. BUILDING VIDEO-LEVEL SEQUENCES")
print("=" * 70)

def create_sequences(video_df):
    video_df = video_df.sort_values("frame_number").reset_index(drop=True)

    sequences = []

    values = video_df[FEATURES].values.astype(np.float32)
    frames = video_df["frame_number"].values
    labels = video_df["label"].values.astype(int)

    start = 0

    while start + SEQUENCE_LENGTH <= len(video_df):
        end = start + SEQUENCE_LENGTH
        current_frames = frames[start:end]

        # Must be consecutive frames
        expected = np.arange(
            current_frames[0],
            current_frames[0] + SEQUENCE_LENGTH,
        )

        if not np.array_equal(current_frames, expected):
            start += 1
            continue

        current_labels = labels[start:end]

        # All frames must have same label
        if not np.all(current_labels == current_labels[0]):
            start += 1
            continue

        sequences.append(
            {
                "video_name": video_df.iloc[start]["video_name"],
                "start_frame": int(current_frames[0]),
                "end_frame": int(current_frames[-1]),
                "label": int(current_labels[0]),
                "features": values[start:end],
            }
        )

        # Non-overlapping windows
        start += SEQUENCE_LENGTH

    return sequences


all_sequences = []

for video_name, video_df in valid_df.groupby("video_name"):
    video_sequences = create_sequences(video_df)
    all_sequences.extend(video_sequences)

print("\nTotal sequences:", len(all_sequences))

sequence_records = []

for idx, seq in enumerate(all_sequences):
    sequence_records.append(
        {
            "sequence_index": idx,
            "video_name": seq["video_name"],
            "start_frame": seq["start_frame"],
            "end_frame": seq["end_frame"],
            "label": seq["label"],
            "features": seq["features"],
        }
    )

# Check sequence distribution
sequence_labels = np.array(
    [s["label"] for s in sequence_records]
)

print("\nSequence distribution:")
print("  Not Drowsy :", np.sum(sequence_labels == 0))
print("  Drowsy     :", np.sum(sequence_labels == 1))

# Video-level information
video_names = np.array(
    [s["video_name"] for s in sequence_records]
)

unique_videos = np.unique(video_names)

print("\nVideos containing sequences:", len(unique_videos))

# Model
def build_model():
    model = Sequential(
        [
            tf.keras.Input(shape=(SEQUENCE_LENGTH,NUM_FEATURES,)),
            LSTM(64,return_sequences=False,),
            Dropout(0.30),
            Dense(32,activation="relu",),
            Dropout(0.20),
            Dense(1,activation="sigmoid",),
        ],
        name="Phase16_LSTM",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model

# Results storage
results = []
video_results = []

# Repeated video-level splits
for seed in SEEDS:
    print("\n" + "=" * 70)
    print(f"EXPERIMENT SEED = {seed}")
    print("=" * 70)

    set_seed(seed)

    # Video-level train / validation split
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed,
    )

    indices = np.arange(len(sequence_records))

    groups = np.array(
        [s["video_name"] for s in sequence_records]
    )

    train_idx, val_idx = next(
        splitter.split(
            indices,
            sequence_labels,
            groups,
        )
    )

    train_sequences = [
        sequence_records[i]
        for i in train_idx
    ]

    val_sequences = [
        sequence_records[i]
        for i in val_idx
    ]

    train_videos = sorted(set(s["video_name"] for s in train_sequences))

    val_videos = sorted(set(s["video_name"] for s in val_sequences))

    overlap = set(train_videos) & set(val_videos)

    if overlap:
        raise RuntimeError(
            f"VIDEO LEAKAGE DETECTED: {overlap}"
        )

    # Build arrays
    X_train = np.array(
        [s["features"] for s in train_sequences],
        dtype=np.float32,
    )

    y_train = np.array(
        [s["label"] for s in train_sequences],
        dtype=np.int32,
    )

    X_val = np.array(
        [s["features"] for s in val_sequences],
        dtype=np.float32,
    )

    y_val = np.array(
        [s["label"] for s in val_sequences],
        dtype=np.int32,
    )

    print("\nTrain:")
    print("  Videos    :", len(train_videos))
    print("  Sequences :", len(X_train))
    print("  Not Drowsy:", np.sum(y_train == 0))
    print("  Drowsy    :", np.sum(y_train == 1))

    print("\nValidation:")
    print("  Videos    :", len(val_videos))
    print("  Sequences :", len(X_val))
    print("  Not Drowsy:", np.sum(y_val == 0))
    print("  Drowsy    :", np.sum(y_val == 1))

    # Check both classes
    if len(np.unique(y_train)) < 2:
        print(
            "\nSkipping seed because "
            "training set has only one class."
        )
        continue

    if len(np.unique(y_val)) < 2:
        print(
            "\nSkipping seed because "
            "validation set has only one class."
        )
        continue

    # Standardization
    train_flat = X_train.reshape(-1, NUM_FEATURES)

    mean = np.mean(train_flat, axis=0)
    std = np.std(train_flat, axis=0)

    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Class weights
    classes = np.array([0, 1])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    class_weights = {
        int(c): float(w)
        for c, w in zip(classes, weights)
    }

    print("\nClass weights:", class_weights)

    # Train model
    model = build_model()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=0,
    )

    # Validation predictions
    probabilities = (
        model.predict(
            X_val,
            verbose=0,
        )
        .ravel()
    )

    predictions = (probabilities >= 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(y_val,predictions,)

    precision = precision_score(y_val,predictions,zero_division=0,)

    recall = recall_score(y_val,predictions,zero_division=0,)

    f1 = f1_score(y_val,predictions,zero_division=0,)

    cm = confusion_matrix(y_val,predictions,)

    best_epoch = (np.argmin(history.history["val_loss"]) + 1)

    best_val_loss = np.min(history.history["val_loss"])

    print("\nResults:")
    print(f"  Best epoch : {best_epoch}")
    print(f"  Val loss   : {best_val_loss:.5f}")
    print(f"  Accuracy   : {accuracy * 100:.2f}%")
    print(f"  Precision  : {precision * 100:.2f}%")
    print(f"  Recall     : {recall * 100:.2f}%")
    print(f"  F1         : {f1 * 100:.2f}%")

    print("\n  Confusion matrix:")
    print(cm)

    # Store experiment result
    results.append(
        {
            "seed": seed,
            "train_videos": len(train_videos),
            "validation_videos": len(val_videos),
            "train_sequences": len(X_train),
            "validation_sequences": len(X_val),
            "train_drowsy": int(np.sum(y_train == 1)),
            "validation_drowsy": int(np.sum(y_val == 1)),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }
    )

    # Video-wise results
    for video in val_videos:
        mask = np.array(
            [
                s["video_name"] == video
                for s in val_sequences
            ]
        )

        video_true = y_val[mask]
        video_pred = predictions[mask]

        video_results.append(
            {
                "seed": seed,
                "video_name": video,
                "sequences": len(video_true),
                "drowsy_sequences": int(
                    np.sum(video_true == 1)
                ),
                "accuracy": accuracy_score(
                    video_true,
                    video_pred,
                ),
                "recall_drowsy": (
                    recall_score(
                        video_true,
                        video_pred,
                        zero_division=0,
                    )
                    if 1 in video_true
                    else np.nan
                ),
            }
        )

# Save results
results_df = pd.DataFrame(results)
video_results_df = pd.DataFrame(video_results)

results_path = os.path.join(
    OUTPUT_DIR,
    "repeated_video_level_results.csv",
)

video_results_path = os.path.join(
    OUTPUT_DIR,
    "video_wise_results.csv",
)

results_df.to_csv(results_path,index=False,)

video_results_df.to_csv(video_results_path,index=False,)

# Aggregate statistics
print("\n" + "=" * 70)
print("PHASE 16 — AGGREGATE RESULTS")
print("=" * 70)

if len(results_df) > 0:
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    for metric in metric_columns:
        mean_value = results_df[metric].mean()
        std_value = results_df[metric].std(ddof=1)
        min_value = results_df[metric].min()
        max_value = results_df[metric].max()

        print(f"\n{metric.upper()}:")
        print(f"  Mean : {mean_value * 100:.2f}%")
        print(f"  Std  : {std_value * 100:.2f}%")
        print(f"  Min  : {min_value * 100:.2f}%")
        print(f"  Max  : {max_value * 100:.2f}%")

    aggregate = []

    for metric in metric_columns:
        aggregate.append(
            {
                "metric": metric,
                "mean": results_df[metric].mean(),
                "std": results_df[metric].std(ddof=1),
                "min": results_df[metric].min(),
                "max": results_df[metric].max(),
            }
        )

    aggregate_df = pd.DataFrame(aggregate)

    aggregate_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "aggregate_statistics.csv",
        ),
        index=False,
    )

# Final status
print("\n" + "=" * 70)
print("PHASE 16 COMPLETED")
print("=" * 70)

print("\nRepeated video-level evaluation completed.")
print("Existing held-out test set was NOT modified.")
print("No sequence-level random splitting was used.")
print("No mouth/yawning features were used.")

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\n" + "=" * 70)