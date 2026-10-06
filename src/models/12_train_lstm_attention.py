import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from tensorflow.keras.callbacks import (EarlyStopping,ModelCheckpoint,)

# CONFIGURATION
TRAIN_FILE = "data/sequences/train_sequences.csv"
VAL_FILE = "data/sequences/val_sequences.csv"
TEST_FILE = "data/sequences/test_sequences.csv"

MODEL_DIR = "models"
RESULT_DIR = "results/lstm_attention"

SEQUENCE_LENGTH = 30
FEATURES = ["ear_mean", "pitch", "yaw", "roll"]
NUM_FEATURES = len(FEATURES)

SEED = 42
BATCH_SIZE = 16
EPOCHS = 50

# CREATE DIRECTORIES
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# REPRODUCIBILITY
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# HEADER
print("=" * 70)
print("LSTM + ATTENTION")
print("=" * 70)

print("\nConfiguration:")
print(f"  Sequence length : {SEQUENCE_LENGTH} frames")
print("  FPS             : 25")
print(f"  Duration        : {SEQUENCE_LENGTH / 25:.2f} seconds")
print(f"  Features        : {NUM_FEATURES}")
print(f"  Feature names   : {', '.join(FEATURES)}")
print(f"  Batch size      : {BATCH_SIZE}")
print(f"  Maximum epochs  : {EPOCHS}")
print(f"  Random seed     : {SEED}")

# 1. LOAD TEMPORAL SEQUENCES
print("\n" + "=" * 70)
print("1. LOADING TEMPORAL SEQUENCES")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train sequences      : {len(train_df)}")
print(f"Validation sequences : {len(val_df)}")
print(f"Test sequences       : {len(test_df)}")

# 2. BUILD TEMPORAL COLUMN MAPPING
print("\n" + "=" * 70)
print("2. BUILDING TEMPORAL FEATURE COLUMNS")
print("=" * 70)

feature_columns = {}

for feature in FEATURES:
    feature_columns[feature] = [
        f"{feature}_t{t:02d}"
        for t in range(1, SEQUENCE_LENGTH + 1)
    ]

print("Temporal layout:")
print("  t01 -> [ear_mean, pitch, yaw, roll]")
print("  t02 -> [ear_mean, pitch, yaw, roll]")
print("  ...")
print("  t30 -> [ear_mean, pitch, yaw, roll]")

# 3. PREPARE DATA
def prepare_data(df, split_name):
    X = np.zeros(
        (len(df), SEQUENCE_LENGTH, NUM_FEATURES),
        dtype=np.float32,
    )

    # The CSV is stored feature-wise:
    # ear_mean_t01 ... ear_mean_t30
    # pitch_t01    ... pitch_t30
    # yaw_t01      ... yaw_t30
    # roll_t01     ... roll_t30
    #
    # Reconstruct:
    # t01 -> [ear, pitch, yaw, roll]
    # t02 -> [ear, pitch, yaw, roll]
    # ...
    # t30 -> [ear, pitch, yaw, roll]

    for feature_index, feature in enumerate(FEATURES):
        columns = feature_columns[feature]
        X[:, :, feature_index] = (
            df[columns].values.astype(np.float32)
        )

    y = df["label"].values.astype(np.float32)

    print(f"\n{split_name}:")
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Not Drowsy : {np.sum(y == 0)}")
    print(f"  Drowsy     : {np.sum(y == 1)}")

    return X, y

X_train, y_train = prepare_data(train_df,"TRAIN",)

X_val, y_val = prepare_data(val_df,"VALIDATION",)

X_test, y_test = prepare_data(test_df,"TEST",)

# 3. DATA QUALITY CHECK
print("\n" + "=" * 70)
print("3. DATA QUALITY CHECK")
print("=" * 70)

print(f"Train NaN : {np.isnan(X_train).sum()}")
print(f"Val NaN   : {np.isnan(X_val).sum()}")
print(f"Test NaN  : {np.isnan(X_test).sum()}")

print(f"Train Inf : {np.isinf(X_train).sum()}")
print(f"Val Inf   : {np.isinf(X_val).sum()}")
print(f"Test Inf  : {np.isinf(X_test).sum()}")

if np.isnan(X_train).any():
    raise ValueError("NaN detected in training sequences.")

if np.isnan(X_val).any():
    raise ValueError("NaN detected in validation sequences.")

if np.isnan(X_test).any():
    raise ValueError("NaN detected in test sequences.")

if np.isinf(X_train).any():
    raise ValueError("Inf detected in training sequences.")

if np.isinf(X_val).any():
    raise ValueError("Inf detected in validation sequences.")

if np.isinf(X_test).any():
    raise ValueError("Inf detected in test sequences.")

print("\nData quality check passed.")

# 4. FEATURE STANDARDIZATION
print("\n" + "=" * 70)
print("4. FEATURE STANDARDIZATION")
print("=" * 70)

# Fit normalization statistics ONLY on TRAINING DATA.
train_2d = X_train.reshape(-1, NUM_FEATURES)

mean = train_2d.mean(axis=0)
std = train_2d.std(axis=0)
std[std == 0] = 1.0

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

print("\nTraining statistics:")

for i, feature in enumerate(FEATURES):
    print(
        f"  {feature:10s} "
        f"mean = {mean[i]:.6f}   "
        f"std = {std[i]:.6f}"
    )

# Save normalization parameters
normalization_path = os.path.join(RESULT_DIR,"normalization.npz",)

np.savez(normalization_path,mean=mean,std=std,)

print(
    f"\nNormalization parameters saved:"
    f" {normalization_path}"
)

# 5. CLASS WEIGHTS
print("\n" + "=" * 70)
print("5. CLASS WEIGHTS")
print("=" * 70)

n_not_drowsy = np.sum(y_train == 0)
n_drowsy = np.sum(y_train == 1)
n_total = len(y_train)

class_weight = {
    0: n_total / (2 * n_not_drowsy),
    1: n_total / (2 * n_drowsy),
}

print(
    f"  Not Drowsy (0): "
    f"{class_weight[0]:.4f}"
)

print(
    f"  Drowsy (1)    : "
    f"{class_weight[1]:.4f}"
)

print("\nClass weights calculated from TRAIN only.")

# 6. BUILD LSTM + SELF-ATTENTION MODEL
print("\n" + "=" * 70)
print("6. BUILDING LSTM + ATTENTION MODEL")
print("=" * 70)

# Input
inputs = tf.keras.Input(
    shape=(SEQUENCE_LENGTH, NUM_FEATURES),
    name="sequence_input",
)

# LSTM
# return_sequences=True is REQUIRED because Attention
# needs the hidden representation from every timestep.
lstm_output = tf.keras.layers.LSTM(64,return_sequences=True,name="lstm",)(inputs)

# Self-attention
# Query = LSTM output
# Value = LSTM output
#
# This allows the model to learn temporal relationships
# between different frames.
attention_output = tf.keras.layers.Attention(name="temporal_attention",)([lstm_output,lstm_output,])

# Temporal pooling
pooled = tf.keras.layers.GlobalAveragePooling1D(name="global_average_pooling",)(attention_output)

# Classification head
x = tf.keras.layers.Dropout(0.30,name="dropout_1",)(pooled)

x = tf.keras.layers.Dense(32,activation="relu",name="dense",)(x)

x = tf.keras.layers.Dropout(0.20,name="dropout_2",)(x)

outputs = tf.keras.layers.Dense(1,activation="sigmoid",name="output",)(x)

# Final model
model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="LSTM_Attention",
)

# 8. COMPILE MODEL
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy",
        ),
        tf.keras.metrics.Precision(
            name="precision",
        ),
        tf.keras.metrics.Recall(
            name="recall",
        ),
    ],
)

model.summary()

# 9. CALLBACKS
checkpoint_path = os.path.join(
    MODEL_DIR,
    "lstm_attention_best.keras",
)

callbacks = [
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    ),
]

# 10. TRAINING
print("\n" + "=" * 70)
print("7. TRAINING LSTM + ATTENTION")
print("=" * 70)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

# 11. SAVE TRAINING HISTORY
history_df = pd.DataFrame(history.history)

history_path = os.path.join(RESULT_DIR,"training_history.csv",)

history_df.to_csv(history_path,index=False,)

print(
    f"\nTraining history saved:"
    f" {history_path}"
)

# 12. ACCURACY CURVE
plt.figure(figsize=(10, 5))

plt.plot(history.history["accuracy"],label="Train Accuracy",)

plt.plot(history.history["val_accuracy"],label="Validation Accuracy",)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("LSTM + Attention — Accuracy")

plt.legend()
plt.grid(True)

accuracy_plot = os.path.join(RESULT_DIR,"accuracy_curve.png",)

plt.savefig(
    accuracy_plot,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

# 13. LOSS CURVE
plt.figure(figsize=(10, 5))

plt.plot(history.history["loss"],label="Train Loss",)

plt.plot(history.history["val_loss"],label="Validation Loss",)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("LSTM + Attention — Loss")

plt.legend()
plt.grid(True)

loss_plot = os.path.join(RESULT_DIR,"loss_curve.png",)

plt.savefig(loss_plot,dpi=200,bbox_inches="tight",)

plt.close()

# 14. TEST PREDICTION
print("\n" + "=" * 70)
print("8. TEST EVALUATION")
print("=" * 70)

test_probability = model.predict(X_test,verbose=0,).ravel()

test_prediction = (test_probability >= 0.5).astype(int)

# 15. METRICS
accuracy = accuracy_score(y_test,test_prediction,)

precision = precision_score(y_test,test_prediction,zero_division=0,)

recall = recall_score(y_test,test_prediction,zero_division=0,)

f1 = f1_score(y_test,test_prediction,zero_division=0,)

cm = confusion_matrix(y_test,test_prediction,)

print("\nTest Metrics:")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")

# 16. CONFUSION MATRIX
print("\nConfusion Matrix:")
print(cm)

# 17. CLASSIFICATION REPORT
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_prediction,
        target_names=[
            "Not Drowsy",
            "Drowsy",
        ],
        zero_division=0,
    )
)

# 18. SAVE TEST PREDICTIONS
results_df = test_df[
    [
        "sequence_id",
        "video_name",
        "split",
        "start_frame",
        "end_frame",
        "label",
    ]
].copy()

results_df["probability_drowsy"] = test_probability
results_df["predicted_label"] = test_prediction

prediction_path = os.path.join(RESULT_DIR,"test_predictions.csv",)

results_df.to_csv(prediction_path,index=False,)

# 19. SAVE METRICS
metrics_df = pd.DataFrame(
    {
        "metric": [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ],
        "value": [
            accuracy,
            precision,
            recall,
            f1,
        ],
    }
)

metrics_path = os.path.join(RESULT_DIR,"test_metrics.csv",)

metrics_df.to_csv(metrics_path,index=False,)

# 20. FINAL SUMMARY
print("\n" + "=" * 70)
print("PHASE 14 COMPLETED")
print("=" * 70)

print("\nArchitecture:")
print("  Input: 30 × 4")
print("  LSTM(64, return_sequences=True)")
print("  Self-Attention")
print("  GlobalAveragePooling1D")
print("  Dropout(0.30)")
print("  Dense(32, ReLU)")
print("  Dropout(0.20)")
print("  Dense(1, Sigmoid)")

print("\nSaved:")
print(f"  Best model       : {checkpoint_path}")
print(f"  Training history : {history_path}")
print(f"  Test predictions : {prediction_path}")
print(f"  Test metrics     : {metrics_path}")
print(f"  Accuracy plot    : {accuracy_plot}")
print(f"  Loss plot        : {loss_plot}")

print("\nBaseline reference:")
print("  LSTM accuracy  : 98.88%")
print("  LSTM precision : 91.67%")
print("  LSTM recall    : 91.67%")
print("  LSTM F1        : 91.67%")

print("\nIMPORTANT:")
print("Compare Attention against the baseline.")
print("Do not change the test set or threshold for comparison.")
print("No mouth/yawning features are used.")

print("\nNext step:")
print("Analyze LSTM vs LSTM + Attention.")