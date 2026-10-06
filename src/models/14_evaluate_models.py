import os
import random
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Config
TRAIN_FILE = "data/sequences/train_sequences.csv"
VAL_FILE = "data/sequences/val_sequences.csv"
TEST_FILE = "data/sequences/test_sequences.csv"

BASELINE_MODEL = "models/lstm_baseline_best.keras"
ATTENTION_MODEL = "models/lstm_attention_best.keras"
GRL_DIR = "models/grl_lambda_experiment"
RESULT_DIR = "results/model_evaluation"

SEQUENCE_LENGTH = 30
FEATURES = ["ear_mean", "pitch", "yaw", "roll"]
SEED = 42

# Reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load data
print("=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)
print("\nLoading sequence datasets...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train sequences:      {len(train_df)}")
print(f"Validation sequences: {len(val_df)}")
print(f"Test sequences:       {len(test_df)}")

# Feature columns
feature_columns = []

for feature in FEATURES:
    for t in range(1, SEQUENCE_LENGTH + 1):
        feature_columns.append(f"{feature}_t{t:02d}")

# Prepare sequences
def prepare_sequences(df):
    X = np.zeros(
        (len(df), SEQUENCE_LENGTH, len(FEATURES)),
        dtype=np.float32,
    )

    for feature_index, feature in enumerate(FEATURES):
        columns = [
            f"{feature}_t{t:02d}"
            for t in range(1, SEQUENCE_LENGTH + 1)
        ]
        X[:, :, feature_index] = df[columns].values.astype(np.float32)

    y = df["label"].values.astype(np.int32)

    return X, y


X_train, y_train = prepare_sequences(train_df)
X_val, y_val = prepare_sequences(val_df)
X_test, y_test = prepare_sequences(test_df)

print("\nSequence shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"X_test:  {X_test.shape}")

# Train-only standardization
print("\nCalculating train-only normalization statistics...")

# Mean/std are calculated feature-wise across all training sequences and all 30 timesteps.
train_mean = np.mean(X_train, axis=(0, 1))
train_std = np.std(X_train, axis=(0, 1))
train_std[train_std == 0] = 1.0

print("\nTrain normalization statistics:")

for feature, mean, std in zip(FEATURES, train_mean, train_std):
    print(f"{feature:10s} mean={mean:.6f}  std={std:.6f}")

# Apply train-only standardization
X_train = (X_train - train_mean.reshape(1, 1, -1)) / train_std.reshape(1, 1, -1)
X_val = (X_val - train_mean.reshape(1, 1, -1)) / train_std.reshape(1, 1, -1)
X_test = (X_test - train_mean.reshape(1, 1, -1)) / train_std.reshape(1, 1, -1)

# GRL custom layer
@tf.keras.utils.register_keras_serializable()
class GradientReversal(tf.keras.layers.Layer):

    def __init__(self, lambda_value=0.1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_value = lambda_value

    def call(self, x):
        lambda_value = self.lambda_value

        @tf.custom_gradient
        def reverse_gradient(x):
            def grad(dy):
                return -lambda_value * dy

            return x, grad

        return reverse_gradient(x)

    def get_config(self):
        config = super().get_config()
        config.update({"lambda_value": self.lambda_value})
        return config

# Find GRL model
def find_grl_model():
    if not os.path.exists(GRL_DIR):
        return None

    keras_files = glob.glob(os.path.join(GRL_DIR, "*.keras"))

    if not keras_files:
        return None

    # Prefer the lambda 0.10 model
    preferred = [
        f
        for f in keras_files
        if "0.10" in os.path.basename(f)
        or "lambda_0.10" in os.path.basename(f)
    ]

    if preferred:
        return preferred[0]

    print("\nGRL models found:")

    for f in keras_files:
        print(" -", os.path.basename(f))

    return keras_files[0]


GRL_MODEL = find_grl_model()

if GRL_MODEL:
    print(f"\nGRL model selected:\n{GRL_MODEL}")
else:
    print(f"\nNo GRL model found in:\n{GRL_DIR}")

# Evaluation function
def evaluate_model(model_name, model_path, grl=False):
    print("\n" + "=" * 70)
    print(f"EVALUATING: {model_name}")
    print("=" * 70)

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found:\n{model_path}")
        return None

    print(f"Loading model:\n{model_path}")

    if grl:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"GradientReversal": GradientReversal},
            compile=False,
        )
    else:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

    print("Model loaded successfully.")

    # Prediction
    predictions = model.predict(X_test, verbose=0)

    # GRL model has:
    # output 0 = drowsiness
    # output 1 = domain
    if isinstance(predictions, list):
        probability = np.asarray(predictions[0]).reshape(-1)
    else:
        probability = np.asarray(predictions).reshape(-1)

    predicted_label = (probability >= 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, predicted_label)
    precision = precision_score(y_test, predicted_label, zero_division=0)
    recall = recall_score(y_test, predicted_label, zero_division=0)
    f1 = f1_score(y_test, predicted_label, zero_division=0)
    roc_auc = roc_auc_score(y_test, probability)
    cm = confusion_matrix(y_test, predicted_label)

    # Print results
    print("\nTest Results")
    print("-" * 50)

    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1 Score : {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"ROC-AUC  : {roc_auc:.4f} ({roc_auc * 100:.2f}%)")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predicted_label,
            target_names=["Not Drowsy", "Drowsy"],
            zero_division=0,
        )
    )

    # Save predictions
    prediction_df = test_df[
        [
            "sequence_id",
            "video_name",
            "split",
            "start_frame",
            "end_frame",
            "label",
        ]
    ].copy()

    prediction_df["probability_drowsy"] = probability
    prediction_df["predicted_label"] = predicted_label

    safe_name = model_name.lower().replace(" ", "_")

    prediction_file = os.path.join(
        RESULT_DIR,
        f"{safe_name}_predictions.csv",
    )

    prediction_df.to_csv(prediction_file, index=False)

    print(f"Predictions saved:\n{prediction_file}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    }

# Output directory
os.makedirs(RESULT_DIR, exist_ok=True)

# Evaluate models
results = []

result = evaluate_model(
    "LSTM_Baseline",
    BASELINE_MODEL,
)

if result:
    results.append(result)

result = evaluate_model(
    "LSTM_Attention",
    ATTENTION_MODEL,
)

if result:
    results.append(result)

if GRL_MODEL:
    result = evaluate_model(
        "LSTM_GRL",
        GRL_MODEL,
        grl=True,
    )

    if result:
        results.append(result)

# Save final metrics
results_df = pd.DataFrame(results)

results_file = os.path.join(
    RESULT_DIR,
    "test_metrics.csv",
)

results_df.to_csv(results_file, index=False)

# Final summary
print("\n" + "=" * 70)
print("FINAL TEST COMPARISON")
print("=" * 70)

print(
    results_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]
    ].to_string(index=False)
)

print("\nConfusion Matrices")
print("-" * 50)

for result in results:
    print(f"\n{result['model']}")
    print(
        f"TN={result['tn']}  "
        f"FP={result['fp']}  "
        f"FN={result['fn']}  "
        f"TP={result['tp']}"
    )

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(f"\nResults saved in:\n{RESULT_DIR}")