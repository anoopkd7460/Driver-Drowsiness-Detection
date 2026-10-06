from pathlib import Path
import random
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
)
from sklearn.utils.class_weight import compute_class_weight

# REPRODUCIBILITY
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_DIR = PROJECT_ROOT / "data" / "sequences"
RESULT_DIR = PROJECT_ROOT / "results" / "grl_lambda_experiment"
MODEL_DIR = PROJECT_ROOT / "models" / "grl_lambda_experiment"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SEQUENCE_DIR / "train_sequences.csv"
VAL_FILE = SEQUENCE_DIR / "val_sequences.csv"
TEST_FILE = SEQUENCE_DIR / "test_sequences.csv"

# CONFIGURATION
SEQUENCE_LENGTH = 30
FEATURES = ["ear_mean", "pitch", "yaw", "roll"]
LSTM_UNITS = 64
DROPOUT = 0.30
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 8

# Controlled GRL experiment
LAMBDA_VALUES = [0.01, 0.05, 0.10, 0.20]

# LOAD DATA
print("=" * 70)
print("CONTROLLED GRL EXPERIMENT")
print("=" * 70)
print("\nLoading sequence files...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train sequences : {len(train_df)}")
print(f"Val sequences   : {len(val_df)}")
print(f"Test sequences  : {len(test_df)}")

# FEATURE COLUMNS
feature_columns = []
for feature in FEATURES:
    for t in range(1, SEQUENCE_LENGTH + 1):
        feature_columns.append(f"{feature}_t{t:02d}")

# CONVERT DATAFRAME
def dataframe_to_arrays(dataframe):
    X = dataframe[feature_columns].values.astype(np.float32)
    y = dataframe["label"].values.astype(np.float32)
    videos = dataframe["video_name"].values
    return X, y, videos

X_train_flat, y_train, videos_train = dataframe_to_arrays(train_df)
X_val_flat, y_val, videos_val = dataframe_to_arrays(val_df)
X_test_flat, y_test, videos_test = dataframe_to_arrays(test_df)

# RESHAPE
def reshape_sequences(X):
    n = X.shape[0]
    result = np.zeros(
        (n, SEQUENCE_LENGTH, len(FEATURES)),
        dtype=np.float32,
    )

    for feature_index in range(len(FEATURES)):
        start = feature_index * SEQUENCE_LENGTH
        end = start + SEQUENCE_LENGTH
        result[:, :, feature_index] = X[:, start:end]

    return result

X_train = reshape_sequences(X_train_flat)
X_val = reshape_sequences(X_val_flat)
X_test = reshape_sequences(X_test_flat)

print("\nInput shapes:")
print(f"Train: {X_train.shape}")
print(f"Val  : {X_val.shape}")
print(f"Test : {X_test.shape}")

# STANDARDIZATION
print("\nStandardizing using training data only...")

train_means = np.mean(X_train, axis=(0, 1))
train_stds = np.std(X_train, axis=(0, 1))
train_stds[train_stds == 0] = 1.0

X_train = (X_train - train_means) / train_stds
X_val = (X_val - train_means) / train_stds
X_test = (X_test - train_means) / train_stds

# DOMAIN LABELS
print("\nCreating training video-domain labels...")

training_videos = sorted(np.unique(videos_train))

domain_to_id = {
    video: i
    for i, video in enumerate(training_videos)
}

id_to_domain = {
    i: video
    for video, i in domain_to_id.items()
}

domain_train = np.array(
    [domain_to_id[v] for v in videos_train],
    dtype=np.int32,
)

num_domains = len(training_videos)

print(f"Number of training domains: {num_domains}")

# CLASS WEIGHTS
classes = np.array([0, 1])

class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train.astype(int),
)

CLASS_WEIGHT_0 = float(class_weight_values[0])
CLASS_WEIGHT_1 = float(class_weight_values[1])

print("\nClass weights:")
print(f"Not Drowsy: {CLASS_WEIGHT_0:.4f}")
print(f"Drowsy    : {CLASS_WEIGHT_1:.4f}")

# GRL
@tf.custom_gradient
def gradient_reversal(x, lambda_value):
    def grad(dy):
        return -lambda_value * dy, None

    return x, grad


class GradientReversal(tf.keras.layers.Layer):
    def __init__(self, lambda_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_value = float(lambda_value)

    def call(self, inputs):
        return gradient_reversal(
            inputs,
            tf.cast(self.lambda_value, inputs.dtype),
        )

    def get_config(self):
        config = super().get_config()
        config.update({"lambda_value": self.lambda_value})
        return config

# MODEL BUILDER
def build_model(lambda_value):
    inputs = tf.keras.Input(
        shape=(SEQUENCE_LENGTH, len(FEATURES)),
        name="sequence_input",
    )

    # Shared feature extractor
    features = tf.keras.layers.LSTM(
        LSTM_UNITS,
        name="feature_lstm",
    )(inputs)

    features = tf.keras.layers.Dropout(
        DROPOUT,
        name="feature_dropout",
    )(features)

    # Drowsiness head
    drowsiness = tf.keras.layers.Dense(
        32,
        activation="relu",
        name="drowsiness_dense",
    )(features)

    drowsiness = tf.keras.layers.Dropout(
        0.20,
        name="drowsiness_dropout",
    )(drowsiness)

    drowsiness_output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="drowsiness_output",
    )(drowsiness)

    # Domain head
    reversed_features = GradientReversal(
        lambda_value=lambda_value,
        name="gradient_reversal",
    )(features)

    domain = tf.keras.layers.Dense(
        32,
        activation="relu",
        name="domain_dense",
    )(reversed_features)

    domain = tf.keras.layers.Dropout(
        0.20,
        name="domain_dropout",
    )(domain)

    domain_output = tf.keras.layers.Dense(
        num_domains,
        activation="softmax",
        name="domain_output",
    )(domain)

    return tf.keras.Model(
        inputs=inputs,
        outputs=[drowsiness_output, domain_output],
    )

# LOSSES
def weighted_binary_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    base_loss = tf.keras.backend.binary_crossentropy(
        y_true,
        y_pred,
    )

    weights = (
        y_true * CLASS_WEIGHT_1
        + (1.0 - y_true) * CLASS_WEIGHT_0
    )

    return tf.reduce_mean(base_loss * weights)

# TRAIN ONE MODEL
def train_one_lambda(lambda_value):
    print("\n")
    print("=" * 70)
    print(f"TRAINING GRL λ = {lambda_value}")
    print("=" * 70)

    # Reproducibility for each run
    tf.keras.backend.clear_session()
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    model = build_model(lambda_value)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
    )

    domain_loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

    # TRAIN STEP
    @tf.function
    def train_step(x, y_class, y_domain):
        with tf.GradientTape() as tape:
            class_pred, domain_pred = model(
                x,
                training=True,
            )

            class_loss = weighted_binary_loss(
                y_class,
                class_pred,
            )

            domain_loss = domain_loss_function(
                y_domain,
                domain_pred,
            )

            total_loss = class_loss + domain_loss

        gradients = tape.gradient(
            total_loss,
            model.trainable_variables,
        )

        optimizer.apply_gradients(
            zip(
                gradients,
                model.trainable_variables,
            )
        )

        return total_loss, class_loss, domain_loss

    # VALIDATION
    def validation_metrics():
        class_pred, domain_pred = model(
            X_val,
            training=False,
        )

        probabilities = class_pred.numpy().reshape(-1)
        predictions = (probabilities >= 0.5).astype(int)

        class_loss = float(
            weighted_binary_loss(
                tf.constant(y_val),
                class_pred,
            ).numpy()
        )

        accuracy = accuracy_score(
            y_val,
            predictions,
        )

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        return (
            class_loss,
            accuracy,
            precision,
            recall,
            f1,
        )

    # TRAINING LOOP
    best_val_f1 = -np.inf
    best_val_loss = np.inf
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        indices = np.random.permutation(len(X_train))

        total_losses = []
        class_losses = []
        domain_losses = []

        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start:start + BATCH_SIZE]

            x_batch = X_train[batch_indices]
            y_class_batch = y_train[batch_indices]
            y_domain_batch = domain_train[batch_indices]

            loss_values = train_step(
                x_batch,
                y_class_batch,
                y_domain_batch,
            )

            total_losses.append(
                float(loss_values[0].numpy())
            )

            class_losses.append(
                float(loss_values[1].numpy())
            )

            domain_losses.append(
                float(loss_values[2].numpy())
            )

        train_total_loss = np.mean(total_losses)
        train_class_loss = np.mean(class_losses)
        train_domain_loss = np.mean(domain_losses)

        (
            val_class_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
        ) = validation_metrics()

        history.append(
            {
                "lambda": lambda_value,
                "epoch": epoch,
                "train_total_loss": train_total_loss,
                "train_class_loss": train_class_loss,
                "train_domain_loss": train_domain_loss,
                "val_class_loss": val_class_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train {train_total_loss:.4f} | "
            f"Class {train_class_loss:.4f} | "
            f"Domain {train_domain_loss:.4f} | "
            f"Val Acc {val_accuracy:.4f} | "
            f"Val F1 {val_f1:.4f}"
        )

        # MODEL SELECTION
        # Primary criterion: validation F1
        # Secondary criterion: validation loss
        improved = (
            val_f1 > best_val_f1
            or (
                np.isclose(val_f1, best_val_f1)
                and val_class_loss < best_val_loss
            )
        )

        if improved:
            best_val_f1 = val_f1
            best_val_loss = val_class_loss
            best_epoch = epoch
            patience_counter = 0

            model_path = (
                MODEL_DIR
                / f"lstm_grl_lambda_{lambda_value:.2f}.keras"
            )

            model.save(model_path)

            print("  -> Best validation model saved.")

        else:
            patience_counter += 1

            if patience_counter >= PATIENCE:
                print("  -> Early stopping.")
                break

    # SAVE HISTORY
    history_df = pd.DataFrame(history)

    history_df.to_csv(
        RESULT_DIR / f"history_lambda_{lambda_value:.2f}.csv",
        index=False,
    )

    return {
        "lambda": lambda_value,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "model_path": str(
            MODEL_DIR / f"lstm_grl_lambda_{lambda_value:.2f}.keras"
        ),
    }

# RUN ALL LAMBDA VALUES
all_results = []

for lambda_value in LAMBDA_VALUES:
    result = train_one_lambda(lambda_value)
    all_results.append(result)

# VALIDATION COMPARISON
results_df = pd.DataFrame(all_results)

results_df = (
    results_df
    .sort_values(
        ["best_val_f1", "best_val_loss"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)

print("\n")
print("=" * 70)
print("GRL VALIDATION COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

results_df.to_csv(
    RESULT_DIR / "lambda_validation_comparison.csv",
    index=False,
)

# SELECT BEST LAMBDA
best_result = results_df.iloc[0]
best_lambda = float(best_result["lambda"])

best_model_path = Path(
    best_result["model_path"]
)

print("\n" + "=" * 70)
print("SELECTED MODEL")
print("=" * 70)

print(f"Selected λ       : {best_lambda}")
print(
    f"Best Val F1      : "
    f"{best_result['best_val_f1']:.4f}"
)
print(
    f"Best Val Loss    : "
    f"{best_result['best_val_loss']:.6f}"
)
print(
    f"Best Epoch       : "
    f"{int(best_result['best_epoch'])}"
)

# LOAD SELECTED MODEL
print("\nLoading selected model...")

best_model = tf.keras.models.load_model(
    best_model_path,
    custom_objects={
        "GradientReversal": GradientReversal
    },
)

# OFFICIAL TEST
print("\n")
print("=" * 70)
print("OFFICIAL TEST EVALUATION")
print("=" * 70)

print(
    "\nThe test set is being used for final evaluation only."
)

test_class_output, test_domain_output = best_model(
    X_test,
    training=False,
)

test_probabilities = (
    test_class_output.numpy().reshape(-1)
)

test_predictions = (
    test_probabilities >= 0.5
).astype(int)

# TEST METRICS
test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_auc = roc_auc_score(
    y_test,
    test_probabilities,
)

cm = confusion_matrix(
    y_test,
    test_predictions,
)

print(f"\nAccuracy  : {test_accuracy:.4f}")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"F1-score  : {test_f1:.4f}")
print(f"ROC-AUC   : {test_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

# TEST PREDICTIONS
test_predictions_df = pd.DataFrame(
    {
        "sequence_id": test_df["sequence_id"],
        "video_name": test_df["video_name"],
        "split": test_df["split"],
        "start_frame": test_df["start_frame"],
        "end_frame": test_df["end_frame"],
        "label": y_test.astype(int),
        "probability_drowsy": test_probabilities,
        "predicted_label": test_predictions,
    }
)

test_predictions_df.to_csv(
    RESULT_DIR / "test_predictions.csv",
    index=False,
)

# VIDEO-WISE TEST RESULTS
video_results = []

for video in sorted(test_df["video_name"].unique()):
    mask = test_df["video_name"].values == video

    true_values = y_test[mask]
    predicted_values = test_predictions[mask]

    video_accuracy = accuracy_score(
        true_values,
        predicted_values,
    )

    video_precision = precision_score(
        true_values,
        predicted_values,
        zero_division=0,
    )

    video_recall = recall_score(
        true_values,
        predicted_values,
        zero_division=0,
    )

    video_f1 = f1_score(
        true_values,
        predicted_values,
        zero_division=0,
    )

    video_results.append(
        {
            "video_name": video,
            "sequences": int(np.sum(mask)),
            "drowsy_sequences": int(
                np.sum(true_values == 1)
            ),
            "accuracy": video_accuracy,
            "precision": video_precision,
            "recall": video_recall,
            "f1": video_f1,
        }
    )

video_results_df = pd.DataFrame(video_results)

print("\n" + "-" * 70)
print("VIDEO-WISE TEST RESULTS")
print("-" * 70)

print(
    video_results_df.to_string(index=False)
)

video_results_df.to_csv(
    RESULT_DIR / "video_wise_test_results.csv",
    index=False,
)

# FINAL METRICS FILE
final_metrics = pd.DataFrame(
    [
        {
            "model": "LSTM_GRL",
            "selected_lambda": best_lambda,
            "best_validation_f1": float(
                best_result["best_val_f1"]
            ),
            "best_validation_loss": float(
                best_result["best_val_loss"]
            ),
            "best_epoch": int(
                best_result["best_epoch"]
            ),
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "test_roc_auc": test_auc,
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        }
    ]
)

final_metrics.to_csv(
    RESULT_DIR / "final_test_metrics.csv",
    index=False,
)

# FINAL OUTPUT
print("\n")
print("=" * 70)
print("EXPERIMENT FINISHED")
print("=" * 70)

print(f"\nSelected λ: {best_lambda}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall   : {test_recall:.4f}")
print(f"Test F1       : {test_f1:.4f}")
print(f"Test ROC-AUC  : {test_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nResults saved to:")
print(RESULT_DIR)

print("\nModels saved to:")
print(MODEL_DIR)