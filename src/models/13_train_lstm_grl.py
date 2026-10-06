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
RESULT_DIR = PROJECT_ROOT / "results" / "lstm_grl"
MODEL_DIR = PROJECT_ROOT / "models"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SEQUENCE_DIR / "train_sequences.csv"
VAL_FILE = SEQUENCE_DIR / "val_sequences.csv"
TEST_FILE = SEQUENCE_DIR / "test_sequences.csv"

# CONFIG
SEQUENCE_LENGTH = 30
FEATURES = ["ear_mean", "pitch", "yaw", "roll"]

LSTM_UNITS = 64
DROPOUT = 0.30
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 8
DOMAIN_LOSS_WEIGHT = 0.10

# LOAD DATA
print("=" * 70)
print("LSTM + GRL TRAINING")
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

# BUILD INPUT ARRAYS
def dataframe_to_sequences(dataframe):
    X = dataframe[feature_columns].values.astype(np.float32)
    y = dataframe["label"].values.astype(np.float32)
    videos = dataframe["video_name"].values
    return X, y, videos

X_train, y_train, videos_train = dataframe_to_sequences(train_df)
X_val, y_val, videos_val = dataframe_to_sequences(val_df)
X_test, y_test, videos_test = dataframe_to_sequences(test_df)

# RESHAPE
# Current flattened order:
# ear_mean_t01 ... ear_mean_t30
# pitch_t01 ... pitch_t30
# yaw_t01 ... yaw_t30
# roll_t01 ... roll_t30
#
# Convert to:
# samples × 30 × 4

def reshape_sequences(X):
    n = X.shape[0]

    reshaped = np.zeros(
        (n, SEQUENCE_LENGTH, len(FEATURES)),
        dtype=np.float32,
    )

    for feature_index in range(len(FEATURES)):
        start = feature_index * SEQUENCE_LENGTH
        end = start + SEQUENCE_LENGTH
        reshaped[:, :, feature_index] = X[:, start:end]

    return reshaped

X_train = reshape_sequences(X_train)
X_val = reshape_sequences(X_val)
X_test = reshape_sequences(X_test)

print("\nInput shape:")
print(f"Train: {X_train.shape}")
print(f"Val  : {X_val.shape}")
print(f"Test : {X_test.shape}")

# STANDARDIZATION
print("\nFitting standardization using TRAIN ONLY...")

train_means = np.mean(X_train,axis=(0, 1),)

train_stds = np.std(X_train,axis=(0, 1),)

train_stds[train_stds == 0] = 1.0

X_train = (X_train - train_means) / train_stds
X_val = (X_val - train_means) / train_stds
X_test = (X_test - train_means) / train_stds

print("\nTraining statistics:")

for i, feature in enumerate(FEATURES):
    print(
        f"{feature:<10} "
        f"mean={train_means[i]:.6f} "
        f"std={train_stds[i]:.6f}"
    )

# DOMAIN LABELS
print("\nCreating video-domain labels...")

# Domain labels are fitted ONLY from training videos.
train_video_names = sorted(np.unique(videos_train))

domain_to_id = {
    video: index
    for index, video in enumerate(train_video_names)
}

id_to_domain = {
    index: video
    for video, index in domain_to_id.items()
}

def encode_training_domains(videos):
    return np.array(
        [domain_to_id[v] for v in videos],
        dtype=np.int32,
    )

domain_train = encode_training_domains(videos_train)

domain_val = np.array(
    [
        domain_to_id[v] if v in domain_to_id else -1
        for v in videos_val
    ],
    dtype=np.int32,
)

num_domains = len(train_video_names)

print(f"Training domains: {num_domains}")

for domain_id, video in id_to_domain.items():
    count = np.sum(domain_train == domain_id)

    print(
        f"  {domain_id:2d} "
        f"{video:<20} "
        f"{count:3d} sequences"
    )

# GRL LAYER
@tf.custom_gradient
def gradient_reversal(x, lambda_value):
    def grad(dy):
        return -lambda_value * dy, None

    return x, grad


class GradientReversal(tf.keras.layers.Layer):
    def __init__(self, lambda_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_value = lambda_value

    def call(self, inputs):
        return gradient_reversal(
            inputs,
            tf.cast(
                self.lambda_value,
                inputs.dtype,
            ),
        )

# MODEL
print("\nBuilding GRL model...")

sequence_input = tf.keras.Input(
    shape=(
        SEQUENCE_LENGTH,
        len(FEATURES),
    ),
    name="sequence_input",
)

# Feature extractor
feature_representation = tf.keras.layers.LSTM(LSTM_UNITS,name="feature_lstm",)(sequence_input)

feature_representation = tf.keras.layers.Dropout(DROPOUT)(feature_representation)

# Drowsiness classifier
drowsiness = tf.keras.layers.Dense(32,activation="relu",name="drowsiness_dense",)(feature_representation)

drowsiness = tf.keras.layers.Dropout(0.20)(drowsiness)

drowsiness_output = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="drowsiness_output",
)(drowsiness)

# Domain classifier
reversed_features = GradientReversal(
    lambda_value=DOMAIN_LOSS_WEIGHT,
    name="gradient_reversal",
)(feature_representation)

domain = tf.keras.layers.Dense(
    32,
    activation="relu",
    name="domain_dense",
)(reversed_features)

domain = tf.keras.layers.Dropout(
    0.20
)(domain)

domain_output = tf.keras.layers.Dense(
    num_domains,
    activation="softmax",
    name="domain_output",
)(domain)

# COMPLETE MODEL
model = tf.keras.Model(
    inputs=sequence_input,
    outputs=[
        drowsiness_output,
        domain_output,
    ],
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
    ),
    loss={
        "drowsiness_output":
            tf.keras.losses.BinaryCrossentropy(),
        "domain_output":
            tf.keras.losses.SparseCategoricalCrossentropy(),
    },
    loss_weights={
        "drowsiness_output": 1.0,
        "domain_output": 1.0,
    },
    metrics={
        "drowsiness_output": [
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
        "domain_output": [
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy",
            ),
        ],
    },
)

print("\nModel summary:\n")
model.summary()

# CLASS WEIGHTS
classes = np.array([0, 1])

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train.astype(int),
)

class_weights = {
    0: float(class_weights_array[0]),
    1: float(class_weights_array[1]),
}

print("\nDrowsiness class weights:")
print(f"Not Drowsy: {class_weights[0]:.4f}")
print(f"Drowsy    : {class_weights[1]:.4f}")

# CUSTOM TRAINING LOOP
optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

bce = tf.keras.losses.BinaryCrossentropy()
sce = tf.keras.losses.SparseCategoricalCrossentropy()

def weighted_binary_loss(y_true, y_pred):
    y_true = tf.cast(y_true,tf.float32,)

    y_pred = tf.cast(y_pred,tf.float32,)

    base_loss = tf.keras.backend.binary_crossentropy(y_true,y_pred,)

    weights = (
        y_true * class_weights[1]
        + (1.0 - y_true) * class_weights[0]
    )

    return tf.reduce_mean(base_loss * weights)


@tf.function
def train_step(x, y_class, y_domain):
    with tf.GradientTape() as tape:
        class_pred, domain_pred = model(x,training=True,)

        class_loss = weighted_binary_loss(y_class,class_pred,)

        domain_loss = sce(y_domain,domain_pred,)

        total_loss = class_loss + domain_loss

    gradients = tape.gradient(total_loss,model.trainable_variables,)

    optimizer.apply_gradients(
        zip(
            gradients,
            model.trainable_variables,
        )
    )

    return (
        total_loss,
        class_loss,
        domain_loss,
    )

# VALIDATION FUNCTION
def evaluate_validation():
    class_pred, domain_pred = model(X_val,training=False,)

    probabilities = (class_pred.numpy().reshape(-1))

    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_val,predictions,)

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
        accuracy,
        precision,
        recall,
        f1,
    )

# TRAINING
print("\nStarting training...")

best_val_loss = np.inf
best_epoch = 0
patience_counter = 0
history = []

for epoch in range(1, EPOCHS + 1):
    # Shuffle training data
    indices = np.random.permutation(
        len(X_train)
    )

    epoch_losses = []
    epoch_class_losses = []
    epoch_domain_losses = []

    for start in range(0,len(indices),BATCH_SIZE,):
        batch_indices = indices[
            start:start + BATCH_SIZE
        ]

        x_batch = X_train[batch_indices]
        y_class_batch = y_train[batch_indices]
        y_domain_batch = domain_train[batch_indices]

        losses = train_step(
            x_batch,
            y_class_batch,
            y_domain_batch,
        )

        total_loss = float(losses[0].numpy())

        class_loss = float(losses[1].numpy())

        domain_loss = float(losses[2].numpy())

        epoch_losses.append(total_loss)
        epoch_class_losses.append(class_loss)
        epoch_domain_losses.append(domain_loss)

    # Training loss
    train_total_loss = np.mean(epoch_losses)

    train_class_loss = np.mean(epoch_class_losses)

    train_domain_loss = np.mean(epoch_domain_losses)

    # Validation
    class_pred_val, domain_pred_val = model(X_val,training=False,)

    val_class_prob = (class_pred_val.numpy().reshape(-1))

    val_domain_prob = domain_pred_val.numpy()

    val_class_loss = float(
        weighted_binary_loss(
            tf.constant(y_val),
            class_pred_val,
        ).numpy()
    )

    # Domain validation only for videos that
    # are also known training domains.
    known_mask = domain_val >= 0

    if np.any(known_mask):
        val_domain_loss = float(
            sce(
                tf.constant(
                    domain_val[known_mask]
                ),
                tf.constant(
                    val_domain_prob[known_mask]
                ),
            ).numpy()
        )
    else:
        val_domain_loss = np.nan

    val_total_loss = (
        val_class_loss
        + (
            1.0 * val_domain_loss
            if not np.isnan(val_domain_loss)
            else 0.0
        )
    )

    val_predictions = (val_class_prob >= 0.5).astype(int)

    val_accuracy = accuracy_score(y_val,val_predictions,)

    val_precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    val_recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    history.append(
        {
            "epoch": epoch,
            "train_total_loss": train_total_loss,
            "train_class_loss": train_class_loss,
            "train_domain_loss": train_domain_loss,
            "val_total_loss": val_total_loss,
            "val_class_loss": val_class_loss,
            "val_domain_loss": val_domain_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
        }
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss {train_total_loss:.4f} | "
        f"Class {train_class_loss:.4f} | "
        f"Domain {train_domain_loss:.4f} | "
        f"Val Loss {val_total_loss:.4f} | "
        f"Val Acc {val_accuracy:.4f} | "
        f"Val F1 {val_f1:.4f}"
    )

    # EARLY STOPPING
    if val_total_loss < best_val_loss:
        best_val_loss = val_total_loss
        best_epoch = epoch
        patience_counter = 0

        model.save(
            MODEL_DIR / "lstm_grl_best.keras"
        )

        print("  -> Best model saved.")

    else:
        patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping.")
            break

# SAVE HISTORY
history_df = pd.DataFrame(history)

history_df.to_csv(
    RESULT_DIR / "training_history.csv",
    index=False,
)

# LOAD BEST MODEL
print(f"\nBest epoch: {best_epoch}")
print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print("\nLoading best model...")

best_model = tf.keras.models.load_model(
    MODEL_DIR / "lstm_grl_best.keras",
    custom_objects={
        "GradientReversal": GradientReversal,
    },
)

# OFFICIAL TEST EVALUATION
print("\n" + "=" * 70)
print("OFFICIAL TEST EVALUATION")
print("=" * 70)

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

# METRICS
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
predictions_df = pd.DataFrame(
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

predictions_df.to_csv(
    RESULT_DIR / "test_predictions.csv",
    index=False,
)

# VIDEO-WISE TEST PERFORMANCE
print("\n" + "-" * 70)
print("VIDEO-WISE TEST PERFORMANCE")
print("-" * 70)

video_results = []

for video in sorted(
    test_df["video_name"].unique()
):
    mask = (
        test_df["video_name"].values == video
    )

    true_values = y_test[mask]
    predicted_values = test_predictions[mask]

    video_accuracy = accuracy_score(
        true_values,
        predicted_values,
    )

    video_recall = recall_score(
        true_values,
        predicted_values,
        zero_division=0,
    )

    video_precision = precision_score(
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

print(video_results_df.to_string(index=False))

video_results_df.to_csv(RESULT_DIR / "video_wise_test_results.csv",index=False,)

# SAVE METRICS
metrics_df = pd.DataFrame([{
            "model": "LSTM_GRL",
            "best_epoch": best_epoch,
            "domain_loss_weight": DOMAIN_LOSS_WEIGHT,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "test_roc_auc": test_auc,
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        }])

metrics_df.to_csv(RESULT_DIR / "test_metrics.csv",index=False,)

# SAVE DOMAIN MAPPING
domain_mapping_df = pd.DataFrame([
        {
            "domain_id": domain_id,
            "video_name": video,
        }
        for domain_id, video in id_to_domain.items()
    ])

domain_mapping_df.to_csv(RESULT_DIR / "training_domain_mapping.csv",index=False,)

# FINAL SUMMARY
print("\n" + "=" * 70)
print("GRL EXPERIMENT COMPLETED")
print("=" * 70)

print("\nBest epoch:", best_epoch)

print("\nOfficial test results:")
print(f"Accuracy  : {test_accuracy:.4f}")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"F1-score  : {test_f1:.4f}")
print(f"ROC-AUC   : {test_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nSaved model:")
print(MODEL_DIR / "lstm_grl_best.keras")

print("\nSaved results:")
print(RESULT_DIR)