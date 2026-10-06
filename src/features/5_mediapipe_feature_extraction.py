"""
MEDIAPIPE FEATURE EXTRACTION

Purpose:
    Extract frame-level eye and head movement features from NITYMED.

Features:
    Eye:
        - EAR left
        - EAR right
        - EAR mean
        - Eye openness left
        - Eye openness right
        - Eye openness mean
        - Eye validity

    Head:
        - Pitch
        - Yaw
        - Roll

Rules:
    - NO mouth/yawning features.
    - Missing images -> NaN features.
    - No face -> NaN features.
    - Unreliable head pose -> NaN.
    - Checkpoint/resume enabled.

Input:
    data/features/annotation_metadata.csv

Model:
    src/models/face_landmarker.task

Output:
    data/features/frame_features.csv
"""

from pathlib import Path
import math

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# PATHS
ANNOTATION_FILE = Path("data/features/annotation_metadata.csv")
FRAME_ROOT = Path("dataset/NITYMED/classification_frames")
MODEL_FILE = Path("src/models/face_landmarker.task")
OUTPUT_FILE = Path("data/features/frame_features.csv")
CHECKPOINT_FILE = Path("data/features/frame_features_checkpoint.csv")

# LANDMARK INDICES
# Eye landmarks
LEFT_EYE = [
    33,    # outer corner
    160,   # upper eyelid
    158,   # upper eyelid
    133,   # inner corner
    153,   # lower eyelid
    144,   # lower eyelid
]

RIGHT_EYE = [
    362,   # outer corner
    385,   # upper eyelid
    387,   # upper eyelid
    263,   # inner corner
    373,   # lower eyelid
    380,   # lower eyelid
]

# Head-pose landmarks

# IMPORTANT:
# NO mouth landmarks are used.
HEAD_POINTS = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_cheek": 234,
    "right_cheek": 454,
}

# UTILITY FUNCTIONS
def distance(p1, p2):
    """
    Euclidean distance between two 2D points.
    """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# EAR
def calculate_ear(landmarks, indices):
    """
    Calculate Eye Aspect Ratio.

    EAR =
        (vertical_1 + vertical_2)
        / (2 * horizontal)
    """
    try:
        p1 = landmarks[indices[0]]
        p2 = landmarks[indices[1]]
        p3 = landmarks[indices[2]]
        p4 = landmarks[indices[3]]
        p5 = landmarks[indices[4]]
        p6 = landmarks[indices[5]]

        # First vertical distance
        vertical_1 = distance(
            (p2.x, p2.y),
            (p6.x, p6.y)
        )

        # Second vertical distance
        vertical_2 = distance(
            (p3.x, p3.y),
            (p5.x, p5.y)
        )

        # Eye width
        horizontal = distance(
            (p1.x, p1.y),
            (p4.x, p4.y)
        )

        if horizontal <= 1e-8:
            return np.nan

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

        if not np.isfinite(ear):
            return np.nan

        return float(ear)

    except Exception:
        return np.nan


# EYE OPENNESS
def calculate_eye_openness(landmarks, indices):
    """
    Calculate normalized vertical eye aperture.

    Unlike EAR, this uses:

        distance(
            upper-eyelid midpoint,
            lower-eyelid midpoint
        )

    normalized by eye-corner distance.
    """
    try:
        outer = landmarks[indices[0]]
        upper_1 = landmarks[indices[1]]
        upper_2 = landmarks[indices[2]]
        inner = landmarks[indices[3]]
        lower_1 = landmarks[indices[4]]
        lower_2 = landmarks[indices[5]]

        # Eye width
        eye_width = distance((outer.x, outer.y), (inner.x, inner.y))

        if eye_width <= 1e-8:
            return np.nan

        # Upper eyelid midpoint
        upper_mid = (
            (upper_1.x + upper_2.x) / 2.0,
            (upper_1.y + upper_2.y) / 2.0
        )

        # Lower eyelid midpoint
        lower_mid = (
            (lower_1.x + lower_2.x) / 2.0,
            (lower_1.y + lower_2.y) / 2.0
        )

        # Vertical eye aperture
        vertical_aperture = distance(upper_mid, lower_mid)
        openness = vertical_aperture / eye_width

        if not np.isfinite(openness):
            return np.nan

        return float(openness)

    except Exception:
        return np.nan


# HEAD POSE
def calculate_head_pose(
    landmarks,
    image_width,
    image_height
):
    """
    Estimate relative head pose using facial landmark geometry.

    IMPORTANT:
        No mouth/yawning landmarks are used.

    Landmarks:
        1   = nose
        33  = left eye outer corner
        263 = right eye outer corner
        152 = chin
        234 = left cheek
        454 = right cheek

    Returns:
        pitch, yaw, roll

    These are relative geometric head-pose features intended for
    machine-learning analysis. They are NOT calibrated physical
    measurements in degrees.
    """
    try:
        # Get landmarks
        nose = landmarks[HEAD_POINTS["nose"]]
        chin = landmarks[HEAD_POINTS["chin"]]
        left_eye = landmarks[HEAD_POINTS["left_eye"]]
        right_eye = landmarks[HEAD_POINTS["right_eye"]]
        left_cheek = landmarks[HEAD_POINTS["left_cheek"]]
        right_cheek = landmarks[HEAD_POINTS["right_cheek"]]

        # Validate coordinates
        coordinates = np.array(
            [
                [nose.x, nose.y],
                [chin.x, chin.y],
                [left_eye.x, left_eye.y],
                [right_eye.x, right_eye.y],
                [left_cheek.x, left_cheek.y],
                [right_cheek.x, right_cheek.y],
            ],
            dtype=np.float64
        )

        if not np.isfinite(coordinates).all():
            return np.nan, np.nan, np.nan

        # Eye center
        eye_center_x = (left_eye.x + right_eye.x) / 2.0
        eye_center_y = (left_eye.y + right_eye.y) / 2.0

        # Basic facial dimensions
        eye_distance = distance(
            (left_eye.x, left_eye.y),
            (right_eye.x, right_eye.y)
        )

        face_width = distance(
            (left_cheek.x, left_cheek.y),
            (right_cheek.x, right_cheek.y)
        )

        face_height = distance(
            (eye_center_x, eye_center_y),
            (chin.x, chin.y)
        )

        if eye_distance <= 1e-6:
            return np.nan, np.nan, np.nan

        if face_width <= 1e-6:
            return np.nan, np.nan, np.nan

        if face_height <= 1e-6:
            return np.nan, np.nan, np.nan

        # ROLL

        # Rotation of the eye line relative to horizontal.
        roll = math.degrees(
            math.atan2(
                right_eye.y - left_eye.y,
                right_eye.x - left_eye.x
            )
        )

        # YAW
        
        # Horizontal displacement of the nose relative to the
        # midpoint between the eyes.
        
        # Normalization by eye distance makes this independent of
        # image resolution.
        nose_x_offset = nose.x - eye_center_x
        yaw_ratio = nose_x_offset / eye_distance
        yaw = math.degrees(math.atan(yaw_ratio))

        # PITCH
        
        # Vertical position of the nose between the eye center
        # and chin.
        
        # No mouth landmarks are required.
        eye_to_chin = chin.y - eye_center_y

        if abs(eye_to_chin) <= 1e-6:
            return np.nan, np.nan, np.nan

        nose_position = (
            nose.y - eye_center_y
        ) / eye_to_chin

        # Approximate neutral position.
        neutral_position = 0.50
        pitch_ratio = nose_position - neutral_position

        pitch = math.degrees(
            math.atan(
                pitch_ratio * 2.0
            )
        )

        # Final validation
        angles = np.array(
            [
                pitch,
                yaw,
                roll
            ],
            dtype=np.float64
        )

        if not np.isfinite(angles).all():
            return np.nan, np.nan, np.nan

        # Reject impossible/extreme geometric values.
        if abs(pitch) > 90.0:
            return np.nan, np.nan, np.nan

        if abs(yaw) > 90.0:
            return np.nan, np.nan, np.nan

        if abs(roll) > 90.0:
            return np.nan, np.nan, np.nan

        return (
            float(pitch),
            float(yaw),
            float(roll)
        )

    except Exception:
        return np.nan, np.nan, np.nan


# PROCESS ONE FRAME
def process_frame(
    image_path,
    detector
):
    result = {
        "image_exists": 0,
        "face_detected": 0,
        "ear_left": np.nan,
        "ear_right": np.nan,
        "ear_mean": np.nan,
        "eye_openness_left": np.nan,
        "eye_openness_right": np.nan,
        "eye_openness_mean": np.nan,
        "eye_validity": 0,
        "pitch": np.nan,
        "yaw": np.nan,
        "roll": np.nan,
    }

    # IMAGE EXISTENCE
    if not image_path.exists():
        return result

    result["image_exists"] = 1

    # READ IMAGE
    image = cv2.imread(str(image_path))

    if image is None:
        result["image_exists"] = 0
        return result

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        result["image_exists"] = 0
        return result

    # BGR -> RGB
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # FACE DETECTION
    detection = detector.detect(mp_image)

    if not detection.face_landmarks:
        return result

    result["face_detected"] = 1
    landmarks = detection.face_landmarks[0]

    # EYE FEATURES
    ear_left = calculate_ear(landmarks,LEFT_EYE)

    ear_right = calculate_ear(landmarks,RIGHT_EYE)

    openness_left = calculate_eye_openness(landmarks,LEFT_EYE)

    openness_right = calculate_eye_openness(landmarks,RIGHT_EYE)

    # Store individual values
    result["ear_left"] = ear_left
    result["ear_right"] = ear_right
    result["eye_openness_left"] = openness_left
    result["eye_openness_right"] = openness_right

    # EAR mean
    if (np.isfinite(ear_left) and np.isfinite(ear_right)):
        result["ear_mean"] = (ear_left + ear_right) / 2.0

    # Eye openness mean
    if (np.isfinite(openness_left) and np.isfinite(openness_right)):
        result["eye_openness_mean"] = (openness_left + openness_right) / 2.0

    # EYE VALIDITY
    if (
        np.isfinite(ear_left)
        and np.isfinite(ear_right)
        and np.isfinite(openness_left)
        and np.isfinite(openness_right)
    ):
        result["eye_validity"] = 1

    # HEAD POSE
    pitch, yaw, roll = calculate_head_pose(
        landmarks,
        width,
        height
    )

    result["pitch"] = pitch
    result["yaw"] = yaw
    result["roll"] = roll

    return result


# MAIN
def main():
    print("=" * 70)
    print("PHASE 9 — MEDIAPIPE FEATURE EXTRACTION")
    print("=" * 70)

    # VALIDATE REQUIRED FILES
    if not ANNOTATION_FILE.exists():
        print(f"ERROR: {ANNOTATION_FILE} not found.")
        return

    if not MODEL_FILE.exists():
        print(f"ERROR: {MODEL_FILE} not found.")
        return

    # LOAD ANNOTATIONS
    df = pd.read_csv(ANNOTATION_FILE)
    print(f"\nAnnotation rows: {len(df):,}")

    # LOAD CHECKPOINT
    if CHECKPOINT_FILE.exists():
        checkpoint = pd.read_csv(CHECKPOINT_FILE)
        results = checkpoint.to_dict("records")

        processed_paths = set(
            checkpoint["image_path"].astype(str)
        )

        print("\nCheckpoint found.")
        print(
            f"Already processed: "
            f"{len(processed_paths):,}"
        )
    else:
        results = []
        processed_paths = set()

    # MEDIAPIPE CONFIGURATION
    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_FILE)
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # CREATE FACE LANDMARKER
    with vision.FaceLandmarker.create_from_options(
        options
    ) as detector:
        total = len(df)

        for index, row in df.iterrows():
            image_path_string = str(row["image_path"])

            # Skip already processed rows
            if image_path_string in processed_paths:
                continue

            # RESOLVE IMAGE PATH
            relative_path = (
                image_path_string
                .replace("./classification_frames/", "")
                .replace("classification_frames/", "")
                .lstrip("/\\")
            )

            image_path = FRAME_ROOT / relative_path

            # EXTRACT FEATURES
            features = process_frame(
                image_path,
                detector
            )

            # CREATE OUTPUT ROW
            output_row = {
                "video_name": row["video_name"],
                "frame_number": row["frame_number"],
                "driver_state": row["driver_state"],
                "label": row["label"],
                "image_path": row["image_path"],
                **features
            }

            results.append(output_row)
            processed_paths.add(image_path_string)

            processed_count = len(results)

            # PROGRESS
            if (
                processed_count % 100 == 0
                or index == total - 1
            ):
                print(
                    f"Processed: "
                    f"{processed_count:,}/"
                    f"{total:,}"
                )

            # CHECKPOINT
            if processed_count % 1000 == 0:
                pd.DataFrame(results).to_csv(
                    CHECKPOINT_FILE,
                    index=False
                )

                print("  Checkpoint saved.")

    # FINAL DATAFRAME
    final_df = pd.DataFrame(results)

    final_df = final_df.drop_duplicates(
        subset=[
            "video_name",
            "frame_number"
        ],
        keep="last"
    )

    # SAVE OUTPUT
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # STATISTICS
    image_rate = (
        final_df["image_exists"]
        .mean()
        * 100
    )

    face_rate = (
        final_df["face_detected"]
        .mean()
        * 100
    )

    eye_rate = (
        final_df["eye_validity"]
        .mean()
        * 100
    )

    pose_rate = (
        final_df[
            [
                "pitch",
                "yaw",
                "roll"
            ]
        ]
        .notna()
        .all(axis=1)
        .mean()
        * 100
    )

    # FINAL REPORT
    print("\n" + "=" * 70)
    print("PHASE 9 COMPLETED")
    print("=" * 70)

    print(
        f"Total output frames : "
        f"{len(final_df):,}"
    )

    print(
        f"Image availability : "
        f"{image_rate:.2f}%"
    )

    print(
        f"Face detection     : "
        f"{face_rate:.2f}%"
    )

    print(
        f"Eye validity       : "
        f"{eye_rate:.2f}%"
    )

    print(
        f"Head pose validity : "
        f"{pose_rate:.2f}%"
    )

    print(
        f"\nOutput:"
        f"\n{OUTPUT_FILE}"
    )

    print("=" * 70)


# PROGRAM ENTRY POINT
if __name__ == "__main__":
    main()