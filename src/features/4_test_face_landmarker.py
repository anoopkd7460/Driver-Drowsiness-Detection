# MEDIAPIPE FACE LANDMARKER MODEL TEST
# Purpose:
# Verify that the MediaPipe Face Landmarker model is correctly
# installed and can detect facial landmarks from one NITYMED image.
#
# Only one image is tested before processing the complete dataset.

from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# PATHS
MODEL_PATH = Path("src/models/face_landmarker.task")

IMAGE_PATH = Path(
    "dataset/NITYMED/classification_frames/"
    "P1043067_720/frame637.jpg"
)

# CHECK FILES
print("=" * 60)
print("FACE LANDMARKER TEST")
print("=" * 60)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Test image not found:\n{IMAGE_PATH}"
    )

print(f"\nModel : {MODEL_PATH}")
print(f"Image : {IMAGE_PATH}")

# CREATE FACE LANDMARKER
base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)

# LOAD IMAGE
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise ValueError("OpenCV could not read the test image.")

print(
    f"\nImage size: "
    f"{image.shape[1]} x {image.shape[0]}"
)

# CONVERT BGR -> RGB
rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# CREATE MEDIAPIPE IMAGE
mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)

# DETECT FACE LANDMARKS
print("\nRunning Face Landmarker...")

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_image)

# DISPLAY RESULT
print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

if not result.face_landmarks:
    print("\n❌ No face detected.")
else:
    print("\n✅ Face detected!")

    print(
        f"Number of faces: "
        f"{len(result.face_landmarks)}"
    )

    landmarks = result.face_landmarks[0]

    print(
        f"Number of landmarks: "
        f"{len(landmarks)}"
    )

    print("\nFirst 5 landmarks:")

    for i, landmark in enumerate(landmarks[:5]):
        print(
            f"{i}: "
            f"x={landmark.x:.4f}, "
            f"y={landmark.y:.4f}, "
            f"z={landmark.z:.4f}"
        )

# FINAL
print("\n" + "=" * 60)
print("PHASE 9A COMPLETED")
print("=" * 60)