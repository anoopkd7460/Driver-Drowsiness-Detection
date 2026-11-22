
from pathlib import Path
from collections import deque
import threading
import time
import ctypes
import winsound

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "lstm_baseline_best.keras"
FACE_LANDMARKER_MODEL = PROJECT_ROOT / "src" / "models" / "face_landmarker.task"

SEQUENCE_LENGTH = 30
LSTM_THRESHOLD = 0.50
ALARM_DURATION_SECONDS = 8.0
PREDICTION_TIMEOUT_SECONDS = 2.0

# Real-time head detection thresholds
# Enter posture state at 15 degrees; release only below 10 degrees.
HEAD_DOWN_CHANGE_DEGREES = 12.0
HEAD_UP_CHANGE_DEGREES = 12.0
HEAD_TILT_CHANGE_DEGREES = 10.0
HEAD_RELEASE_DEGREES = 6.0

# Internal neutral-pose calibration is retained but not displayed.
# Sit upright and face the camera during startup.
HEAD_DOWN_SIGN = 1.0
HEAD_TILT_SIGN = 1.0

HEAD_POSE_HISTORY_LENGTH = 7
# 75 valid frames provide about 3 seconds of neutral-pose samples at 25 FPS.
MIN_CALIBRATION_SAMPLES = 75

MIN_EYE_WIDTH = 8
MIN_EYE_HEIGHT = 4

ALARM_FREQUENCY = 1200
ALARM_BEEP_DURATION_MS = 500
ALARM_REPEAT_DELAY = 0.15

# Original training normalization: DO NOT CHANGE
TRAIN_MEAN = np.array(
    [0.195844, -21.444487, 4.021970, -1.334033],
    dtype=np.float32
)

TRAIN_STD = np.array(
    [0.070170, 7.267564, 8.950607, 5.163262],
    dtype=np.float32
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

HEAD_POINTS = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263
}


# ============================================================
# KEYBOARD
# ============================================================

def key_is_pressed(vk_code):
    try:
        state = ctypes.windll.user32.GetAsyncKeyState(vk_code)
        return bool(state & 0x8000)
    except Exception:
        return False


def q_pressed():
    return key_is_pressed(0x51)


def escape_pressed():
    return key_is_pressed(0x1B)


# ============================================================
# EAR CALCULATION
# ============================================================

def calculate_ear(landmarks, eye_indices):

    p1, p2, p3, p4, p5, p6 = [
        landmarks[i] for i in eye_indices
    ]

    vertical_1 = np.linalg.norm(
        np.array([p2.x-p6.x, p2.y-p6.y], dtype=np.float32)
    )

    vertical_2 = np.linalg.norm(
        np.array([p3.x-p5.x, p3.y-p5.y], dtype=np.float32)
    )

    horizontal = np.linalg.norm(
        np.array([p1.x-p4.x, p1.y-p4.y], dtype=np.float32)
    )

    if horizontal < 1e-8:
        return np.nan

    return float(
        (vertical_1 + vertical_2) / (2.0 * horizontal)
    )


def eye_is_clear(frame, landmarks, eye_indices):

    height, width = frame.shape[:2]

    points = np.asarray([
        [
            int(landmarks[i].x * width),
            int(landmarks[i].y * height)
        ]
        for i in eye_indices
    ], dtype=np.int32)

    x, y, w, h = cv2.boundingRect(points)

    if w < MIN_EYE_WIDTH or h < MIN_EYE_HEIGHT:
        return False

    if x+w < 0 or y+h < 0 or x >= width or y >= height:
        return False

    return True


def both_eyes_clear(frame, landmarks):

    return (
        eye_is_clear(frame, landmarks, LEFT_EYE)
        and eye_is_clear(frame, landmarks, RIGHT_EYE)
    )


# ============================================================
# ORIGINAL HEAD POSE
# PRESERVED FOR LSTM FEATURE COMPATIBILITY
# ============================================================

def calculate_head_pose(landmarks):

    nose = landmarks[HEAD_POINTS["nose"]]
    chin = landmarks[HEAD_POINTS["chin"]]

    left_eye = landmarks[HEAD_POINTS["left_eye"]]
    right_eye = landmarks[HEAD_POINTS["right_eye"]]

    eye_center_x = (left_eye.x + right_eye.x) / 2.0
    eye_center_y = (left_eye.y + right_eye.y) / 2.0

    eye_distance = np.sqrt(
        (right_eye.x-left_eye.x)**2
        + (right_eye.y-left_eye.y)**2
    )

    roll = np.degrees(
        np.arctan2(
            right_eye.y-left_eye.y,
            right_eye.x-left_eye.x
        )
    )

    if eye_distance < 1e-8:
        yaw = np.nan
    else:
        yaw = np.degrees(
            np.arctan(
                (nose.x-eye_center_x)/eye_distance
            )
        )

    denominator = chin.y-eye_center_y

    if abs(denominator) < 1e-8:
        pitch = np.nan
    else:
        nose_position = (
            (nose.y-eye_center_y)/denominator
        )

        pitch = np.degrees(
            np.arctan((nose_position-0.50)*2.0)
        )

    return float(pitch), float(yaw), float(roll)


# ============================================================
# NEW REAL-TIME HEAD POSE USING SOLVEPnP
# ============================================================

def calculate_head_pose_pnp(landmarks, frame_shape):

    height, width = frame_shape[:2]

    # Nose, chin, eye corners and cheeks.
    # No mouth landmarks are used.
    landmark_indices = [1, 152, 33, 263, 234, 454]

    image_points = np.asarray([
        [
            landmarks[index].x * width,
            landmarks[index].y * height
        ]
        for index in landmark_indices
    ], dtype=np.float64)

    # Approximate generic 3D face reference model.
    model_points = np.asarray([
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0]
    ], dtype=np.float64)

    focal_length = float(width)

    camera_matrix = np.asarray([
        [focal_length, 0, width / 2.0],
        [0, focal_length, height / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)

    try:

        success, rotation_vector, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        angles = cv2.RQDecomp3x3(rotation_matrix)[0]

        # Use the line joining the eye corners for roll. This gives a
        # symmetric left/right tilt measurement from the image itself.
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        eye_dx = (right_eye.x - left_eye.x) * width
        eye_dy = (right_eye.y - left_eye.y) * height
        eye_line_roll = np.degrees(np.arctan2(eye_dy, eye_dx))

        pose = np.asarray(
            [angles[0], angles[1], eye_line_roll],
            dtype=np.float32
        )

        if not np.all(np.isfinite(pose)):
            return None

        return pose

    except cv2.error:
        return None


def calculate_angle_difference(current, neutral):

    difference = np.asarray(current) - np.asarray(neutral)

    return (difference + 180.0) % 360.0 - 180.0


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(face_landmarker, frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:

        return {
            "features": None,
            "landmarks": None,
            "eyes_clear": False,
            "head_pose": None,
            "realtime_pose": None
        }

    landmarks = result.face_landmarks[0]

    if len(landmarks) < 455:

        return {
            "features": None,
            "landmarks": landmarks,
            "eyes_clear": False,
            "head_pose": None,
            "realtime_pose": None
        }

    # Original pose for the LSTM.
    original_pose = calculate_head_pose(landmarks)

    # New pose for real-time head detection.
    realtime_pose = calculate_head_pose_pnp(
        landmarks,
        frame.shape
    )

    eyes_clear = both_eyes_clear(frame, landmarks)

    if not eyes_clear:

        return {
            "features": None,
            "landmarks": landmarks,
            "eyes_clear": False,
            "head_pose": original_pose,
            "realtime_pose": realtime_pose
        }

    left_ear = calculate_ear(landmarks, LEFT_EYE)
    right_ear = calculate_ear(landmarks, RIGHT_EYE)

    if not np.isfinite(left_ear) or not np.isfinite(right_ear):

        return {
            "features": None,
            "landmarks": landmarks,
            "eyes_clear": False,
            "head_pose": original_pose,
            "realtime_pose": realtime_pose
        }

    ear_mean = (left_ear + right_ear) / 2.0

    # Preserve original feature order and calculations.
    pitch, yaw, roll = original_pose

    features = np.asarray(
        [ear_mean, pitch, yaw, roll],
        dtype=np.float32
    )

    if not np.all(np.isfinite(features)):
        features = None

    return {
        "features": features,
        "landmarks": landmarks,
        "eyes_clear": eyes_clear,
        "head_pose": original_pose,
        "realtime_pose": realtime_pose
    }


# ============================================================
# LSTM INFERENCE
# ============================================================

def standardize_sequence(sequence):

    sequence = np.asarray(sequence, dtype=np.float32)

    return (sequence-TRAIN_MEAN)/TRAIN_STD


def predict_drowsiness(model, sequence):

    sequence = standardize_sequence(sequence)

    model_input = np.expand_dims(sequence, axis=0)

    tensor_input = tf.convert_to_tensor(
        model_input,
        dtype=tf.float32
    )

    output = model(tensor_input, training=False)

    probability = float(output.numpy().reshape(-1)[0])

    prediction = int(probability >= LSTM_THRESHOLD)

    return probability, prediction


def inference_worker(model, shared_state, state_lock, stop_event):

    while not stop_event.is_set():

        sequence = None

        with state_lock:

            if (
                shared_state["sequence"] is not None
                and not shared_state["inference_running"]
            ):

                sequence = shared_state["sequence"].copy()

                shared_state["sequence"] = None

                shared_state["inference_running"] = True

        if sequence is not None:

            try:

                probability, prediction = predict_drowsiness(
                    model,
                    sequence
                )

                with state_lock:

                    shared_state["probability"] = probability
                    shared_state["prediction"] = prediction

                    shared_state["inference_running"] = False

                    shared_state["last_inference_time"] = time.time()

            except Exception as exc:

                print(f"\nInference error: {exc}")

                with state_lock:
                    shared_state["inference_running"] = False

        else:
            stop_event.wait(0.005)


# ============================================================
# DISPLAY
# ============================================================

def draw_eye_rectangle(frame, landmarks, eye_indices, eyes_clear):

    height, width = frame.shape[:2]

    points = np.asarray([
        [
            int(landmarks[i].x * width),
            int(landmarks[i].y * height)
        ]
        for i in eye_indices
    ], dtype=np.int32)

    x, y, w, h = cv2.boundingRect(points)

    x1 = max(0, x-8)
    y1 = max(0, y-7)
    x2 = min(width-1, x+w+8)
    y2 = min(height-1, y+h+7)

    color = (0, 255, 0) if eyes_clear else (0, 255, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def draw_eye_boxes(frame, landmarks, eyes_clear):

    draw_eye_rectangle(frame, landmarks, LEFT_EYE, eyes_clear)
    draw_eye_rectangle(frame, landmarks, RIGHT_EYE, eyes_clear)


def draw_text(frame, text, position, scale=0.55, thickness=1):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA
    )


def draw_status(
    frame, probability, prediction, drowsy_duration,
    head_status, head_position_duration, alarm_active,
    eyes_clear, head_alarm, eye_alarm
):
    x, y = 25, 35
    gap = 27

    if alarm_active:
        status = "DROWSINESS / HEAD ALERT"
    elif head_status in ("DOWN", "UP", "LEFT TILT", "RIGHT TILT"):
        status = f"HEAD {head_status}"
    elif prediction == 1:
        status = "DROWSY"
    else:
        status = "NOT DROWSY"

    draw_text(frame, f"Status: {status}", (x, y), 0.58, 2)
    draw_text(frame, f"Drowsy: {probability:.3f}", (x, y+gap))
    draw_text(frame, f"Drowsy Time: {drowsy_duration:.1f}/{ALARM_DURATION_SECONDS:.0f}s", (x, y+gap*2))
    draw_text(frame, f"Alarm: {'ON' if alarm_active else 'OFF'}", (x, y+gap*3))

    height, width = frame.shape[:2]
    panel_w, panel_h = 220, 58
    panel_x, panel_y = width-panel_w-20, 15
    head_color = (0, 0, 255) if head_alarm else (0, 255, 0)
    eye_color = (0, 0, 255) if eye_alarm else (0, 255, 0)

    cv2.rectangle(frame, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), head_color, 2)
    cv2.putText(frame, "HEAD POSITION", (panel_x+10, panel_y+23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, head_color, 1)
    cv2.putText(frame, head_status, (panel_x+10, panel_y+46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, head_color, 1)

    eye_y = panel_y+panel_h+10
    cv2.rectangle(frame, (panel_x, eye_y), (panel_x+panel_w, eye_y+panel_h), eye_color, 2)
    cv2.putText(frame, "EYE POSITION", (panel_x+10, eye_y+23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, eye_color, 1)
    eye_text = "ALARM" if eye_alarm else ("CLEAR" if eyes_clear else "UNCLEAR")
    cv2.putText(frame, eye_text, (panel_x+10, eye_y+46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, eye_color, 1)


# ============================================================
# ALARM THREAD
# ============================================================

def alarm_worker(alarm_state, alarm_lock, stop_event):
    """
    Play a continuous Windows system alert sound while the alarm is active.
    This replaces repeated winsound.Beep calls, which can fail silently on
    some Windows audio configurations.
    """
    sound_playing = False

    while not stop_event.is_set():
        with alarm_lock:
            active = alarm_state["active"]

        if active and not sound_playing:
            try:
                winsound.PlaySound(
                    "SystemExclamation",
                    winsound.SND_ALIAS | winsound.SND_ASYNC | winsound.SND_LOOP
                )
                sound_playing = True
            except Exception as exc:
                # Keep the application running, but expose the audio failure.
                print(f"Alarm sound could not start: {exc}")
                stop_event.wait(0.5)

        elif not active and sound_playing:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception as exc:
                print(f"Alarm sound could not stop cleanly: {exc}")
            sound_playing = False

        stop_event.wait(0.05)

    # Always stop playback when the application exits.
    if sound_playing:
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass


def start_alarm(alarm_state, alarm_lock):
    with alarm_lock:
        alarm_state["active"] = True


def stop_alarm(alarm_state, alarm_lock):
    with alarm_lock:
        alarm_state["active"] = False


# ============================================================
# MAIN
# ============================================================

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LSTM model not found:\n{MODEL_PATH}"
        )

    if not FACE_LANDMARKER_MODEL.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found:\n{FACE_LANDMARKER_MODEL}"
        )

    print("Loading LSTM model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Model input shape: {model.input_shape}")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(FACE_LANDMARKER_MODEL)
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )

    print("Opening webcam...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FPS, 25)

    feature_buffer = deque(maxlen=SEQUENCE_LENGTH)

    pose_history = deque(
        maxlen=HEAD_POSE_HISTORY_LENGTH
    )

    shared_state = {
        "sequence": None,
        "probability": 0.0,
        "prediction": 0,
        "inference_running": False,
        "last_inference_time": 0.0
    }

    state_lock = threading.Lock()
    stop_event = threading.Event()

    inference_thread = threading.Thread(
        target=inference_worker,
        args=(model, shared_state, state_lock, stop_event),
        daemon=True
    )

    inference_thread.start()

    alarm_state = {"active": False}
    alarm_lock = threading.Lock()

    alarm_thread = threading.Thread(
        target=alarm_worker,
        args=(alarm_state, alarm_lock, stop_event),
        daemon=True
    )

    alarm_thread.start()

    last_probability = 0.0
    last_prediction = 0

    last_landmarks = None
    last_eyes_clear = False

    drowsy_start_time = None
    drowsy_duration = 0.0

    head_status = "INITIALIZING"
    head_position_start_time = None
    head_position_duration = 0.0
    active_posture = None

    neutral_pose_samples = []
    neutral_pose = None
    calibration_complete = False

    processed_frames = 0
    valid_feature_frames = 0

    print("Camera started.")
    print("Keep your head upright and face the camera until initialization completes.")
    print(f"Neutral initialization samples: {MIN_CALIBRATION_SAMPLES} valid frames")
    print(f"Head posture alarm duration: {ALARM_DURATION_SECONDS:.0f} seconds")
    print("Press Q or ESC to quit.")

    try:

        with FaceLandmarker.create_from_options(options) as face_landmarker:

            while not stop_event.is_set():

                if q_pressed() or escape_pressed():
                    break

                success, frame = cap.read()

                if not success:
                    break

                processed_frames += 1

                frame = cv2.flip(frame, 1)

                result = extract_features(
                    face_landmarker,
                    frame
                )

                features = result["features"]
                landmarks = result["landmarks"]
                eyes_clear = result["eyes_clear"]
                realtime_pose = result["realtime_pose"]

                current_time = time.time()
                head_down = head_up = head_tilted = False
                head_alarm_condition = False

                if landmarks is not None:
                    last_landmarks = landmarks
                    last_eyes_clear = eyes_clear
                else:
                    last_landmarks = None
                    last_eyes_clear = False

                # Smooth real-time head pose and establish a reliable neutral baseline.
                if realtime_pose is not None:
                    pose_history.append(realtime_pose)
                    filtered_pose = np.median(np.stack(pose_history), axis=0)

                    # Calibration starts only after valid webcam pose frames arrive.
                    if not calibration_complete:
                        neutral_pose_samples.append(filtered_pose.copy())
                        if len(neutral_pose_samples) >= MIN_CALIBRATION_SAMPLES:
                            neutral_pose = np.median(
                                np.stack(neutral_pose_samples[-MIN_CALIBRATION_SAMPLES:]),
                                axis=0
                            )
                            calibration_complete = True
                            head_status = "NORMAL"
                            print("Neutral head position initialized from valid frames.")

                    if calibration_complete:
                        pose_deltas = calculate_angle_difference(filtered_pose, neutral_pose)
                        pitch_delta = HEAD_DOWN_SIGN * float(pose_deltas[0])
                        roll_delta = HEAD_TILT_SIGN * float(pose_deltas[2])

                        # Hysteresis prevents small fluctuations from repeatedly
                        # starting/stopping the posture timer.
                        down_limit = HEAD_RELEASE_DEGREES if active_posture == "DOWN" else HEAD_DOWN_CHANGE_DEGREES
                        up_limit = HEAD_RELEASE_DEGREES if active_posture == "UP" else HEAD_UP_CHANGE_DEGREES
                        tilt_limit = HEAD_RELEASE_DEGREES if active_posture in ("LEFT TILT", "RIGHT TILT") else HEAD_TILT_CHANGE_DEGREES

                        # Check tilt first: diagonal head movement should not
                        # be incorrectly consumed by the UP/DOWN conditions.
                        if abs(roll_delta) >= tilt_limit:
                            detected_posture = "LEFT TILT" if roll_delta < 0 else "RIGHT TILT"
                        elif pitch_delta >= down_limit:
                            detected_posture = "DOWN"
                        elif pitch_delta <= -up_limit:
                            detected_posture = "UP"
                        else:
                            detected_posture = "NORMAL"

                        head_status = detected_posture
                    else:
                        head_status = "INITIALIZING"
                else:
                    head_status = "NO FACE" if landmarks is None else "POSE UNAVAILABLE"

                # A single continuous timer follows the currently detected posture.
                if realtime_pose is not None and calibration_complete and head_status != "NORMAL":
                    if active_posture != head_status:
                        active_posture = head_status
                        head_position_start_time = current_time

                    head_position_duration = (
                        current_time - head_position_start_time
                        if head_position_start_time is not None else 0.0
                    )
                else:
                    active_posture = None
                    head_position_start_time = None
                    head_position_duration = 0.0

                # ----------------------------------------
                # LSTM SEQUENCE
                # ----------------------------------------

                if features is not None:

                    valid_feature_frames += 1

                    feature_buffer.append(features)

                if len(feature_buffer) == SEQUENCE_LENGTH:

                    with state_lock:

                        inference_busy = (
                            shared_state["inference_running"]
                        )

                        sequence_pending = (
                            shared_state["sequence"] is not None
                        )

                    if not inference_busy and not sequence_pending:

                        sequence = np.asarray(
                            feature_buffer,
                            dtype=np.float32
                        )

                        with state_lock:
                            shared_state["sequence"] = sequence

                # ----------------------------------------
                # GET LSTM PREDICTION
                # ----------------------------------------

                with state_lock:

                    last_probability = shared_state["probability"]
                    last_prediction = shared_state["prediction"]

                    last_inference_time = (
                        shared_state["last_inference_time"]
                    )

                prediction_fresh = (
                    last_inference_time > 0
                    and current_time-last_inference_time
                    <= PREDICTION_TIMEOUT_SECONDS
                )

                if last_prediction == 1 and prediction_fresh:

                    if drowsy_start_time is None:
                        drowsy_start_time = current_time

                    drowsy_duration = (
                        current_time-drowsy_start_time
                    )

                else:

                    drowsy_start_time = None
                    drowsy_duration = 0.0

                # ----------------------------------------
                # INDEPENDENT ALARM CONDITIONS
                # ----------------------------------------

                lstm_alarm_condition = (
                    drowsy_duration >= ALARM_DURATION_SECONDS
                )

                head_alarm_condition = (
                    active_posture in ("DOWN", "UP", "LEFT TILT", "RIGHT TILT")
                    and head_position_duration >= ALARM_DURATION_SECONDS
                )

                alarm_condition = lstm_alarm_condition or head_alarm_condition

                if alarm_condition:
                    start_alarm(alarm_state, alarm_lock)
                else:
                    stop_alarm(alarm_state, alarm_lock)

                with alarm_lock:
                    alarm_active = alarm_state["active"]

                # ----------------------------------------
                # DISPLAY
                # ----------------------------------------

                if last_landmarks is not None:

                    draw_eye_boxes(
                        frame,
                        last_landmarks,
                        last_eyes_clear
                    )

                draw_status(
                    frame=frame,
                    probability=last_probability,
                    prediction=last_prediction,
                    drowsy_duration=drowsy_duration,
                    head_status=head_status,
                    head_position_duration=head_position_duration,
                    alarm_active=alarm_active,
                    eyes_clear=last_eyes_clear,
                    head_alarm=head_alarm_condition,
                    eye_alarm=lstm_alarm_condition
                )

                cv2.imshow(
                    "Driver Drowsiness Detection",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q")) or key == 27:
                    break

    except KeyboardInterrupt:
        pass

    finally:

        stop_alarm(alarm_state, alarm_lock)

        stop_event.set()

        inference_thread.join(timeout=2.0)
        alarm_thread.join(timeout=2.0)

        cap.release()

        cv2.destroyAllWindows()
        cv2.waitKey(1)

    print(f"Processed frames: {processed_frames}")
    print(f"Valid LSTM feature frames: {valid_feature_frames}")

    if processed_frames > 0:

        validity_rate = (
            valid_feature_frames/processed_frames
        )*100.0

        print(
            f"Feature validity rate: {validity_rate:.2f}%"
        )


if __name__ == "__main__":
    main()
