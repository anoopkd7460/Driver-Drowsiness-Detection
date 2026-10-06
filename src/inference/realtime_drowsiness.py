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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_baseline_best.keras"
FACE_LANDMARKER_MODEL = PROJECT_ROOT / "src" / "models" / "face_landmarker.task"

SEQUENCE_LENGTH = 30
LSTM_THRESHOLD = 0.50
ALARM_DURATION_SECONDS = 8.0
HEAD_DOWN_CHANGE_DEGREES = 5.0
HEAD_POSE_HISTORY_LENGTH = 5
PREDICTION_TIMEOUT_SECONDS = 2.0
MIN_EYE_WIDTH = 8
MIN_EYE_HEIGHT = 4
ALARM_FREQUENCY = 1200
ALARM_BEEP_DURATION_MS = 500
ALARM_REPEAT_DELAY = 0.15
NEUTRAL_CALIBRATION_SECONDS = 3.0

TRAIN_MEAN = np.array([0.195844,-21.444487,4.021970,-1.334033],dtype=np.float32)
TRAIN_STD = np.array([0.070170,7.267564,8.950607,5.163262],dtype=np.float32)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
HEAD_POINTS = {"nose":1,"chin":152,"left_eye":33,"right_eye":263}

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

def calculate_ear(landmarks,eye_indices):
    p1,p2,p3,p4,p5,p6 = [landmarks[i] for i in eye_indices]
    vertical_1 = np.linalg.norm(np.array([p2.x-p6.x,p2.y-p6.y],dtype=np.float32))
    vertical_2 = np.linalg.norm(np.array([p3.x-p5.x,p3.y-p5.y],dtype=np.float32))
    horizontal = np.linalg.norm(np.array([p1.x-p4.x,p1.y-p4.y],dtype=np.float32))
    if horizontal < 1e-8:
        return np.nan
    return float((vertical_1+vertical_2)/(2.0*horizontal))

def calculate_head_pose(landmarks):
    nose = landmarks[HEAD_POINTS["nose"]]
    chin = landmarks[HEAD_POINTS["chin"]]
    left_eye = landmarks[HEAD_POINTS["left_eye"]]
    right_eye = landmarks[HEAD_POINTS["right_eye"]]
    eye_center_x = (left_eye.x+right_eye.x)/2.0
    eye_center_y = (left_eye.y+right_eye.y)/2.0
    eye_distance = np.sqrt((right_eye.x-left_eye.x)**2+(right_eye.y-left_eye.y)**2)

    roll = np.degrees(np.arctan2(right_eye.y-left_eye.y,right_eye.x-left_eye.x))

    if eye_distance < 1e-8:
        yaw = np.nan
    else:
        yaw = np.degrees(np.arctan((nose.x-eye_center_x)/eye_distance))

    denominator = chin.y-eye_center_y
    if abs(denominator) < 1e-8:
        pitch = np.nan
    else:
        nose_position = (nose.y-eye_center_y)/denominator
        pitch = np.degrees(np.arctan((nose_position-0.50)*2.0))

    return float(pitch),float(yaw),float(roll)

def eye_is_clear(frame,landmarks,eye_indices):
    height,width = frame.shape[:2]
    points = np.asarray([[int(landmarks[i].x*width),int(landmarks[i].y*height)] for i in eye_indices],dtype=np.int32)
    x,y,w,h = cv2.boundingRect(points)
    if w < MIN_EYE_WIDTH or h < MIN_EYE_HEIGHT:
        return False
    if x+w < 0 or y+h < 0 or x >= width or y >= height:
        return False
    return True

def both_eyes_clear(frame,landmarks):
    return eye_is_clear(frame,landmarks,LEFT_EYE) and eye_is_clear(frame,landmarks,RIGHT_EYE)

def extract_features(face_landmarker,frame):
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {"features":None,"landmarks":None,"eyes_clear":False,"head_pose":None}

    landmarks = result.face_landmarks[0]

    if len(landmarks) < 455:
        return {"features":None,"landmarks":landmarks,"eyes_clear":False,"head_pose":None}

    head_pose = calculate_head_pose(landmarks)
    eyes_clear = both_eyes_clear(frame,landmarks)

    if not eyes_clear:
        return {"features":None,"landmarks":landmarks,"eyes_clear":False,"head_pose":head_pose}

    left_ear = calculate_ear(landmarks,LEFT_EYE)
    right_ear = calculate_ear(landmarks,RIGHT_EYE)

    if not np.isfinite(left_ear) or not np.isfinite(right_ear):
        return {"features":None,"landmarks":landmarks,"eyes_clear":False,"head_pose":head_pose}

    ear_mean = (left_ear+right_ear)/2.0
    pitch,yaw,roll = head_pose
    features = np.array([ear_mean,pitch,yaw,roll],dtype=np.float32)

    if not np.all(np.isfinite(features)):
        return {"features":None,"landmarks":landmarks,"eyes_clear":False,"head_pose":head_pose}

    return {"features":features,"landmarks":landmarks,"eyes_clear":True,"head_pose":head_pose}

def standardize_sequence(sequence):
    sequence = np.asarray(sequence,dtype=np.float32)
    return (sequence-TRAIN_MEAN)/TRAIN_STD

def predict_drowsiness(model,sequence):
    sequence = standardize_sequence(sequence)
    model_input = np.expand_dims(sequence,axis=0)
    tensor_input = tf.convert_to_tensor(model_input,dtype=tf.float32)
    output = model(tensor_input,training=False)
    probability = float(output.numpy().reshape(-1)[0])
    prediction = int(probability >= LSTM_THRESHOLD)
    return probability,prediction

def inference_worker(model,shared_state,state_lock,stop_event):
    while not stop_event.is_set():
        sequence = None

        with state_lock:
            if shared_state["sequence"] is not None and not shared_state["inference_running"]:
                sequence = shared_state["sequence"].copy()
                shared_state["sequence"] = None
                shared_state["inference_running"] = True

        if sequence is not None:
            try:
                probability,prediction = predict_drowsiness(model,sequence)
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

def draw_eye_rectangle(frame,landmarks,eye_indices,eyes_clear):
    height,width = frame.shape[:2]
    points = np.asarray([[int(landmarks[i].x*width),int(landmarks[i].y*height)] for i in eye_indices],dtype=np.int32)
    x,y,w,h = cv2.boundingRect(points)
    padding_x = 8
    padding_y = 7
    x1 = max(0,x-padding_x)
    y1 = max(0,y-padding_y)
    x2 = min(width-1,x+w+padding_x)
    y2 = min(height-1,y+h+padding_y)
    rectangle_color = (0,255,0) if eyes_clear else (0,255,255)
    cv2.rectangle(frame,(x1,y1),(x2,y2),rectangle_color,2)

def draw_eye_boxes(frame,landmarks,eyes_clear):
    draw_eye_rectangle(frame,landmarks,LEFT_EYE,eyes_clear)
    draw_eye_rectangle(frame,landmarks,RIGHT_EYE,eyes_clear)

def draw_text(frame,text,position,font_scale=0.55,thickness=1):
    cv2.putText(frame,text,position,cv2.FONT_HERSHEY_SIMPLEX,font_scale,(0,0,255),thickness,cv2.LINE_AA)

def draw_status(frame,probability,prediction,drowsy_duration,head_down_duration,alarm_active,head_movement_status,head_down,eyes_clear,neutral_pitch,head_alarm,eye_alarm):
    x=25
    y=35
    line_gap=25
    height,width=frame.shape[:2]

    if alarm_active:
        status="DROWSINESS ALERT"
    elif head_down:
        status="HEAD DOWN"
    elif prediction==1:
        status="DROWSY"
    else:
        status="NOT DROWSY"

    draw_text(frame,f"Status: {status}",(x,y),0.60,2)
    draw_text(frame,f"Drowsy: {probability:.3f}",(x,y+line_gap),0.48,1)
    draw_text(frame,f"Drowsy Time: {drowsy_duration:.1f}/{ALARM_DURATION_SECONDS:.0f}s",(x,y+line_gap*2),0.48,1)
    draw_text(frame,f"Head Down: {head_down_duration:.1f}/{ALARM_DURATION_SECONDS:.0f}s",(x,y+line_gap*3),0.48,1)
    draw_text(frame,f"Head: {head_movement_status}",(x,y+line_gap*4),0.48,1)
    draw_text(frame,f"Alarm: {'ON' if alarm_active else 'OFF'}",(x,y+line_gap*5),0.48,1)

    panel_width=190
    panel_height=55
    panel_x=width-panel_width-20
    panel_y=15
    gap=10

    head_color=(0,0,255) if head_alarm else (0,255,0)
    eye_color=(0,0,255) if eye_alarm else (0,255,0)

    cv2.rectangle(frame,(panel_x,panel_y),(panel_x+panel_width,panel_y+panel_height),head_color,2)
    cv2.putText(frame,"HEAD POSITION",(panel_x+10,panel_y+22),cv2.FONT_HERSHEY_SIMPLEX,0.48,head_color,1,cv2.LINE_AA)
    head_text="ALARM" if head_alarm else ("DOWN" if head_down else "NORMAL")
    cv2.putText(frame,head_text,(panel_x+10,panel_y+44),cv2.FONT_HERSHEY_SIMPLEX,0.45,head_color,1,cv2.LINE_AA)

    eye_y=panel_y+panel_height+gap

    cv2.rectangle(frame,(panel_x,eye_y),(panel_x+panel_width,eye_y+panel_height),eye_color,2)
    cv2.putText(frame,"EYE POSITION",(panel_x+10,eye_y+22),cv2.FONT_HERSHEY_SIMPLEX,0.48,eye_color,1,cv2.LINE_AA)
    eye_text="ALARM" if eye_alarm else ("CLEAR" if eyes_clear else "UNCLEAR")
    cv2.putText(frame,eye_text,(panel_x+10,eye_y+44),cv2.FONT_HERSHEY_SIMPLEX,0.45,eye_color,1,cv2.LINE_AA)

    draw_text(frame,f"Neutral Pitch: {neutral_pitch:.1f}",(x,y+line_gap*6),0.40,1)

def alarm_worker(alarm_state,alarm_lock,stop_event):
    while not stop_event.is_set():
        with alarm_lock:
            active = alarm_state["active"]

        if active:
            try:
                winsound.Beep(ALARM_FREQUENCY,ALARM_BEEP_DURATION_MS)
            except Exception:
                pass
            stop_event.wait(ALARM_REPEAT_DELAY)
        else:
            stop_event.wait(0.05)

def start_alarm(alarm_state,alarm_lock):
    with alarm_lock:
        alarm_state["active"] = True

def stop_alarm(alarm_state,alarm_lock):
    with alarm_lock:
        alarm_state["active"] = False

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"LSTM model not found:\n{MODEL_PATH}")

    if not FACE_LANDMARKER_MODEL.exists():
        raise FileNotFoundError(f"MediaPipe model not found:\n{FACE_LANDMARKER_MODEL}")

    print("Loading LSTM model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model input shape: {model.input_shape}")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )

    print("Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FPS,25)

    feature_buffer = deque(maxlen=SEQUENCE_LENGTH)
    pose_history = deque(maxlen=HEAD_POSE_HISTORY_LENGTH)

    shared_state = {
        "sequence":None,
        "probability":0.0,
        "prediction":0,
        "inference_running":False,
        "last_inference_time":0.0
    }

    state_lock = threading.Lock()
    stop_event = threading.Event()

    inference_thread = threading.Thread(
        target=inference_worker,
        args=(model,shared_state,state_lock,stop_event),
        daemon=True
    )
    inference_thread.start()

    alarm_state = {"active":False}
    alarm_lock = threading.Lock()

    alarm_thread = threading.Thread(
        target=alarm_worker,
        args=(alarm_state,alarm_lock,stop_event),
        daemon=True
    )
    alarm_thread.start()

    last_probability = 0.0
    last_prediction = 0
    last_landmarks = None
    last_eyes_clear = False

    drowsy_start_time = None
    drowsy_duration = 0.0
    head_down_start_time = None
    head_down_duration = 0.0

    previous_head_pose = None
    head_movement_status = "STABLE"

    neutral_pitch_samples = []
    neutral_pitch = None
    calibration_start = time.time()
    calibration_complete = False

    processed_frames = 0
    valid_feature_frames = 0

    print("Camera started.")
    print(f"Neutral head calibration: {NEUTRAL_CALIBRATION_SECONDS:.0f} seconds")
    print(f"Head-down change threshold: {HEAD_DOWN_CHANGE_DEGREES:.1f} degrees")
    print(f"Alarm duration threshold: {ALARM_DURATION_SECONDS:.0f} seconds")
    print("Press Q or ESC to quit.")

    try:
        with FaceLandmarker.create_from_options(options) as face_landmarker:
            while not stop_event.is_set():
                if q_pressed() or escape_pressed():
                    break

                success,frame = cap.read()

                if not success:
                    break

                processed_frames += 1
                frame = cv2.flip(frame,1)

                result = extract_features(face_landmarker,frame)
                features = result["features"]
                landmarks = result["landmarks"]
                eyes_clear = result["eyes_clear"]
                current_head_pose = result["head_pose"]

                if landmarks is not None:
                    last_landmarks = landmarks
                    last_eyes_clear = eyes_clear

                if current_head_pose is not None:
                    current_pitch = current_head_pose[0]

                    if np.isfinite(current_pitch):
                        if not calibration_complete:
                            neutral_pitch_samples.append(current_pitch)
                            elapsed = time.time()-calibration_start

                            if elapsed >= NEUTRAL_CALIBRATION_SECONDS and len(neutral_pitch_samples) >= 15:
                                neutral_pitch = float(np.median(neutral_pitch_samples))
                                calibration_complete = True
                                print(f"\nNeutral pitch calibrated: {neutral_pitch:.2f} degrees")

                if current_head_pose is not None and previous_head_pose is not None:
                    current_pose = np.asarray(current_head_pose,dtype=np.float32)
                    previous_pose = np.asarray(previous_head_pose,dtype=np.float32)

                    if np.all(np.isfinite(current_pose)) and np.all(np.isfinite(previous_pose)):
                        pose_difference = np.abs(current_pose-previous_pose)
                        movement = float(np.max(pose_difference))
                        pose_history.append(movement)

                if current_head_pose is not None:
                    previous_head_pose = current_head_pose

                average_movement = float(np.mean(pose_history)) if pose_history else 0.0
                head_movement_status = "MOVING" if average_movement >= 3.0 else "STABLE"

                head_down = False

                if calibration_complete and current_head_pose is not None:
                    current_pitch = current_head_pose[0]

                    if np.isfinite(current_pitch):
                        pitch_change = current_pitch-neutral_pitch
                        head_down = pitch_change >= HEAD_DOWN_CHANGE_DEGREES

                current_time = time.time()

                if head_down:
                    if head_down_start_time is None:
                        head_down_start_time = current_time
                    head_down_duration = current_time-head_down_start_time
                else:
                    head_down_start_time = None
                    head_down_duration = 0.0

                if features is not None:
                    valid_feature_frames += 1
                    feature_buffer.append(features)

                if len(feature_buffer) == SEQUENCE_LENGTH:
                    with state_lock:
                        inference_busy = shared_state["inference_running"]
                        sequence_pending = shared_state["sequence"] is not None

                    if not inference_busy and not sequence_pending:
                        sequence = np.asarray(feature_buffer,dtype=np.float32)
                        with state_lock:
                            shared_state["sequence"] = sequence

                with state_lock:
                    last_probability = shared_state["probability"]
                    last_prediction = shared_state["prediction"]
                    last_inference_time = shared_state["last_inference_time"]

                prediction_fresh = (
                    last_inference_time > 0
                    and current_time-last_inference_time <= PREDICTION_TIMEOUT_SECONDS
                )

                if last_prediction == 1 and prediction_fresh:
                    if drowsy_start_time is None:
                        drowsy_start_time = current_time
                    drowsy_duration = current_time-drowsy_start_time
                else:
                    drowsy_start_time = None
                    drowsy_duration = 0.0

                lstm_alarm_condition = drowsy_duration >= ALARM_DURATION_SECONDS
                head_down_alarm_condition = head_down_duration >= ALARM_DURATION_SECONDS
                alarm_condition = lstm_alarm_condition or head_down_alarm_condition

                if alarm_condition:
                    start_alarm(alarm_state,alarm_lock)
                else:
                    stop_alarm(alarm_state,alarm_lock)

                with alarm_lock:
                    alarm_active = alarm_state["active"]

                if last_landmarks is not None:
                    draw_eye_boxes(frame,last_landmarks,last_eyes_clear)

                display_neutral_pitch = neutral_pitch if neutral_pitch is not None else 0.0

                draw_status(
                    frame,
                    last_probability,
                    last_prediction,
                    drowsy_duration,
                    head_down_duration,
                    alarm_active,
                    head_movement_status,
                    head_down,
                    last_eyes_clear,
                    display_neutral_pitch,
                    head_down_alarm_condition,
                    lstm_alarm_condition
                )

                cv2.imshow("Driver Drowsiness Detection",frame)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"),ord("Q")) or key == 27:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        stop_alarm(alarm_state,alarm_lock)
        stop_event.set()
        inference_thread.join(timeout=2.0)
        alarm_thread.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    print(f"Processed frames: {processed_frames}")
    print(f"Valid LSTM feature frames: {valid_feature_frames}")

    if processed_frames > 0:
        validity_rate = valid_feature_frames/processed_frames*100
        print(f"Feature validity rate: {validity_rate:.2f}%")

if __name__ == "__main__":
    main()