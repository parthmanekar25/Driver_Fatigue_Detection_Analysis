import cv2
import mediapipe as mp
import numpy as np
import tflite_runtime.interpreter as tflite
from collections import deque
import RPi.GPIO as GPIO
import time
import os
from picamera2 import Picamera2, Preview # Preview might not be needed if running headless
from libcamera import Transform, controls as Controls # Import Controls for AWB
# from picamera2.encoders import H264Encoder # Not used in current logic
# from picamera2.outputs import FileOutput # Not used in current logic
import pygame
import math # Needed for EAR calculation

# --- Constants ---
MODEL_DIR = "/home/pi/project"  # Update if needed
EYE_MODEL_PATH = os.path.join(MODEL_DIR, "eye_state_model.tflite")
YAWN_MODEL_PATH = os.path.join(MODEL_DIR, "best_yawn_model.tflite")

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# MediaPipe Indices
# Eye indices for BBox (can be simplified if only corners needed)
LEFT_EYE_BBOX_INDICES = [33, 133, 160, 159, 158, 153, 144, 145, 246]
RIGHT_EYE_BBOX_INDICES = [362, 263, 385, 386, 387, 380, 373, 374, 466]
# EAR landmark indices (Vertical P2, P6, P3, P5 and Horizontal P1, P4)
LEFT_EYE_EAR_INDICES = [159, 145, 158, 153, 33, 133] # P2, P6, P3, P5, P1, P4
RIGHT_EYE_EAR_INDICES = [386, 374, 385, 380, 362, 263] # P2, P6, P3, P5, P1, P4

MOUTH_LANDMARKS = list(set([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324,
    318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267,
    269, 270, 409, 415, 310, 311, 312, 13
]))

# Buffers & Smoothing
EYE_STATE_BUFFER_LENGTH = 10
EYE_STATE_EMA_ALPHA = 0.5
MAR_BUFFER_LENGTH = 5
# --- ADJUSTED BUFFER/THRESHOLD ---
EYE_CLOSED_BUFFER_LENGTH = 20 # Reduced lookback window
EAR_BUFFER_LENGTH = 5 # Buffer for EAR smoothing
CNN_YAWN_BUFFER_LENGTH = 25

# Thresholds (Tune these on the Pi)
# --- ADJUSTED/ADDED THRESHOLDS ---
EYE_CLOSED_THRESHOLD_CNN_EMA = 0.60 # Tune based on CNN output (lower = easier to trigger closed)
EAR_THRESHOLD = 0.20 # Tune based on observed EAR values (lower = eyes must be more closed)
MAR_THRESHOLD = 0.5
YAWN_MAR_FRAMES_THRESHOLD = 15
CNN_YAWN_PROB_THRESHOLD = 0.7
CNN_YAWN_FRAMES_TRIGGER = 18
EYE_CLOSED_COUNT_THRESHOLD = 6 # Adjusted for smaller buffer (tune this!)
HEAD_TILT_THRESHOLD = 18
DROWSY_RESET_FRAMES = 20 # Reduced recovery time
YAWN_COOLDOWN_FRAMES = 40

EYE_CNN_INPUT_SIZE = (128, 128)  # Verify with your eye_model.tflite
YAWN_CNN_INPUT_SIZE = (96, 96)   # Verify with your yawn_model.tflite
INFERENCE_INTERVAL = 3 # Run CNNs every 3 frames
EYE_BBOX_SCALE = 1.7 # Slightly increased scale factor for bbox (tune this)

# GPIO
BUZZER_PIN = 17
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)
buzzer_active_until = 0


def play_audio(audio_file):
    # Initialize pygame mixer if not already initialized
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    try:
        # Stop any currently playing music
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            pygame.mixer.music.unload() # Unload to be safe

        pygame.mixer.music.load(audio_file)
        print(f"Playing audio: {audio_file}")
        pygame.mixer.music.play()
        # Don't block here with a while loop if other things need to run
        # Check status elsewhere or use sound objects if overlap is needed
    except pygame.error as e:
        print(f"Error playing audio {audio_file}: {e}")
    # Don't quit mixer here if you plan to play more sounds soon


# --- Buzzer Control ---
def trigger_buzzer(alert_type, duration_pattern):
    global buzzer_active_until, last_alert_trigger_time # Make sure last_alert_trigger_time is global
    current_time = time.time()

    # Check main alert cooldown first
    if current_time < last_alert_trigger_time + ALERT_COOLDOWN_SECONDS:
        # print(f"Alert cooldown active. Skipping {alert_type}.") # Optional debug
        return

    # Check if a pattern is already playing
    if current_time < buzzer_active_until:
        # print(f"Buzzer pattern already active. Skipping {alert_type}.") # Optional debug
        return

    print(f"--- ALERT TRIGGERED: {alert_type} ---")
    # Optional: Play corresponding audio alert
    # if alert_type == "SEVERE" or alert_type == "EYES":
    #     play_audio("/home/pi/project/alert_sound.mp3") # Replace with your sound file
    # elif alert_type == "YAWN":
    #      play_audio("/home/pi/project/yawn_sound.mp3")

    total_duration = 0
    try:
        for on_time, off_time in duration_pattern:
            if on_time > 0:
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                time.sleep(on_time)
                total_duration += on_time
            if off_time > 0: # Ensure buzzer is off between pulses or at the end
                GPIO.output(BUZZER_PIN, GPIO.LOW)
                time.sleep(off_time)
                total_duration += off_time
        GPIO.output(BUZZER_PIN, GPIO.LOW) # Ensure off at the very end
        buzzer_active_until = current_time + total_duration + 0.1 # Small buffer
        last_alert_trigger_time = current_time # Update main cooldown time
    except Exception as e:
        print(f"GPIO Error: {e}")
        try:
            GPIO.output(BUZZER_PIN, GPIO.LOW) # Attempt to turn off buzzer on error
        except Exception:
            pass # Ignore errors during cleanup


BUZZER_PATTERNS = {
    "EYES": [(0.1, 0.1), (0.1, 0.1)],
    "YAWN": [(0.4, 0.2)],
    "TILT": [(0.15, 0.1), (0.15, 0.3)],
    "SEVERE": [(0.2, 0.05), (0.2, 0.05), (0.2, 0.4)]
}
ALERT_COOLDOWN_SECONDS = 4.0
last_alert_trigger_time = 0 # Initialize


# --- TFLite Model Loading ---
def load_tflite_model(model_path):
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"Loaded TFLite model: {os.path.basename(model_path)}")
        print(f"  Input Details: {input_details[0]['shape']}, {input_details[0]['dtype']}")
        print(f"  Output Details: {output_details[0]['shape']}, {output_details[0]['dtype']}")
        return interpreter, input_details, output_details
    except Exception as e:
        print(f"Error loading TFLite model {model_path}: {e}")
        return None, None, None

eye_interpreter, eye_input_details, eye_output_details = load_tflite_model(EYE_MODEL_PATH)
yawn_interpreter, yawn_input_details, yawn_output_details = load_tflite_model(YAWN_MODEL_PATH)

if not eye_interpreter or not yawn_interpreter:
    print("Failed to load one or more TFLite models. Exiting.")
    GPIO.cleanup() # Cleanup GPIO if exiting early
    exit()

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Needed for detailed eye/mouth landmarks
    min_detection_confidence=0.6, # Start here, maybe lower slightly (e.g., 0.55) if face detection is lost too easily
    min_tracking_confidence=0.6
)

# --- Helper Functions ---

# --- ADDED CLAHE ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_eye(eye_img, input_details):
    input_shape = input_details[0]['shape']
    target_size = (input_shape[2], input_shape[1]) # W, H
    expected_channels = input_shape[3]

    # Basic validation
    if eye_img is None or eye_img.size == 0 or eye_img.shape[0] < 5 or eye_img.shape[1] < 5:
        # print("Warning: Small or invalid eye image received.") # Optional debug
        return None

    # Ensure Grayscale
    if len(eye_img.shape) == 3 and eye_img.shape[2] == 3:
        eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    elif len(eye_img.shape) == 2:
        eye_gray = eye_img
    else:
        print(f"Warning: Unexpected eye image shape: {eye_img.shape}")
        return None # Cannot process unexpected shape

    # --- Apply CLAHE for contrast enhancement ---
    try:
        eye_eq = clahe.apply(eye_gray)
    except Exception as e:
        print(f"Warning: CLAHE failed: {e}. Using original grayscale image.")
        eye_eq = eye_gray

    # Resize
    try:
        eye_resized = cv2.resize(eye_eq, target_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"Warning: Eye resize failed: {e}")
        return None

    # Normalize pixel values to [0, 1]
    eye_normalized = eye_resized.astype(np.float32) / 255.0

    # Add batch and channel dimensions
    if expected_channels == 1:
        eye_input = np.expand_dims(np.expand_dims(eye_normalized, axis=-1), axis=0)
    # elif expected_channels == 3: # If your model expects color
    #     # Convert normalized grayscale back to 3 channels if needed
    #     eye_color_normalized = cv2.cvtColor(eye_normalized, cv2.COLOR_GRAY2BGR)
    #     eye_input = np.expand_dims(eye_color_normalized, axis=0)
    else:
        print(f"Warning: Eye model expects {expected_channels} channels, but preprocessing generated 1. Check model requirements.")
        # Assuming fallback to grayscale might work, adjust if needed
        eye_input = np.expand_dims(np.expand_dims(eye_normalized, axis=-1), axis=0)

    # Final shape check
    if eye_input.shape != tuple(input_shape):
        print(f"ERROR: Preprocessed eye shape {eye_input.shape} mismatch model input {tuple(input_shape)}")
        return None

    return eye_input


def preprocess_mouth(mouth_img, input_details):
    # Assuming yawn model expects color input based on pi2.py, adjust if needed
    input_shape = input_details[0]['shape'] # e.g., [1, 96, 96, 3]
    target_size = (input_shape[2], input_shape[1]) # W, H
    expected_channels = input_shape[3]

    if mouth_img is None or mouth_img.size == 0 or mouth_img.shape[0] < 10 or mouth_img.shape[1] < 10:
        return None

    try:
        processed_img = None
        if expected_channels == 3:
            if len(mouth_img.shape) == 2: # Convert grayscale to BGR
                processed_img = cv2.cvtColor(mouth_img, cv2.COLOR_GRAY2BGR)
            elif len(mouth_img.shape) == 3 and mouth_img.shape[2] == 1: # Convert [H, W, 1] to BGR
                 processed_img = cv2.cvtColor(mouth_img, cv2.COLOR_GRAY2BGR)
            elif len(mouth_img.shape) == 3 and mouth_img.shape[2] == 3: # Already BGR
                 processed_img = mouth_img
            else: return None # Invalid shape

            # Optional: Apply color-based contrast enhancement if needed, e.g., HSV equalization
            # hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
            # hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2]) # Equalize Value channel
            # processed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif expected_channels == 1:
             if len(mouth_img.shape) == 3 and mouth_img.shape[2] == 3: # Convert color to gray
                 processed_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)
             elif len(mouth_img.shape) == 3 and mouth_img.shape[2] == 1: # Squeeze if [H, W, 1]
                 processed_img = mouth_img.squeeze(axis=-1)
             elif len(mouth_img.shape) == 2: # Already gray
                 processed_img = mouth_img
             else: return None # Invalid shape
             # Apply CLAHE for grayscale
             processed_img = clahe.apply(processed_img)
        else:
             print(f"ERROR: Unsupported channel count in yawn model: {expected_channels}")
             return None

        # Resize
        mouth_resized = cv2.resize(processed_img, target_size, interpolation=cv2.INTER_AREA)
        # Normalize
        mouth_normalized = mouth_resized.astype(np.float32) / 255.0

        # Add channel dimension if grayscale model needs it and batch dim
        if expected_channels == 1 and len(mouth_normalized.shape) == 2:
            mouth_normalized = np.expand_dims(mouth_normalized, axis=-1)

        mouth_input = np.expand_dims(mouth_normalized, axis=0)

        # Final shape check
        if mouth_input.shape != tuple(input_shape):
            print(f"ERROR: Preprocessed mouth shape {mouth_input.shape} mismatch model input {tuple(input_shape)}")
            return None

        return mouth_input

    except Exception as e:
        print(f"Error preprocessing mouth image: {e}")
        return None


def predict_tflite(interpreter, input_details, output_details, data):
    if data is None or interpreter is None: return None
    try:
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data
    except Exception as e:
        # Provide more context for the error if possible
        input_shape_str = str(input_details[0]['shape']) if input_details else "N/A"
        data_shape_str = str(data.shape) if hasattr(data, 'shape') else "N/A"
        print(f"TFLite prediction error: {e}. Input shape expected: {input_shape_str}, Data shape provided: {data_shape_str}")
        return None

# --- REVISED BBOX FUNCTION ---
def get_scaled_bbox(landmarks, indices, width, height, scale=1.5, min_dim=10, last_bbox=None):
    """Calculates a bounding box around specified landmarks and scales it, with robustness."""
    try:
        # Filter landmarks to be within valid screen coordinates
        points = np.array([(landmarks.landmark[i].x * width, landmarks.landmark[i].y * height)
                           for i in indices
                           if 0 <= landmarks.landmark[i].x <= 1 and 0 <= landmarks.landmark[i].y <= 1],
                          dtype=np.int32)

        # Need at least 3 points to form a reasonable bounding box
        if len(points) < 3:
            # print("Warning: Not enough valid landmarks for bbox.") # Optional debug
            return last_bbox if last_bbox else None

        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        # Calculate center and original dimensions
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        bbox_w, bbox_h = x_max - x_min, y_max - y_min

        # Check for degenerate box
        if bbox_w <= 0 or bbox_h <= 0:
            # print("Warning: Degenerate bbox calculated.") # Optional debug
            return last_bbox if last_bbox else None

        # Calculate scaled dimensions
        scaled_w, scaled_h = bbox_w * scale, bbox_h * scale

        # Calculate new corners, ensuring they are within frame bounds
        x1 = max(0, int(center_x - scaled_w / 2))
        y1 = max(0, int(center_y - scaled_h / 2))
        x2 = min(width - 1, int(center_x + scaled_w / 2)) # Use width-1 and height-1 for inclusive bounds
        y2 = min(height - 1, int(center_y + scaled_h / 2))

        # Ensure minimum dimensions AFTER scaling and clipping
        final_w = x2 - x1
        final_h = y2 - y1

        if final_w < min_dim or final_h < min_dim:
             # Optional: Slightly expand small boxes if needed, respecting boundaries
            # print(f"Warning: Scaled bbox too small ({final_w}x{final_h}). Falling back.") # Optional debug
            # Fallback to last known good bbox if the new one is too small
            return last_bbox if last_bbox else None

        # Final check for validity
        if x1 >= x2 or y1 >= y2:
            # print("Warning: Invalid bbox coordinates after scaling/clipping.") # Optional debug
            return last_bbox if last_bbox else None

        return x1, y1, x2, y2

    except Exception as e:
        # print(f"Error calculating scaled bbox: {e}") # Optional debug print
        return last_bbox if last_bbox else None


def calculate_mar(landmarks, width, height):
    try:
        # Ensure landmarks are valid before accessing
        if not (landmarks and
                hasattr(landmarks.landmark[13], 'y') and hasattr(landmarks.landmark[14], 'y') and
                hasattr(landmarks.landmark[61], 'x') and hasattr(landmarks.landmark[291], 'x')):
            return 0.0

        # Check coordinates are within expected range [0, 1] - adjust if necessary
        if not (0 <= landmarks.landmark[13].y <= 1 and 0 <= landmarks.landmark[14].y <= 1 and
                0 <= landmarks.landmark[61].x <= 1 and 0 <= landmarks.landmark[291].x <= 1):
             # print("Warning: MAR landmarks out of bounds.") # Optional debug
             return 0.0 # Or handle as appropriate

        top_lip_y = landmarks.landmark[13].y * height
        bottom_lip_y = landmarks.landmark[14].y * height
        mouth_height = abs(bottom_lip_y - top_lip_y)

        left_corner_x = landmarks.landmark[61].x * width
        right_corner_x = landmarks.landmark[291].x * width
        mouth_width = abs(right_corner_x - left_corner_x)

        if mouth_width < 1e-6: # Avoid division by zero
            return 0.0
        return mouth_height / mouth_width
    except (IndexError, AttributeError, Exception) as e:
        # print(f"Error calculating MAR: {e}") # Optional debug
        return 0.0

# --- ADDED EAR CALCULATION ---
def distance(p1, p2):
    """Calculate Euclidean distance."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(eye_landmarks, width, height):
    """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
    try:
        # Extract points P1 to P6
        # Vertical points (indices 0-3 in eye_landmarks list)
        p2 = (eye_landmarks[0].x * width, eye_landmarks[0].y * height)
        p6 = (eye_landmarks[1].x * width, eye_landmarks[1].y * height)
        p3 = (eye_landmarks[2].x * width, eye_landmarks[2].y * height)
        p5 = (eye_landmarks[3].x * width, eye_landmarks[3].y * height)
        # Horizontal points (indices 4-5 in eye_landmarks list)
        p1 = (eye_landmarks[4].x * width, eye_landmarks[4].y * height)
        p4 = (eye_landmarks[5].x * width, eye_landmarks[5].y * height)

        # Calculate vertical distances
        ver1 = distance(p2, p6)
        ver2 = distance(p3, p5)
        # Calculate horizontal distance
        hor = distance(p1, p4)

        if hor < 1e-6: # Avoid division by zero
            return 0.0

        ear = (ver1 + ver2) / (2.0 * hor)
        return ear
    except (IndexError, AttributeError, TypeError, Exception) as e:
         # print(f"Error calculating EAR: {e}") # Optional debug
         return 0.0 # Return a default value indicating failure


def detect_head_tilt(landmarks, width, height):
    try:
        # Use inner eye corners for stability
        left_eye_inner = landmarks.landmark[133]
        right_eye_inner = landmarks.landmark[362]

        # Validate landmark coordinates
        if not (0 <= left_eye_inner.x <= 1 and 0 <= left_eye_inner.y <= 1 and \
                0 <= right_eye_inner.x <= 1 and 0 <= right_eye_inner.y <= 1):
             # print("Warning: Head tilt landmarks out of bounds.") # Optional debug
             return 0.0

        left_eye = np.array([left_eye_inner.x * width, left_eye_inner.y * height])
        right_eye = np.array([right_eye_inner.x * width, right_eye_inner.y * height])

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        if abs(dx) < 1e-6: # Handle vertical alignment
             return 90.0 * np.sign(dy) if dy != 0 else 0.0

        angle = np.degrees(np.arctan2(dy, dx))
        return angle
    except (IndexError, AttributeError, Exception) as e:
        # print(f"Error calculating head tilt: {e}") # Optional debug
        return 0.0

# --- Buffers ---
left_eye_cnn_buffer = deque(maxlen=EYE_STATE_BUFFER_LENGTH) # For CNN output prob
right_eye_cnn_buffer = deque(maxlen=EYE_STATE_BUFFER_LENGTH)
left_ear_buffer = deque(maxlen=EAR_BUFFER_LENGTH) # Buffer for Left EAR
right_ear_buffer = deque(maxlen=EAR_BUFFER_LENGTH) # Buffer for Right EAR
mar_buffer = deque(maxlen=MAR_BUFFER_LENGTH)
eye_closed_buffer = deque(maxlen=EYE_CLOSED_BUFFER_LENGTH) # Combined closure state
cnn_yawn_buffer = deque(maxlen=CNN_YAWN_BUFFER_LENGTH)
yawn_mar_trigger_buffer = deque(maxlen=YAWN_MAR_FRAMES_THRESHOLD) # Buffer for MAR based yawn trigger


# --- State Variables ---
frame_count = 0
drowsy = False
yawning = False
head_tilt_detected = False
consecutive_open_eye_frames = 0
yawn_cooldown_counter = 0
# Last known *smoothed* values
last_lp_cnn_ema = 0.5 # Initialize assuming eyes are open
last_rp_cnn_ema = 0.5
last_left_ear_avg = 0.3 # Initialize assuming eyes are open (EAR > threshold)
last_right_ear_avg = 0.3
last_cnn_yawn_prob = 0.0
# Last known bounding boxes
last_left_bbox = None
last_right_bbox = None
last_mouth_bbox = None

# --- PiCamera2 Initialization ---
print("Initializing PiCamera2...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
    # --- ADDED CAMERA CONTROLS ---
    controls={
        "AwbEnable": True, # Enable Auto White Balance
        "AwbMode": Controls.AwbModeEnum.Auto, # Set AWB mode (Auto is default, try others if needed: Incandescent, Fluorescent, Daylight etc.)
        "AeEnable": True # Enable Auto Exposure
    }
)
config["transform"] = Transform(vflip=True, hflip=True) # Apply flip
picam2.configure(config)
try:
    picam2.start()
    print("Waiting for camera to stabilize...")
    time.sleep(2.0) # Give camera time to adjust settings
    print("Camera started.")
except Exception as e:
    print(f"FATAL: Error starting camera: {e}")
    GPIO.cleanup()
    exit()

# --- Main Loop ---
print("Starting detection loop...")
try:
    while True:
        start_time = time.time()

        # --- 1. Frame Capture ---
        frame_rgb = picam2.capture_array()
        if frame_rgb is None:
            # print("Warning: Failed to capture frame.") # Optional debug
            time.sleep(0.05) # Wait briefly before retrying
            continue

        # Ensure correct shape (some captures might be RGBA or different)
        if frame_rgb.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2RGB)
        elif frame_rgb.shape[2] != 3:
             print(f"Warning: Unexpected frame channels: {frame_rgb.shape[2]}. Skipping frame.")
             continue


        # Convert to BGR for OpenCV drawing
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        frame_count += 1
        run_inference = (frame_count % INFERENCE_INTERVAL == 0)

        # --- 2. Face Landmark Detection ---
        frame_rgb.flags.writeable = False # Optimization
        results = face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        face_detected = bool(results.multi_face_landmarks)

        # --- 3. Process Face Data ---
        if face_detected:
            face_landmarks = results.multi_face_landmarks[0]

            # --- 3a. Calculate Metrics (MAR, Head Tilt, EAR) ---
            current_mar = calculate_mar(face_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
            mar_buffer.append(current_mar)
            smoothed_mar = np.mean(list(mar_buffer)) if mar_buffer else 0.0

            head_angle = detect_head_tilt(face_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
            head_tilt_detected = abs(head_angle) > HEAD_TILT_THRESHOLD

            # Calculate EAR for both eyes
            left_ear_landmarks = [face_landmarks.landmark[i] for i in LEFT_EYE_EAR_INDICES]
            right_ear_landmarks = [face_landmarks.landmark[i] for i in RIGHT_EYE_EAR_INDICES]
            current_left_ear = calculate_ear(left_ear_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
            current_right_ear = calculate_ear(right_ear_landmarks, FRAME_WIDTH, FRAME_HEIGHT)

            # Update EAR buffers and calculate smoothed EAR
            left_ear_buffer.append(current_left_ear)
            right_ear_buffer.append(current_right_ear)
            left_ear_avg = np.mean(list(left_ear_buffer)) if left_ear_buffer else last_left_ear_avg
            right_ear_avg = np.mean(list(right_ear_buffer)) if right_ear_buffer else last_right_ear_avg
            last_left_ear_avg = left_ear_avg # Store for fallback
            last_right_ear_avg = right_ear_avg

            # --- 3b. Get Bounding Boxes ---
            # Use the more robust get_scaled_bbox function
            current_left_bbox = get_scaled_bbox(face_landmarks, LEFT_EYE_BBOX_INDICES, FRAME_WIDTH, FRAME_HEIGHT, scale=EYE_BBOX_SCALE, min_dim=8, last_bbox=last_left_bbox)
            current_right_bbox = get_scaled_bbox(face_landmarks, RIGHT_EYE_BBOX_INDICES, FRAME_WIDTH, FRAME_HEIGHT, scale=EYE_BBOX_SCALE, min_dim=8, last_bbox=last_right_bbox)

            # Mouth bounding box
            mouth_points = np.array([[int(face_landmarks.landmark[i].x * FRAME_WIDTH),
                                      int(face_landmarks.landmark[i].y * FRAME_HEIGHT)]
                                     for i in MOUTH_LANDMARKS
                                     if 0 <= face_landmarks.landmark[i].x <= 1 and 0 <= face_landmarks.landmark[i].y <= 1])
            current_mouth_bbox = None
            if mouth_points.shape[0] > 2:
                try:
                    x_m, y_m, w_m, h_m = cv2.boundingRect(mouth_points)
                    # Use scaled bbox logic for mouth too (optional, simple bbox might be fine)
                    center_x, center_y = x_m + w_m / 2, y_m + h_m / 2
                    scaled_w_m = w_m * 1.3 # Adjust scale as needed
                    scaled_h_m = h_m * 1.4
                    mx1 = max(0, int(center_x - scaled_w_m / 2))
                    my1 = max(0, int(center_y - scaled_h_m / 2))
                    mx2 = min(FRAME_WIDTH - 1, int(center_x + scaled_w_m / 2))
                    my2 = min(FRAME_HEIGHT - 1, int(center_y + scaled_h_m / 2))
                    if mx1 < mx2 and my1 < my2:
                        current_mouth_bbox = (mx1, my1, mx2, my2)
                except Exception as e:
                    # print(f"Error calculating mouth bbox: {e}") # Optional debug
                    current_mouth_bbox = last_mouth_bbox # Fallback

            # Update last known good bboxes
            last_left_bbox = current_left_bbox if current_left_bbox else last_left_bbox
            last_right_bbox = current_right_bbox if current_right_bbox else last_right_bbox
            last_mouth_bbox = current_mouth_bbox if current_mouth_bbox else last_mouth_bbox

            # --- 3c. Run CNN Inference (Periodically) ---
            # Use last known EMA/Prob values if not running inference this frame
            lp_cnn_ema = last_lp_cnn_ema
            rp_cnn_ema = last_rp_cnn_ema
            cnn_yawn_prob = last_cnn_yawn_prob

            if run_inference:
                # Left Eye CNN
                if last_left_bbox:
                    lx1, ly1, lx2, ly2 = last_left_bbox
                    left_eye_img_crop = frame_bgr[ly1:ly2, lx1:lx2]
                    left_eye_input = preprocess_eye(left_eye_img_crop, eye_input_details)
                    left_output = predict_tflite(eye_interpreter, eye_input_details, eye_output_details, left_eye_input)
                    if left_output is not None:
                        lp_cnn = left_output[0][0] # Raw CNN prob
                        # Apply EMA smoothing
                        prev_lp_cnn = left_eye_cnn_buffer[-1] if left_eye_cnn_buffer else lp_cnn
                        lp_cnn_ema = EYE_STATE_EMA_ALPHA * lp_cnn + (1 - EYE_STATE_EMA_ALPHA) * prev_lp_cnn
                        last_lp_cnn_ema = lp_cnn_ema # Store smoothed value

                # Right Eye CNN
                if last_right_bbox:
                    rx1, ry1, rx2, ry2 = last_right_bbox
                    right_eye_img_crop = frame_bgr[ry1:ry2, rx1:rx2]
                    right_eye_input = preprocess_eye(right_eye_img_crop, eye_input_details)
                    right_output = predict_tflite(eye_interpreter, eye_input_details, eye_output_details, right_eye_input)
                    if right_output is not None:
                        rp_cnn = right_output[0][0] # Raw CNN prob
                        # Apply EMA smoothing
                        prev_rp_cnn = right_eye_cnn_buffer[-1] if right_eye_cnn_buffer else rp_cnn
                        rp_cnn_ema = EYE_STATE_EMA_ALPHA * rp_cnn + (1 - EYE_STATE_EMA_ALPHA) * prev_rp_cnn
                        last_rp_cnn_ema = rp_cnn_ema

                # Mouth Yawn CNN
                if last_mouth_bbox:
                    mx1, my1, mx2, my2 = last_mouth_bbox
                    mouth_img_crop = frame_bgr[my1:my2, mx1:mx2]
                    mouth_input = preprocess_mouth(mouth_img_crop, yawn_input_details)
                    yawn_output = predict_tflite(yawn_interpreter, yawn_input_details, yawn_output_details, mouth_input)
                    if yawn_output is not None:
                        cnn_yawn_prob = float(yawn_output[0][0])
                        last_cnn_yawn_prob = cnn_yawn_prob # Store raw prob for buffer

            # --- 3d. Update State Buffers (Post-Inference/Fallback) ---
            left_eye_cnn_buffer.append(lp_cnn_ema) # Store EMA smoothed CNN value
            right_eye_cnn_buffer.append(rp_cnn_ema)
            cnn_yawn_buffer.append(cnn_yawn_prob > CNN_YAWN_PROB_THRESHOLD) # Store boolean yawn state
            yawn_mar_trigger_buffer.append(smoothed_mar > MAR_THRESHOLD) # Store boolean MAR state

            # --- 3e. Determine Eye Closure (CNN + EAR) ---
            # An eye is closed if EITHER the CNN probability is low OR the EAR is low
            left_cnn_closed = lp_cnn_ema < EYE_CLOSED_THRESHOLD_CNN_EMA
            left_ear_closed = left_ear_avg < EAR_THRESHOLD
            left_eye_closed = left_cnn_closed or left_ear_closed # Combine using OR

            right_cnn_closed = rp_cnn_ema < EYE_CLOSED_THRESHOLD_CNN_EMA
            right_ear_closed = right_ear_avg < EAR_THRESHOLD
            right_eye_closed = right_cnn_closed or right_ear_closed # Combine using OR

            both_eyes_closed = left_eye_closed and right_eye_closed
            eye_closed_buffer.append(both_eyes_closed) # Store combined state for drowsiness check

            # --- 3f. Drowsiness Logic (Based on combined eye state) ---
            closed_count_in_window = sum(list(eye_closed_buffer)) # Count True values in buffer

            if not drowsy and closed_count_in_window >= EYE_CLOSED_COUNT_THRESHOLD:
                print(f"DROWSINESS DETECTED: Closed count {closed_count_in_window}/{EYE_CLOSED_COUNT_THRESHOLD} in last {EYE_CLOSED_BUFFER_LENGTH} frames.")
                drowsy = True
                consecutive_open_eye_frames = 0
            elif drowsy:
                if not both_eyes_closed: # Eyes opened
                    consecutive_open_eye_frames += 1
                    if consecutive_open_eye_frames >= DROWSY_RESET_FRAMES:
                        print(f"Drowsy state reset after {DROWSY_RESET_FRAMES} open frames.")
                        drowsy = False
                        eye_closed_buffer.clear() # Clear buffer on recovery
                        consecutive_open_eye_frames = 0
                else: # Eyes still closed
                    consecutive_open_eye_frames = 0 # Reset open counter

            # --- 3g. Yawn Logic (CNN + MAR) ---
            cnn_yawn_count = sum(cnn_yawn_buffer)
            is_cnn_yawning = cnn_yawn_count >= CNN_YAWN_FRAMES_TRIGGER

            mar_yawn_count = sum(yawn_mar_trigger_buffer)
            # Trigger MAR yawn only if not in cooldown and enough frames exceeded threshold
            is_mar_yawning_trigger = (yawn_cooldown_counter == 0 and
                                      mar_yawn_count >= YAWN_MAR_FRAMES_THRESHOLD)

            if is_mar_yawning_trigger:
                print(f"MAR YAWN TRIGGERED: Count {mar_yawn_count}/{YAWN_MAR_FRAMES_THRESHOLD}. Cooldown started.")
                yawn_cooldown_counter = YAWN_COOLDOWN_FRAMES
                yawn_mar_trigger_buffer.clear() # Reset buffer after trigger

            # Decrement cooldown counter if active
            if yawn_cooldown_counter > 0:
                yawn_cooldown_counter -= 1

            # Final yawn state: either CNN or MAR triggered it
            # Note: MAR only triggers *once* per cooldown period
            yawning = is_cnn_yawning or is_mar_yawning_trigger

            # --- 3h. Trigger Alerts ---
            current_alert = None
            if yawning and drowsy: current_alert = "SEVERE"
            elif yawning: current_alert = "YAWN"
            elif drowsy: current_alert = "EYES"
            elif head_tilt_detected: current_alert = "TILT"

            if current_alert:
                trigger_buzzer(current_alert, BUZZER_PATTERNS[current_alert])


            # --- 4. Drawing / Visualization ---
            # Draw Eye BBoxes and State
            if last_left_bbox:
                lx1, ly1, lx2, ly2 = last_left_bbox
                left_color = (0, 0, 255) if left_eye_closed else (0, 255, 0)
                cv2.rectangle(frame_bgr, (lx1, ly1), (lx2, ly2), left_color, 1)
                cv2.putText(frame_bgr, f"CNN:{lp_cnn_ema:.2f}", (lx1, ly1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, left_color, 1)
                cv2.putText(frame_bgr, f"EAR:{left_ear_avg:.2f}", (lx1, ly1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, left_color, 1)
            if last_right_bbox:
                rx1, ry1, rx2, ry2 = last_right_bbox
                right_color = (0, 0, 255) if right_eye_closed else (0, 255, 0)
                cv2.rectangle(frame_bgr, (rx1, ry1), (rx2, ry2), right_color, 1)
                cv2.putText(frame_bgr, f"CNN:{rp_cnn_ema:.2f}", (rx1, ry1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, right_color, 1)
                cv2.putText(frame_bgr, f"EAR:{right_ear_avg:.2f}", (rx1, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, right_color, 1)

            # Draw Mouth BBox and Yawn Info
            if last_mouth_bbox:
                mx1, my1, mx2, my2 = last_mouth_bbox
                mouth_color = (0, 255, 255) if yawning else (255, 255, 0)
                cv2.rectangle(frame_bgr, (mx1, my1), (mx2, my2), mouth_color, 1)
                cv2.putText(frame_bgr, f"MAR:{smoothed_mar:.2f}", (mx1, my1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
                cv2.putText(frame_bgr, f"YawnProb:{cnn_yawn_prob:.2f}", (mx1, my1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, mouth_color, 1)

            # Display Status Text (Bottom Left)
            info_y_start = FRAME_HEIGHT - 60
            txt_color = (255, 255, 255)
            cv2.putText(frame_bgr, f"CNN Yawn Buf: {cnn_yawn_count}/{CNN_YAWN_FRAMES_TRIGGER}", (10, info_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1)
            cv2.putText(frame_bgr, f"MAR Yawn Buf: {mar_yawn_count}/{YAWN_MAR_FRAMES_THRESHOLD} (CD:{yawn_cooldown_counter})", (10, info_y_start + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1)
            cv2.putText(frame_bgr, f"Eye Close Buf: {closed_count_in_window}/{EYE_CLOSED_COUNT_THRESHOLD}", (10, info_y_start + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1)
            tilt_color = (0, 165, 255) if head_tilt_detected else txt_color
            cv2.putText(frame_bgr, f"Tilt:{head_angle:.1f}deg", (10, info_y_start + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, tilt_color, 1)

            # Display Overall Status (Top Left)
            final_status = "Awake"
            status_color = (0, 255, 0) # Green
            if current_alert == "SEVERE":
                final_status = "SEVERE DROWSINESS"
                status_color = (0, 0, 255) # Red
            elif current_alert == "YAWN":
                 final_status = "Yawning"
                 status_color = (0, 255, 255) # Yellow
            elif current_alert == "EYES":
                 final_status = "Drowsy"
                 status_color = (0, 165, 255) # Orange
            elif current_alert == "TILT":
                 final_status = "Head Tilt"
                 status_color = (0, 165, 255) # Orange

            cv2.putText(frame_bgr, f"Status: {final_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # --- No Face Detected ---
        else:
            # Reset states if face is lost
            if any([drowsy, yawning, head_tilt_detected, left_eye_cnn_buffer, cnn_yawn_buffer]):
                 print("No face detected, resetting state.")
                 left_eye_cnn_buffer.clear(); right_eye_cnn_buffer.clear()
                 left_ear_buffer.clear(); right_ear_buffer.clear()
                 mar_buffer.clear(); yawn_mar_trigger_buffer.clear()
                 eye_closed_buffer.clear(); cnn_yawn_buffer.clear()
                 drowsy = yawning = head_tilt_detected = False
                 consecutive_open_eye_frames = yawn_cooldown_counter = 0
                 # Reset last known values
                 last_lp_cnn_ema = last_rp_cnn_ema = 0.5
                 last_left_ear_avg = last_right_ear_avg = 0.3
                 last_cnn_yawn_prob = 0.0
                 # Clear last known bboxes immediately
                 last_left_bbox = last_right_bbox = last_mouth_bbox = None

            cv2.putText(frame_bgr, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- 5. Display Frame and FPS ---
        end_time = time.time()
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (FRAME_WIDTH - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow("Driver Fatigue Detection", frame_bgr)

        # --- 6. Exit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

except KeyboardInterrupt:
    print("Interrupted by user.")
except Exception as e:
    print(f"An unexpected error occurred in the main loop: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("Cleaning up...")
    if 'picam2' in locals() and picam2.started:
        picam2.stop()
        print("PiCamera2 stopped.")
    cv2.destroyAllWindows()
    # Ensure pygame mixer is quit
    if pygame.mixer.get_init():
        pygame.mixer.quit()
        print("Pygame mixer quit.")
    GPIO.cleanup()
    print("GPIO cleanup complete.")
    print("Cleanup complete.")