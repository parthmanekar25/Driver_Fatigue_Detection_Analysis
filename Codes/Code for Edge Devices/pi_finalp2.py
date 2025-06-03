import cv2
import mediapipe as mp
import numpy as np
import tflite_runtime.interpreter as tflite # Specific for Raspberry Pi
from collections import deque
import RPi.GPIO as GPIO                     # Specific for Raspberry Pi
import time
import os
from picamera2 import Picamera2, Controls   # Specific for Raspberry Pi

# --- Configuration ---
# --- Constants ---
# !!! IMPORTANT: Ensure this path is correct for your Pi setup !!!
MODEL_DIR = "/home/pi/drowsiness_detector/"
EYE_MODEL_PATH = os.path.join(MODEL_DIR, "eye_model.tflite")    # Verify filename
YAWN_MODEL_PATH = os.path.join(MODEL_DIR, "yawn_model.tflite")  # Verify filename

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# MediaPipe Indices (Standard)
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 153, 144, 145, 246]
RIGHT_EYE_INDICES = [362, 263, 385, 386, 387, 380, 373, 374, 466]
MOUTH_LANDMARKS = list(set([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324,
                            318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267,
                            269, 270, 409, 415, 310, 311, 312, 13]))

# Buffers & Smoothing
EYE_STATE_BUFFER_LENGTH = 10  # Number of frames for eye state smoothing
EYE_STATE_EMA_ALPHA = 0.5     # Smoothing factor (higher = more responsive)
MAR_BUFFER_LENGTH = 5         # Number of frames for MAR smoothing
EYE_CLOSED_BUFFER_LENGTH = 30 # Lookback window for counting closed frames
CNN_YAWN_BUFFER_LENGTH = 25   # Lookback window for counting CNN yawn frames

# --- Thresholds (!!! CRITICAL: TUNE THESE VALUES based on testing !!!) ---
# These are starting points. Observe output values and adjust.
# Model output > threshold means OPEN, < threshold means CLOSED
EYE_CLOSED_THRESHOLD_EMA = 0.60 # Tune based on eye model output for your lighting
MAR_THRESHOLD = 0.5             # Tune based on observed MAR values during yawns
YAWN_MAR_FRAMES_THRESHOLD = 15  # How long MAR must be high
CNN_YAWN_PROB_THRESHOLD = 0.75  # Tune based on yawn model output probability
CNN_YAWN_FRAMES_TRIGGER = 18    # How many frames prob must be high
EYE_CLOSED_COUNT_THRESHOLD = 8  # Frames closed within buffer to trigger drowsy
HEAD_TILT_THRESHOLD = 18        # Max head tilt angle (degrees) before alert
DROWSY_RESET_FRAMES = 30        # Consecutive open frames needed to reset drowsy state
YAWN_COOLDOWN_FRAMES = 40       # Frames to wait after a MAR yawn
ALERT_COOLDOWN_SECONDS = 4.0    # Min time between distinct alert triggers

# Model/Processing Settings
EYE_CNN_INPUT_SIZE = (128, 128)  # Verify with your eye model's expected input shape
YAWN_CNN_INPUT_SIZE = (96, 96)   # Verify with your yawn model's expected input shape
INFERENCE_INTERVAL = 3           # Run CNN inference every N frames
EYE_BBOX_SCALE = 1.5             # Scaling factor for eye bounding box (Adjust if needed)
MOUTH_BBOX_SCALE_W = 1.5         # Scaling factor for mouth bbox width
MOUTH_BBOX_SCALE_H = 1.6         # Scaling factor for mouth bbox height

# GPIO Setup for Buzzer
BUZZER_PIN = 17 # Verify this matches your wiring
GPIO.setwarnings(False)      # Disable GPIO warnings
GPIO.setmode(GPIO.BCM)       # Use Broadcom pin numbering
GPIO.setup(BUZZER_PIN, GPIO.OUT) # Set pin as output
GPIO.output(BUZZER_PIN, GPIO.LOW) # Ensure buzzer is off initially
buzzer_active_until = 0       # Timestamp until the buzzer pattern finishes

# --- Buzzer Control ---
def trigger_buzzer(alert_type, duration_pattern):
    """Activates the buzzer with a specific pattern if cooldown allows."""
    global buzzer_active_until, last_alert_trigger_time # Need global for last_alert_trigger_time
    current_time = time.time()

    # Check main alert cooldown first
    if current_time < last_alert_trigger_time + ALERT_COOLDOWN_SECONDS:
        return

    # Check if a pattern is already playing (using buzzer_active_until)
    # This prevents interrupting a pattern, although the main cooldown mostly handles this.
    if current_time < buzzer_active_until:
       return

    print(f"--- ALERT DETECTED: {alert_type} ---")
    total_duration = 0
    try:
        # Play the pattern: list of (on_time, off_time) tuples
        for on_time, off_time in duration_pattern:
            if on_time > 0:
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                time.sleep(on_time)
                total_duration += on_time
            # Always turn off between pulses or at the end
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            if off_time > 0:
                time.sleep(off_time)
                total_duration += off_time
        # Ensure buzzer is off after the pattern finishes
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        # Update when the current pattern finishes playing
        buzzer_active_until = current_time + total_duration + 0.05 # Small buffer
        # Update the main alert trigger time AFTER successfully starting the alert
        last_alert_trigger_time = current_time
    except Exception as e:
        print(f"GPIO Error during buzzer activation: {e}")
        # Ensure buzzer is turned off in case of error
        try:
            GPIO.output(BUZZER_PIN, GPIO.LOW)
        except Exception: # Catch potential errors during cleanup itself
            pass

# Define buzzer patterns (On duration, Off duration)
BUZZER_PATTERNS = {
    "EYES": [(0.1, 0.1), (0.1, 0.1)],         # Short double beep
    "YAWN": [(0.4, 0.2)],                     # Longer single beep
    "TILT": [(0.15, 0.1), (0.15, 0.3)],       # Two distinct beeps
    "SEVERE": [(0.2, 0.05), (0.2, 0.05), (0.2, 0.4)] # Urgent triple beep
}
# Initialize last alert time to allow immediate first alert
last_alert_trigger_time = 0

# --- TFLite Model Loading ---
def load_tflite_model(model_path):
    """Loads a TFLite model using tflite_runtime and prints details."""
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"Successfully loaded TFLite model: {os.path.basename(model_path)}")
        print(f"  Input Details: Shape={input_details[0]['shape']}, Type={input_details[0]['dtype']}")
        print(f"  Output Details: Shape={output_details[0]['shape']}, Type={output_details[0]['dtype']}")
        return interpreter, input_details, output_details
    except Exception as e:
        print(f"FATAL ERROR: Could not load TFLite model from {model_path}: {e}")
        print("Check file path, permissions, and if tflite_runtime is installed correctly.")
        return None, None, None

print("Loading TFLite models...")
eye_interpreter, eye_input_details, eye_output_details = load_tflite_model(EYE_MODEL_PATH)
yawn_interpreter, yawn_input_details, yawn_output_details = load_tflite_model(YAWN_MODEL_PATH)

if not eye_interpreter or not yawn_interpreter:
    print("Exiting due to model loading failure.")
    # Optional: Add GPIO cleanup here if needed before exit
    # GPIO.cleanup()
    exit()

# --- MediaPipe Initialization ---
print("Initializing MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,          # Assume only one driver
    refine_landmarks=True,    # Get detailed landmarks for eyes/mouth
    min_detection_confidence=0.6, # User-adjusted - Lower may increase detections but also false positives
    min_tracking_confidence=0.6  # User-adjusted - Lower may help maintain tracking but risk inaccuracy
)
print("MediaPipe initialized.")

# --- Helper Functions ---

def preprocess_eye(eye_img, input_details):
    """
    Prepares the extracted eye image for the TFLite eye model.
    Uses CLAHE for contrast enhancement.
    NOTE: Bilateral filter was removed in the user's provided code.
    """
    # Basic validation
    if eye_img is None or eye_img.size == 0 or eye_img.shape[0] < 5 or eye_img.shape[1] < 5:
        return None

    input_shape = input_details[0]['shape'] # e.g., [1, 128, 128, 1]
    target_size = (input_shape[2], input_shape[1]) # (width, height)
    expected_channels = input_shape[3]

    # Ensure Grayscale
    if len(eye_img.shape) == 3 and eye_img.shape[2] == 3:
        eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    elif len(eye_img.shape) == 2:
        eye_gray = eye_img
    else:
        return None # Cannot process unexpected shape

    # --- Preprocessing Steps ---
    # 1. Adaptive Histogram Equalization (CLAHE) - Enhances local contrast
    # Useful for varying lighting, especially shadows/highlights
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eye_eq = clahe.apply(eye_gray)

    # (Bilateral Filter was removed here in the user's code)

    # 2. Resize to model's expected input size
    eye_resized = cv2.resize(eye_eq, target_size, interpolation=cv2.INTER_AREA)

    # 3. Normalize pixel values to [0, 1] (assuming model expects this)
    eye_normalized = eye_resized.astype(np.float32) / 255.0

    # 4. Add batch and channel dimensions (e.g., shape -> [1, H, W, C])
    if expected_channels == 1:
        eye_input = np.expand_dims(np.expand_dims(eye_normalized, axis=-1), axis=0)
    # Add elif for expected_channels == 3 if your eye model needs color
    else:
        print(f"Warning: Eye model expects {expected_channels} channels. Preprocessing assumes 1 (grayscale).")
        # Fallback assuming grayscale, might need adjustment if model expects color
        eye_input = np.expand_dims(np.expand_dims(eye_normalized, axis=-1), axis=0)


    # Final shape check
    if eye_input.shape != tuple(input_shape):
        print(f"ERROR: Preprocessed eye shape {eye_input.shape} mismatch model input {tuple(input_shape)}")
        return None

    return eye_input

def preprocess_mouth(mouth_img, input_details):
    """
    Prepares the extracted mouth image for the TFLite yawn model.
    Handles grayscale (CLAHE) or color (HSV equalization) based on model's expected input channels.
    """
    if mouth_img is None or mouth_img.size == 0 or mouth_img.shape[0] < 10 or mouth_img.shape[1] < 10:
        return None

    input_shape = input_details[0]['shape'] # e.g., [1, 96, 96, 3] or [1, 96, 96, 1]
    target_size = (input_shape[2], input_shape[1]) # (width, height)
    expected_channels = input_shape[3]
    processed_img = None

    try:
        # --- Adjust processing based on model's expected channels ---
        if expected_channels == 3: # Model expects color image
            if len(mouth_img.shape) == 2: # Convert if grayscale
                mouth_img = cv2.cvtColor(mouth_img, cv2.COLOR_GRAY2BGR)
            elif len(mouth_img.shape) == 3 and mouth_img.shape[2] == 1: # Convert if [H, W, 1]
                mouth_img = cv2.cvtColor(mouth_img, cv2.COLOR_GRAY2BGR)
            # Apply HSV equalization to the Value channel
            hsv = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            processed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif expected_channels == 1: # Model expects grayscale image
            if len(mouth_img.shape) == 3 and mouth_img.shape[2] == 3: # Convert if color
                processed_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)
            elif len(mouth_img.shape) == 3 and mouth_img.shape[2] == 1: # Squeeze if [H, W, 1]
                processed_img = mouth_img.squeeze(axis=-1)
            elif len(mouth_img.shape) == 2: # Already grayscale
                 processed_img = mouth_img
            else: return None # Invalid shape

            # Apply CLAHE for grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed_img = clahe.apply(processed_img)
        else:
            print(f"ERROR: Unsupported channel count in yawn model: {expected_channels}")
            return None

        # Resize
        mouth_resized = cv2.resize(processed_img, target_size, interpolation=cv2.INTER_AREA)

        # Normalize
        mouth_normalized = mouth_resized.astype(np.float32) / 255.0

        # Add channel dimension if needed (for grayscale) and batch dimension
        if expected_channels == 1 and len(mouth_normalized.shape) == 2:
            mouth_normalized = np.expand_dims(mouth_normalized, axis=-1) # Add channel dim

        mouth_input = np.expand_dims(mouth_normalized, axis=0) # Add batch dim

        # Final shape check
        if mouth_input.shape != tuple(input_shape):
            print(f"ERROR: Preprocessed mouth shape {mouth_input.shape} mismatch model input {tuple(input_shape)}")
            return None

        return mouth_input

    except Exception as e:
        print(f"Error preprocessing mouth image: {e}")
        return None


def predict_tflite(interpreter, input_details, output_details, data):
    """Runs inference on the loaded TFLite model."""
    if data is None: return None
    if interpreter is None: return None
    try:
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data
    except Exception as e:
        print(f"Error during TFLite inference: {e}")
        return None

def get_scaled_bbox(landmarks, indices, width, height, scale=1.0, last_bbox=None):
    """Calculates a bounding box around specified landmarks and scales it."""
    # Renamed from get_eye_bbox to be more generic
    try:
        points = np.array([(landmarks.landmark[i].x * width, landmarks.landmark[i].y * height)
                           for i in indices
                           if 0 <= landmarks.landmark[i].x <= 1 and 0 <= landmarks.landmark[i].y <= 1],
                          dtype=np.int32)

        if len(points) < 3: return last_bbox

        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        bbox_w, bbox_h = x_max - x_min, y_max - y_min

        if bbox_w <= 0 or bbox_h <= 0: return last_bbox

        scaled_w, scaled_h = bbox_w * scale, bbox_h * scale

        x1 = max(0, int(center_x - scaled_w / 2))
        y1 = max(0, int(center_y - scaled_h / 2))
        x2 = min(width - 1, int(center_x + scaled_w / 2))
        y2 = min(height - 1, int(center_y + scaled_h / 2))

        if x1 >= x2 or y1 >= y2: return last_bbox

        return x1, y1, x2, y2
    except Exception as e:
        # print(f"Error calculating scaled bbox: {e}") # Optional debug print
        return last_bbox

def calculate_mar(landmarks, width, height):
    """Calculates the Mouth Aspect Ratio."""
    try:
        v_upper_lip = landmarks.landmark[13].y * height
        v_lower_lip = landmarks.landmark[14].y * height
        h_left_corner = landmarks.landmark[61].x * width
        h_right_corner = landmarks.landmark[291].x * width
        mouth_height = abs(v_lower_lip - v_upper_lip)
        mouth_width = abs(h_right_corner - h_left_corner)
        if mouth_width < 1e-6: return 0.0
        return mouth_height / mouth_width
    except (IndexError, AttributeError, Exception): # Catch potential errors
        return 0.0

def detect_head_tilt(landmarks, width, height):
    """Calculates head tilt angle based on inner eye corners."""
    try:
        left_eye_inner = landmarks.landmark[133]
        right_eye_inner = landmarks.landmark[362]
        if not (0 <= left_eye_inner.x <= 1 and 0 <= left_eye_inner.y <= 1 and \
                0 <= right_eye_inner.x <= 1 and 0 <= right_eye_inner.y <= 1):
            return 0.0
        dx = (right_eye_inner.x - left_eye_inner.x) * width
        dy = (right_eye_inner.y - left_eye_inner.y) * height
        if abs(dx) < 1e-6: # Avoid division by zero / handle vertical line
            return 90.0 * np.sign(dy) if dy != 0 else 0.0
        return np.degrees(np.arctan2(dy, dx))
    except (IndexError, AttributeError, Exception): # Catch potential errors
        return 0.0

# --- Buffers and State Variables ---
left_eye_state_buffer = deque(maxlen=EYE_STATE_BUFFER_LENGTH)
right_eye_state_buffer = deque(maxlen=EYE_STATE_BUFFER_LENGTH)
mar_buffer = deque(maxlen=MAR_BUFFER_LENGTH)
yawn_mar_trigger_buffer = deque(maxlen=YAWN_MAR_FRAMES_THRESHOLD)
eye_closed_buffer = deque(maxlen=EYE_CLOSED_BUFFER_LENGTH)
cnn_yawn_buffer = deque(maxlen=CNN_YAWN_BUFFER_LENGTH)

# State flags and counters
frame_count = 0
drowsy = False
yawning = False
head_tilt_detected = False
consecutive_open_eye_frames = 0
yawn_cooldown_counter = 0

# Last known values for non-inference frames and fallback
last_lp_ema = 0.5 # Assume initially open
last_rp_ema = 0.5
last_cnn_yawn_prob = 0.0
last_left_bbox = None
last_right_bbox = None
last_mouth_bbox = None

# --- PiCamera2 Initialization ---
print("Initializing PiCamera2...")
picam2 = Picamera2()
# Configure for preview and capture, use RGB format compatible with MediaPipe
config = picam2.create_preview_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
    # Enable Auto White Balance and Auto Exposure for varying light conditions
    controls={"AwbEnable": True, "AwbMode": Controls.AwbModeEnum.Auto, "AeEnable": True}
    # Optional: Adjust AE exposure mode if needed, e.g., Controls.AeExposureModeEnum.Short/Long
    # controls={"AwbEnable": True, "AwbMode": Controls.AwbModeEnum.Auto, "AeEnable": True, "AeExposureMode": Controls.AeExposureModeEnum.Short}
)
picam2.configure(config)
picam2.start()
# Allow camera to adjust settings
time.sleep(2.0)
print("Camera started.")

# --- Main Loop ---
print("Starting detection loop... Press Ctrl+C to exit.")
try:
    while True:
        start_time = time.time()

        # --- 1. Frame Capture (PiCamera2) ---
        # Capture frame as RGB numpy array
        frame_rgb = picam2.capture_array()
        if frame_rgb is None:
            print("Warning: Failed to capture frame from PiCamera2.")
            time.sleep(0.05) # Wait briefly before retrying
            continue

        # Convert to BGR for OpenCV drawing functions later
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        frame_count += 1
        run_inference = (frame_count % INFERENCE_INTERVAL == 0)

        # --- 2. Face Landmark Detection (MediaPipe) ---
        # Process the RGB frame (MediaPipe prefers RGB)
        frame_rgb.flags.writeable = False # Performance optimization
        results = face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True # Re-enable writing for potential drawing on RGB

        # --- 3. Process Results ---
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0] # Assuming one face

            # --- 3a. Calculate Metrics ---
            current_mar = calculate_mar(face_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
            mar_buffer.append(current_mar)
            smoothed_mar = np.mean(list(mar_buffer)) if mar_buffer else 0.0

            head_angle = detect_head_tilt(face_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
            head_tilt_detected = abs(head_angle) > HEAD_TILT_THRESHOLD

            # --- 3b. Get Bounding Boxes ---
            # Use the updated get_scaled_bbox function
            current_left_bbox = get_scaled_bbox(face_landmarks, LEFT_EYE_INDICES, FRAME_WIDTH, FRAME_HEIGHT, scale=EYE_BBOX_SCALE, last_bbox=last_left_bbox)
            current_right_bbox = get_scaled_bbox(face_landmarks, RIGHT_EYE_INDICES, FRAME_WIDTH, FRAME_HEIGHT, scale=EYE_BBOX_SCALE, last_bbox=last_right_bbox)

            # Mouth bounding box calculation
            mouth_points = np.array([[int(face_landmarks.landmark[i].x * FRAME_WIDTH),
                                      int(face_landmarks.landmark[i].y * FRAME_HEIGHT)]
                                     for i in MOUTH_LANDMARKS
                                     if 0 <= face_landmarks.landmark[i].x <= 1 and 0 <= face_landmarks.landmark[i].y <= 1])
            current_mouth_bbox = None
            if mouth_points.shape[0] > 2:
                x_m, y_m, w_m, h_m = cv2.boundingRect(mouth_points)
                center_x, center_y = x_m + w_m / 2, y_m + h_m / 2
                scaled_w_m = w_m * MOUTH_BBOX_SCALE_W
                scaled_h_m = h_m * MOUTH_BBOX_SCALE_H
                mx1 = max(0, int(center_x - scaled_w_m / 2))
                my1 = max(0, int(center_y - scaled_h_m / 2))
                mx2 = min(FRAME_WIDTH - 1, int(center_x + scaled_w_m / 2))
                my2 = min(FRAME_HEIGHT - 1, int(center_y + scaled_h_m / 2))
                if mx1 < mx2 and my1 < my2:
                    current_mouth_bbox = (mx1, my1, mx2, my2)

            # Update last known good bboxes (use current if valid, else keep last)
            last_left_bbox = current_left_bbox if current_left_bbox else last_left_bbox
            last_right_bbox = current_right_bbox if current_right_bbox else last_right_bbox
            last_mouth_bbox = current_mouth_bbox if current_mouth_bbox else last_mouth_bbox

            # --- 3c. Run CNN Inference (Periodically) ---
            lp_ema, rp_ema, cnn_yawn_prob = last_lp_ema, last_rp_ema, last_cnn_yawn_prob

            if run_inference:
                # --- Left Eye ---
                if last_left_bbox:
                    lx1, ly1, lx2, ly2 = last_left_bbox
                    left_eye_img = frame_bgr[ly1:ly2, lx1:lx2] # Crop from BGR frame
                    left_eye_input = preprocess_eye(left_eye_img, eye_input_details)
                    left_output = predict_tflite(eye_interpreter, eye_input_details, eye_output_details, left_eye_input)
                    if left_output is not None:
                        lp = left_output[0][0] # Assuming output is [[probability]]
                        prev_lp = left_eye_state_buffer[-1] if left_eye_state_buffer else lp
                        lp_ema = EYE_STATE_EMA_ALPHA * lp + (1 - EYE_STATE_EMA_ALPHA) * prev_lp
                        last_lp_ema = lp_ema # Store for non-inference frames

                # --- Right Eye ---
                if last_right_bbox:
                    rx1, ry1, rx2, ry2 = last_right_bbox
                    right_eye_img = frame_bgr[ry1:ry2, rx1:rx2]
                    right_eye_input = preprocess_eye(right_eye_img, eye_input_details)
                    right_output = predict_tflite(eye_interpreter, eye_input_details, eye_output_details, right_eye_input)
                    if right_output is not None:
                        rp = right_output[0][0]
                        prev_rp = right_eye_state_buffer[-1] if right_eye_state_buffer else rp
                        rp_ema = EYE_STATE_EMA_ALPHA * rp + (1 - EYE_STATE_EMA_ALPHA) * prev_rp
                        last_rp_ema = rp_ema

                # --- Mouth (Yawn) ---
                if last_mouth_bbox:
                    mx1, my1, mx2, my2 = last_mouth_bbox
                    mouth_img = frame_bgr[my1:my2, mx1:mx2]
                    mouth_input = preprocess_mouth(mouth_img, yawn_input_details)
                    yawn_output = predict_tflite(yawn_interpreter, yawn_input_details, yawn_output_details, mouth_input)
                    if yawn_output is not None:
                        cnn_yawn_prob = float(yawn_output[0][0]) # Assuming output is [[probability]]
                        last_cnn_yawn_prob = cnn_yawn_prob

            # --- 3d. Update State Buffers ---
            left_eye_state_buffer.append(lp_ema)
            right_eye_state_buffer.append(rp_ema)
            cnn_yawn_buffer.append(cnn_yawn_prob > CNN_YAWN_PROB_THRESHOLD)

            # Determine eye closed status based on the crucial threshold
            # *** Tune EYE_CLOSED_THRESHOLD_EMA based on observed lp_ema/rp_ema values ***
            left_eye_closed = lp_ema < EYE_CLOSED_THRESHOLD_EMA
            right_eye_closed = rp_ema < EYE_CLOSED_THRESHOLD_EMA
            both_eyes_closed = left_eye_closed and right_eye_closed
            eye_closed_buffer.append(both_eyes_closed)

            # Update MAR yawn trigger buffer
            yawn_mar_trigger_buffer.append(smoothed_mar > MAR_THRESHOLD)

            # --- 3e. Drowsiness Logic ---
            # Count recent closed frames (within the lookback window)
            # *** Tune EYE_CLOSED_COUNT_THRESHOLD ***
            closed_count_in_window = sum(list(eye_closed_buffer)[-EYE_CLOSED_COUNT_THRESHOLD:]) # Check last N frames only

            if not drowsy and closed_count_in_window >= EYE_CLOSED_COUNT_THRESHOLD:
                print(f"DROWSY DETECTED: Closed count {closed_count_in_window}/{EYE_CLOSED_COUNT_THRESHOLD}")
                drowsy = True
                consecutive_open_eye_frames = 0 # Reset open counter
            elif drowsy:
                if not both_eyes_closed: # If eyes open while drowsy
                    consecutive_open_eye_frames += 1
                    if consecutive_open_eye_frames >= DROWSY_RESET_FRAMES:
                        print("Drowsy state reset (eyes open).")
                        drowsy = False
                        eye_closed_buffer.clear() # Clear buffer on recovery
                        consecutive_open_eye_frames = 0
                else: # Eyes still closed while drowsy
                    consecutive_open_eye_frames = 0 # Reset open counter

            # --- 3f. Yawn Logic ---
            cnn_yawn_count = sum(cnn_yawn_buffer)
            is_cnn_yawning = cnn_yawn_count >= CNN_YAWN_FRAMES_TRIGGER

            mar_yawn_count = sum(yawn_mar_trigger_buffer)
            is_mar_yawning_trigger = (yawn_cooldown_counter == 0 and
                                      mar_yawn_count >= YAWN_MAR_FRAMES_THRESHOLD)

            if is_mar_yawning_trigger:
                print(f"MAR YAWN TRIGGERED: Count {mar_yawn_count}/{YAWN_MAR_FRAMES_THRESHOLD}")
                yawn_cooldown_counter = YAWN_COOLDOWN_FRAMES
                yawn_mar_trigger_buffer.clear() # Reset buffer

            if yawn_cooldown_counter > 0:
                yawn_cooldown_counter -= 1

            yawning = is_cnn_yawning or is_mar_yawning_trigger # Combine yawn triggers

            # --- 3g. Trigger Alerts ---
            # Conditions checked in order of severity
            alert_type = None
            if yawning and drowsy: alert_type = "SEVERE"
            elif yawning: alert_type = "YAWN"
            elif drowsy: alert_type = "EYES"
            elif head_tilt_detected: alert_type = "TILT"

            if alert_type:
                trigger_buzzer(alert_type, BUZZER_PATTERNS[alert_type])

            # --- 4. Drawing / Visualization on BGR Frame ---
            # Draw Eye BBoxes and State (use last known bbox if detection momentary fails)
            if last_left_bbox:
                lx1, ly1, lx2, ly2 = last_left_bbox
                color = (0, 0, 255) if left_eye_closed else (0, 255, 0) # Red if closed, Green if open
                cv2.rectangle(frame_bgr, (lx1, ly1), (lx2, ly2), color, 1)
                cv2.putText(frame_bgr, f"{lp_ema:.2f}", (lx1, ly1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if last_right_bbox:
                rx1, ry1, rx2, ry2 = last_right_bbox
                color = (0, 0, 255) if right_eye_closed else (0, 255, 0)
                cv2.rectangle(frame_bgr, (rx1, ry1), (rx2, ry2), color, 1)
                cv2.putText(frame_bgr, f"{rp_ema:.2f}", (rx1, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw Mouth BBox and Yawn Probability
            if last_mouth_bbox:
                x_m_pad, y_m_pad, x_max_pad, y_max_pad = last_mouth_bbox
                color = (0, 255, 255) if yawning else (255, 255, 0) # Yellow if yawning, Cyan if not
                cv2.rectangle(frame_bgr, (x_m_pad, y_m_pad), (x_max_pad, y_max_pad), color, 1)
                cv2.putText(frame_bgr, f"YawnP:{cnn_yawn_prob:.2f}", (x_m_pad, y_m_pad - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Display Status Text
            info_y_start = FRAME_HEIGHT - 90 # Position from bottom
            txt_color = (255, 255, 255) # White
            cv2.putText(frame_bgr, f"MAR:{smoothed_mar:.2f}", (10, info_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)
            cv2.putText(frame_bgr, f"CNN Yawn Buf: {cnn_yawn_count}/{CNN_YAWN_FRAMES_TRIGGER}", (10, info_y_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)
            cv2.putText(frame_bgr, f"MAR Yawn Buf: {mar_yawn_count}/{YAWN_MAR_FRAMES_THRESHOLD} (CD:{yawn_cooldown_counter})", (10, info_y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)
            cv2.putText(frame_bgr, f"Eye Close Buf: {closed_count_in_window}/{EYE_CLOSED_COUNT_THRESHOLD}", (10, info_y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)
            tilt_color = (0, 165, 255) if head_tilt_detected else txt_color # Orange if tilted
            cv2.putText(frame_bgr, f"Tilt:{head_angle:.1f}deg", (10, info_y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tilt_color, 1)

            # Overall Status Display
            final_status = "Awake"
            if alert_type == "SEVERE": final_status = "SEVERE Drowsiness"
            elif alert_type == "YAWN": final_status = "Yawning Detected"
            elif alert_type == "EYES": final_status = "Drowsiness Detected"
            elif alert_type == "TILT": final_status = "Head Tilt Detected"

            status_color = (0, 0, 255) if "SEVERE" in final_status or "Drowsiness" in final_status else \
                           (0, 255, 255) if "Yawn" in final_status else \
                           (0, 165, 255) if "Tilt" in final_status else \
                           (0, 255, 0) # Awake = Green

            cv2.putText(frame_bgr, f"Status: {final_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        else:
             # --- No Face Detected ---
            if any([drowsy, yawning, head_tilt_detected, left_eye_state_buffer, cnn_yawn_buffer]):
                print("No face detected, resetting state.")
                # Reset states and buffers
                left_eye_state_buffer.clear(); right_eye_state_buffer.clear()
                mar_buffer.clear(); yawn_mar_trigger_buffer.clear()
                eye_closed_buffer.clear(); cnn_yawn_buffer.clear()
                drowsy = yawning = head_tilt_detected = False
                consecutive_open_eye_frames = yawn_cooldown_counter = 0
                last_lp_ema = last_rp_ema = 0.5 # Reset to default assumption
                last_cnn_yawn_prob = 0.0
                # Optionally clear last known bboxes immediately
                # last_left_bbox = last_right_bbox = last_mouth_bbox = None

            cv2.putText(frame_bgr, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


        # --- 5. Display Frame and FPS ---
        loop_duration = time.time() - start_time
        fps = 1.0 / loop_duration if loop_duration > 0 else 0
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (FRAME_WIDTH - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Display the frame (requires a display connected to the Pi, or use streaming)
        # To run headless, comment out cv2.imshow and cv2.waitKey
        cv2.imshow("Driver Fatigue Detection (Pi)", frame_bgr)

        # --- 6. Handle User Input (for exiting/testing) ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exit requested by user.")
            break
        elif key == ord('b'): # Manual buzzer test
            print("Manual buzzer test triggered.")
            # Use a test pattern different from actual alerts if desired
            trigger_buzzer("TEST", [(0.5, 0.1)])

# --- Exception Handling & Cleanup ---
except KeyboardInterrupt:
    print("\nScript interrupted by user (Ctrl+C).")
except Exception as e:
    print(f"\nAn unexpected error occurred in the main loop: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback for debugging

finally:
    print("Performing cleanup...")
    # Stop the camera
    if 'picam2' in locals() and picam2.is_open:
        picam2.stop()
        print("PiCamera2 stopped.")
    # Close OpenCV windows
    cv2.destroyAllWindows()
    print("OpenCV windows destroyed.")
    # Turn off buzzer and release GPIO pins
    try:
        GPIO.output(BUZZER_PIN, GPIO.LOW) # Ensure buzzer is off
        GPIO.cleanup()
        print("GPIO cleanup complete.")
    except Exception as e:
        print(f"Warning: Error during GPIO cleanup: {e}")
    print("Exiting.")