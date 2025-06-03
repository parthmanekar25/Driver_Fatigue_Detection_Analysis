import cv2
import mediapipe as mp
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter as tflite
from collections import deque
import time
import os
import RPi.GPIO as GPIO # Import GPIO library

# --- Configuration ---
MODEL_DIR = "/Users/parth/Projects/ML/drowsiness/models/" # Update this path if needed for Pi
# Ensure the model path is correct for the Pi's filesystem
YAWN_MODEL_PATH = os.path.join(MODEL_DIR, "best_yawn_model.tflite") # Corrected path join

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0 # Usually 0 for the default Pi camera or USB webcam

# --- GPIO Configuration (Raspberry Pi) ---
GPIO_EYES_PIN = 17    # BCM Pin 17 for Eyes Closed Alert
GPIO_YAWN_PIN = 27    # BCM Pin 27 for Yawn Alert
GPIO_TILT_PIN = 22    # BCM Pin 22 for Head Tilt Alert
GPIO_SEVERE_PIN = 23  # BCM Pin 23 for Severe Alert (Eyes + Yawn)

# Map alert types to pins for easier access
GPIO_PINS = {
    "EYES": GPIO_EYES_PIN,
    "YAWN": GPIO_YAWN_PIN,
    "TILT": GPIO_TILT_PIN,
    "SEVERE": GPIO_SEVERE_PIN
}

gpio_available = False
try:
    GPIO.setmode(GPIO.BCM) # Use Broadcom pin numbering
    GPIO.setwarnings(False) # Disable warnings like "channel already in use"
    for pin in GPIO_PINS.values():
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW) # Set pins as output, initially low
    print("GPIO pins initialized successfully.")
    gpio_available = True
except Exception as e:
    print(f"Warning: Could not initialize GPIO. Alerts will only be printed. Error: {e}")
    gpio_available = False


# MediaPipe Indices (Remain the same)
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 153, 144, 145, 246]
RIGHT_EYE_INDICES = [362, 263, 385, 386, 387, 380, 373, 374, 466]
MOUTH_LANDMARKS = list(set([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13]))

# Buffers & Thresholds (Remain the same)
EAR_BUFFER_LENGTH = 15
EAR_THRESHOLD = 0.2
EYE_CLOSED_BUFFER_LENGTH = 30
EYE_CLOSED_COUNT_THRESHOLD = 6
MAR_BUFFER_LENGTH = 3
CNN_YAWN_BUFFER_LENGTH = 25
MAR_THRESHOLD = 0.6
YAWN_MAR_FRAMES_THRESHOLD = 15
CNN_YAWN_PROB_THRESHOLD = 0.75
CNN_YAWN_FRAMES_TRIGGER = 18
HEAD_TILT_THRESHOLD = 18
DROWSY_RESET_FRAMES = 30
YAWN_COOLDOWN_FRAMES = 40
ALERT_COOLDOWN_SECONDS = 4.0
EYE_BBOX_SCALE = 1.4
YAWN_CNN_INPUT_SIZE = (96, 96)
INFERENCE_INTERVAL = 3
MOUTH_BBOX_SCALE_W = 1.5
MOUTH_BBOX_SCALE_H = 1.6

# Buzzer patterns define ON/OFF durations for GPIO pins
# (On Time, Off Time) pairs
BUZZER_PATTERNS = {
    "EYES": [(0.1, 0.1), (0.1, 0.1)],         # Short pulses
    "YAWN": [(0.4, 0.2)],                     # Longer pulse
    "TILT": [(0.15, 0.1), (0.15, 0.3)],       # Different pattern
    "SEVERE": [(0.2, 0.05), (0.2, 0.05), (0.2, 0.4)] # Rapid pulses
}

last_alert_trigger_time = 0

# --- GPIO Alert Function ---
def trigger_gpio_alert(alert_type, duration_pattern):
    """Activates the corresponding GPIO pin based on the alert type and pattern."""
    global last_alert_trigger_time
    current_time = time.time()

    # Enforce cooldown period
    if current_time < last_alert_trigger_time + ALERT_COOLDOWN_SECONDS:
        return

    print(f"--- ALERT DETECTED: {alert_type} ---") # Keep console log

    if gpio_available and alert_type in GPIO_PINS:
        pin_to_activate = GPIO_PINS[alert_type]
        try:
            # Execute the ON/OFF pattern
            for on_time, off_time in duration_pattern:
                GPIO.output(pin_to_activate, GPIO.HIGH) # Turn pin ON
                time.sleep(on_time)
                GPIO.output(pin_to_activate, GPIO.LOW)  # Turn pin OFF
                if off_time > 0: # Avoid sleep if off_time is 0 or less
                    time.sleep(off_time)
            last_alert_trigger_time = current_time # Update time only after successful alert
        except Exception as e:
            print(f"Error activating GPIO pin {pin_to_activate} for {alert_type}: {e}")
    elif not gpio_available:
        print(f"(GPIO not available for alert: {alert_type})")
    else:
        print(f"(Alert type '{alert_type}' not mapped to a GPIO pin)")

# --- TFLite Model Loading ---
def load_tflite_model(model_path):
    try:
        interpreter = tflite(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"Loaded model: {os.path.basename(model_path)}")
        return interpreter, input_details, output_details
    except Exception as e:
        print(f"Error loading TFLite model {model_path}: {e}")
        exit() # Exit if model can't load

print("Loading TFLite yawn model...")
# Make sure YAWN_MODEL_PATH is correct for your Pi filesystem
yawn_interpreter, yawn_input_details, yawn_output_details = load_tflite_model(YAWN_MODEL_PATH)

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.7,
                                   min_tracking_confidence=0.7)

# --- Helper Functions ---
def enhance_contrast_clahe(image_gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(5, 5))
    return clahe.apply(image_gray)

def preprocess_mouth(mouth_img, input_details):
    if mouth_img is None or mouth_img.size == 0:
        return None
    try:
        # Ensure mouth_img is 3-channel BGR if model expects 3 channels
        input_shape = input_details[0]['shape']
        target_h, target_w = input_shape[1], input_shape[2]
        channels = input_shape[3] if len(input_shape) == 4 else 1

        if channels == 3:
            if len(mouth_img.shape) == 2: # If grayscale, convert to BGR
                 mouth_img_bgr = cv2.cvtColor(mouth_img, cv2.COLOR_GRAY2BGR)
            elif len(mouth_img.shape) == 3 and mouth_img.shape[2] == 1: # If single channel 3D, convert
                 mouth_img_bgr = cv2.cvtColor(mouth_img, cv2.COLOR_GRAY2BGR)
            else:
                 mouth_img_bgr = mouth_img # Assume it's already BGR

            mouth_resized = cv2.resize(mouth_img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            # Normalization might depend on the model training
            mouth_normalized = mouth_resized.astype(np.float32) / 255.0 # Common normalization
            # Add batch dimension
            mouth_input = np.expand_dims(mouth_normalized, axis=0)

        else: # Model expects 1 channel (grayscale)
            if len(mouth_img.shape) == 3: # If BGR, convert to grayscale
                mouth_gray = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)
            else:
                mouth_gray = mouth_img # Assume it's already grayscale

            mouth_eq = enhance_contrast_clahe(mouth_gray) # Apply CLAHE
            mouth_resized = cv2.resize(mouth_eq, (target_w, target_h), interpolation=cv2.INTER_AREA)
            mouth_normalized = mouth_resized.astype(np.float32) / 255.0
            # Add batch and channel dimensions
            mouth_input = np.expand_dims(np.expand_dims(mouth_normalized, axis=-1), axis=0)

        # Check data type expected by the model (often float32, sometimes uint8)
        if input_details[0]['dtype'] == np.uint8:
             mouth_input = (mouth_input * 255).astype(np.uint8) # Convert back if model needs uint8

        return mouth_input

    except Exception as e:
        print(f"Error in preprocess_mouth: {e}")
        return None


def predict_tflite(interpreter, input_details, output_details, data):
    if data is None or interpreter is None:
        return None
    try:
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])
    except Exception as e:
        print(f"Error during TFLite inference: {e}")
        return None

def get_scaled_bbox(landmarks, indices, width, height, scale=1.0, last_bbox=None):
    points = np.array([(landmarks.landmark[i].x * width, landmarks.landmark[i].y * height)
                       for i in indices if 0 <= landmarks.landmark[i].x <= 1 and 0 <= landmarks.landmark[i].y <= 1], dtype=np.int32)

    if len(points) < 3: # Need at least 3 points to define a region
        return last_bbox # Return previous bbox if points are invalid

    try:
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        bbox_w, bbox_h = x_max - x_min, y_max - y_min

        # Handle cases where width or height is zero
        if bbox_w <= 0 or bbox_h <= 0:
            return last_bbox

        scaled_w, scaled_h = bbox_w * scale, bbox_h * scale

        x1 = max(0, int(center_x - scaled_w / 2))
        y1 = max(0, int(center_y - scaled_h / 2))
        x2 = min(width - 1, int(center_x + scaled_w / 2))
        y2 = min(height - 1, int(center_y + scaled_h / 2))

        # Ensure coordinates form a valid rectangle
        if x1 < x2 and y1 < y2:
            return x1, y1, x2, y2
        else:
            return last_bbox # Return previous if scaled box is invalid

    except Exception as e:
        # print(f"Error calculating bbox: {e}") # Optional: for debugging
        return last_bbox # Return previous on any error

# Note on EAR Robustness: The current EAR calculation is standard.
# For potential enhancements (outside the scope of this modification request),
# consider:
# 1. Normalizing EAR based on individual baseline or face size.
# 2. Incorporating head pose information to account for tilted views affecting landmark distances.
# 3. Using more sophisticated eye state classifiers (e.g., small CNNs specifically for eye state).
# However, the current logic remains unchanged as requested.
def calculate_ear(landmarks, indices, width, height):
    """Calculates the Eye Aspect Ratio (EAR) for one eye."""
    try:
        # Vertical distances P2-P6, P3-P5
        vert1 = landmarks.landmark[indices[1]].y * height # P2
        vert2 = landmarks.landmark[indices[7]].y * height # P6
        vert3 = landmarks.landmark[indices[3]].y * height # P3
        vert4 = landmarks.landmark[indices[5]].y * height # P5 (Indices adjusted based on common EAR diagrams, verify yours)

        # Horizontal distance P1-P4
        horiz1 = landmarks.landmark[indices[0]].x * width # P1
        horiz2 = landmarks.landmark[indices[4]].x * width # P4

        ear_num = abs(vert1 - vert2) + abs(vert3 - vert4)
        ear_den = 2.0 * abs(horiz1 - horiz2)

        if ear_den < 1e-6: # Avoid division by zero
            return 1.0 # Return a high value if horizontal distance is negligible

        ear = ear_num / ear_den
        return ear if not np.isnan(ear) and ear >= 0 else 1.0 # Ensure non-negative and not NaN

    except (IndexError, TypeError, AttributeError):
        # print("Error calculating EAR: Landmark index issue.") # Optional debug
        return 1.0 # Return a high value indicating likely open eye or error

def calculate_mar(landmarks, width, height):
    """Calculates the Mouth Aspect Ratio (MAR)."""
    try:
        # Using common landmarks for mouth vertical/horizontal distance
        v_upper_lip = landmarks.landmark[13].y * height # Top inner lip
        v_lower_lip = landmarks.landmark[14].y * height # Bottom inner lip
        h_left_corner = landmarks.landmark[61].x * width # Left mouth corner
        h_right_corner = landmarks.landmark[291].x * width # Right mouth corner

        mouth_height = abs(v_lower_lip - v_upper_lip)
        mouth_width = abs(h_right_corner - h_left_corner)

        if mouth_width < 1e-6: # Avoid division by zero
            return 0.0

        mar = mouth_height / mouth_width
        return mar if not np.isnan(mar) and mar >= 0 else 0.0

    except (IndexError, TypeError, AttributeError):
        # print("Error calculating MAR: Landmark index issue.") # Optional debug
        return 0.0

def detect_head_tilt(landmarks, width, height):
    """Estimates head tilt angle based on the line between inner eye corners."""
    try:
        # Use inner eye corners (landmarks 133 and 362)
        left_eye_inner = landmarks.landmark[133]
        right_eye_inner = landmarks.landmark[362]

        # Basic check if landmarks are valid
        if not (0 <= left_eye_inner.x <= 1 and 0 <= left_eye_inner.y <= 1 and
                0 <= right_eye_inner.x <= 1 and 0 <= right_eye_inner.y <= 1):
            return 0.0 # Return neutral if landmarks seem off-screen

        dx = (right_eye_inner.x - left_eye_inner.x) * width
        dy = (right_eye_inner.y - left_eye_inner.y) * height

        if abs(dx) < 1e-6: # Avoid division by zero if eyes are vertically aligned (unlikely)
             return 90.0 if dy > 0 else -90.0 if dy < 0 else 0.0

        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    except (IndexError, TypeError, AttributeError):
         # print("Error detecting head tilt: Landmark index issue.") # Optional debug
        return 0.0

# --- State Variables ---
ear_buffer = deque(maxlen=EAR_BUFFER_LENGTH)
mar_buffer = deque(maxlen=MAR_BUFFER_LENGTH)
yawn_mar_trigger_buffer = deque(maxlen=YAWN_MAR_FRAMES_THRESHOLD)
eye_closed_buffer = deque(maxlen=EYE_CLOSED_BUFFER_LENGTH)
cnn_yawn_buffer = deque(maxlen=CNN_YAWN_BUFFER_LENGTH)
drowsy = False
yawning = False
head_tilt_detected = False
frame_count = 0
consecutive_open_eye_frames = 0
yawn_cooldown_counter = 0
last_cnn_yawn_prob = 0.0
last_left_bbox = last_right_bbox = last_mouth_bbox = None

# --- Webcam Initialization ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open camera {CAMERA_INDEX}.")
    if gpio_available: GPIO.cleanup() # Cleanup GPIO if exiting early
    exit()

# Try setting resolution, but use actual if it differs
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if actual_width != FRAME_WIDTH or actual_height != FRAME_HEIGHT:
    print(f"Warning: Camera does not support {FRAME_WIDTH}x{FRAME_HEIGHT}. Using {actual_width}x{actual_height}")
    FRAME_WIDTH, FRAME_HEIGHT = actual_width, actual_height
else:
    print(f"Camera resolution set to: {actual_width}x{actual_height}")

# --- Main Loop ---
print("Starting detection loop... Press 'q' to quit.")
try:
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Warning: Failed to grab frame, retrying...")
            time.sleep(0.1)
            continue

        # Image processing
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False # Improve performance
        results = face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True # Make writeable again if needed later

        frame_count += 1
        run_inference_this_frame = (frame_count % INFERENCE_INTERVAL == 0)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0] # Process only the first detected face

            # --- Calculations ---
            current_mar = calculate_mar(face_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
            mar_buffer.append(current_mar)
            smoothed_mar = np.mean(list(mar_buffer)) if mar_buffer else 0.0

            head_angle = detect_head_tilt(face_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
            head_tilt_detected = abs(head_angle) > HEAD_TILT_THRESHOLD

            # --- Bounding Boxes (using last known if current fails) ---
            current_left_bbox = get_scaled_bbox(face_landmarks, LEFT_EYE_INDICES, FRAME_WIDTH, FRAME_HEIGHT, EYE_BBOX_SCALE, last_left_bbox)
            current_right_bbox = get_scaled_bbox(face_landmarks, RIGHT_EYE_INDICES, FRAME_WIDTH, FRAME_HEIGHT, EYE_BBOX_SCALE, last_right_bbox)

            mouth_points = np.array([[int(face_landmarks.landmark[i].x * FRAME_WIDTH), int(face_landmarks.landmark[i].y * FRAME_HEIGHT)]
                                     for i in MOUTH_LANDMARKS if 0 <= face_landmarks.landmark[i].x <= 1 and 0 <= face_landmarks.landmark[i].y <= 1], dtype=np.int32)
            current_mouth_bbox = None
            if mouth_points.shape[0] > 2: # Need points for boundingRect
                try:
                    x_m, y_m, w_m, h_m = cv2.boundingRect(mouth_points)
                    if w_m > 0 and h_m > 0: # Ensure valid width/height
                        center_x, center_y = x_m + w_m / 2, y_m + h_m / 2
                        scaled_w_m, scaled_h_m = w_m * MOUTH_BBOX_SCALE_W, h_m * MOUTH_BBOX_SCALE_H
                        mx1, my1 = max(0, int(center_x - scaled_w_m / 2)), max(0, int(center_y - scaled_h_m / 2))
                        mx2, my2 = min(FRAME_WIDTH - 1, int(center_x + scaled_w_m / 2)), min(FRAME_HEIGHT - 1, int(center_y + scaled_h_m / 2))
                        if mx1 < mx2 and my1 < my2:
                            current_mouth_bbox = (mx1, my1, mx2, my2)
                        else:
                             current_mouth_bbox = last_mouth_bbox # Use last if calculation failed
                    else:
                        current_mouth_bbox = last_mouth_bbox
                except Exception:
                     current_mouth_bbox = last_mouth_bbox # Use last on error
            else:
                current_mouth_bbox = last_mouth_bbox # Use last if not enough points

            # Update last known good bounding boxes
            if current_left_bbox: last_left_bbox = current_left_bbox
            if current_right_bbox: last_right_bbox = current_right_bbox
            if current_mouth_bbox: last_mouth_bbox = current_mouth_bbox

            # --- EAR Calculation ---
            left_ear = calculate_ear(face_landmarks, LEFT_EYE_INDICES, FRAME_WIDTH, FRAME_HEIGHT)
            right_ear = calculate_ear(face_landmarks, RIGHT_EYE_INDICES, FRAME_WIDTH, FRAME_HEIGHT)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_buffer.append(avg_ear)
            smoothed_ear = np.mean(list(ear_buffer)) if ear_buffer else avg_ear

            both_eyes_closed = smoothed_ear < EAR_THRESHOLD
            eye_closed_buffer.append(both_eyes_closed)

            # --- Yawn CNN Prediction (throttled by INFERENCE_INTERVAL) ---
            cnn_yawn_prob = last_cnn_yawn_prob # Assume same as last unless inference runs
            if run_inference_this_frame and last_mouth_bbox:
                mx1, my1, mx2, my2 = last_mouth_bbox
                if my1 < my2 and mx1 < mx2: # Check if bbox is valid
                    mouth_img = frame_bgr[my1:my2, mx1:mx2]
                    mouth_input = preprocess_mouth(mouth_img, yawn_input_details)
                    if mouth_input is not None:
                        yawn_output = predict_tflite(yawn_interpreter, yawn_input_details, yawn_output_details, mouth_input)
                        if yawn_output is not None and len(yawn_output) > 0 and len(yawn_output[0]) > 0:
                             # Assuming model output is [[probability]]
                             cnn_yawn_prob = float(yawn_output[0][0])
                             last_cnn_yawn_prob = cnn_yawn_prob # Store the latest prediction
                        #else: print("Warning: Invalid yawn prediction output") # Optional debug
                #else: print("Warning: Invalid mouth bbox for CNN") # Optional debug

            cnn_yawn_buffer.append(cnn_yawn_prob > CNN_YAWN_PROB_THRESHOLD)
            yawn_mar_trigger_buffer.append(smoothed_mar > MAR_THRESHOLD)

            # --- Drowsiness Logic ---
            # Check recent frames for prolonged closure
            closed_count = sum(list(eye_closed_buffer)[-EYE_CLOSED_COUNT_THRESHOLD:]) # Count in the relevant window
            if not drowsy and closed_count >= EYE_CLOSED_COUNT_THRESHOLD:
                drowsy = True
                consecutive_open_eye_frames = 0
                print("Drowsiness detected (eyes closed)")
            elif drowsy and not both_eyes_closed:
                consecutive_open_eye_frames += 1
                if consecutive_open_eye_frames >= DROWSY_RESET_FRAMES:
                    drowsy = False
                    eye_closed_buffer.clear() # Clear buffer on recovery
                    consecutive_open_eye_frames = 0
                    print("Drowsiness reset (eyes open)")
            elif drowsy and both_eyes_closed:
                 consecutive_open_eye_frames = 0 # Reset counter if eyes close again

            # --- Yawn Logic ---
            cnn_yawn_count = sum(cnn_yawn_buffer)
            is_cnn_yawning = cnn_yawn_count >= CNN_YAWN_FRAMES_TRIGGER

            mar_yawn_count = sum(yawn_mar_trigger_buffer)
            is_mar_yawning_trigger = (yawn_cooldown_counter == 0 and mar_yawn_count >= YAWN_MAR_FRAMES_THRESHOLD)

            if is_mar_yawning_trigger:
                print("Yawn detected (MAR trigger)")
                yawn_cooldown_counter = YAWN_COOLDOWN_FRAMES # Start cooldown
                yawn_mar_trigger_buffer.clear() # Reset MAR trigger buffer

            if yawn_cooldown_counter > 0:
                yawn_cooldown_counter -= 1

            # Combine CNN and MAR-based yawn detection
            # Consider CNN yawn more definitive if detected
            if is_cnn_yawning and not yawning:
                 print("Yawn detected (CNN)")
                 yawning = True
            elif is_mar_yawning_trigger:
                 yawning = True
            elif yawning and not is_cnn_yawning and not is_mar_yawning_trigger and yawn_cooldown_counter == 0:
                 # Only reset if both triggers are off AND cooldown finished
                 # Might need refinement - how long should 'yawning' state persist?
                 # Maybe reset yawning based on cnn_yawn_buffer being empty?
                 if sum(cnn_yawn_buffer) == 0: # Reset if CNN buffer is clear
                      yawning = False
                      print("Yawn state reset")


            # --- Alerts ---
            alert_now = False
            alert_type = "NONE"
            pattern = []

            # Prioritize alerts: Severe > Yawn > Drowsy > Tilt
            if yawning and drowsy:
                alert_type = "SEVERE"
                pattern = BUZZER_PATTERNS["SEVERE"]
                alert_now = True
            elif yawning:
                alert_type = "YAWN"
                pattern = BUZZER_PATTERNS["YAWN"]
                alert_now = True
            elif drowsy:
                alert_type = "EYES"
                pattern = BUZZER_PATTERNS["EYES"]
                alert_now = True
            elif head_tilt_detected:
                alert_type = "TILT"
                pattern = BUZZER_PATTERNS["TILT"]
                alert_now = True

            if alert_now:
                trigger_gpio_alert(alert_type, pattern) # Call the GPIO alert function

            # --- Drawing / Display ---
            # Draw eye boxes
            if last_left_bbox:
                lx1, ly1, lx2, ly2 = last_left_bbox
                eye_color = (0, 0, 255) if both_eyes_closed else (0, 255, 0)
                cv2.rectangle(frame_bgr, (lx1, ly1), (lx2, ly2), eye_color, 1)
                cv2.putText(frame_bgr, f"L_EAR: {left_ear:.2f}", (lx1, ly1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if last_right_bbox:
                rx1, ry1, rx2, ry2 = last_right_bbox
                eye_color = (0, 0, 255) if both_eyes_closed else (0, 255, 0)
                cv2.rectangle(frame_bgr, (rx1, ry1), (rx2, ry2), eye_color, 1)
                cv2.putText(frame_bgr, f"R_EAR: {right_ear:.2f}", (rx1, ly1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1) # Position relative to left eye text

            # Draw mouth box
            if last_mouth_bbox:
                mx1, my1, mx2, my2 = last_mouth_bbox
                mouth_color = (0, 255, 255) if yawning else (255, 255, 0) # Yellow if yawning, Cyan otherwise
                cv2.rectangle(frame_bgr, (mx1, my1), (mx2, my2), mouth_color, 1)
                cv2.putText(frame_bgr, f"CNN: {cnn_yawn_prob:.2f}", (mx1, my2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


            # Status Display
            info_y_start = 30
            status_text = "Status: Awake"
            status_color = (0, 255, 0) # Green

            if alert_type == "SEVERE":
                status_text = "Status: SEVERE DROWSINESS!"
                status_color = (0, 0, 255) # Red
            elif alert_type == "YAWN":
                 status_text = "Status: Yawning"
                 status_color = (0, 255, 255) # Yellow
            elif alert_type == "EYES":
                 status_text = "Status: Eyes Closed (Drowsy)"
                 status_color = (0, 165, 255) # Orange
            elif alert_type == "TILT":
                 status_text = "Status: Head Tilt"
                 status_color = (255, 0, 255) # Magenta


            cv2.putText(frame_bgr, status_text, (10, info_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame_bgr, f"Avg EAR: {smoothed_ear:.2f} {'(Closed)' if both_eyes_closed else ''}", (10, info_y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_bgr, f"MAR: {smoothed_mar:.2f}", (10, info_y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_bgr, f"Tilt Angle: {head_angle:.1f}", (10, info_y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if head_tilt_detected else (255,255,255), 1)

        else:
            # No face detected - Reset states? Or maintain last known?
            # For safety, maybe trigger a warning if face is lost for too long.
            # Simple reset for now:
            if any([drowsy, yawning, head_tilt_detected]):
                 print("Face lost, resetting state.")
                 drowsy = yawning = head_tilt_detected = False
                 ear_buffer.clear()
                 mar_buffer.clear()
                 yawn_mar_trigger_buffer.clear()
                 eye_closed_buffer.clear()
                 cnn_yawn_buffer.clear()
                 consecutive_open_eye_frames = yawn_cooldown_counter = 0
                 last_cnn_yawn_prob = 0.0
                 # Keep last known bboxes? Or clear them? Cleared bboxes might be safer.
                 last_left_bbox = last_right_bbox = last_mouth_bbox = None

            cv2.putText(frame_bgr, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # Display the frame
        cv2.imshow("Driver Fatigue Detection (Pi)", frame_bgr)

        # Quit logic
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit key pressed.")
            break

except KeyboardInterrupt:
    print("Interrupted by user (Ctrl+C).")

finally:
    # Release resources
    print("Releasing resources...")
    if cap and cap.isOpened():
        cap.release()
        print("Camera released.")
    cv2.destroyAllWindows()
    print("OpenCV windows closed.")
    if gpio_available:
        GPIO.cleanup() # Important: Clean up GPIO channels
        print("GPIO cleaned up.")
    print("Cleanup complete.")